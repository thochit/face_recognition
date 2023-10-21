import numpy as np
import cv2 as cv
import os
from keras.models import load_model
import faiss
import pickle
from PIL import Image
from unidecode import unidecode
from shutil import copy2
import torch
from sklearn.svm import SVC
import joblib
from detect_face import *

def is_image(file_path:str):
    '''
    Check if a file is an image by attempting to open it with the PIL library.
    Input:
        file_path: str, path to the file
    Output:
        True if the file is an image, False otherwise
    '''

    try:
        with Image.open(file_path) as img:
            return True
    except Exception as e:
        return False

def is_bbox_within_frame(x, y, w, h, frame_width, frame_height):
    '''
    Check if a bounding box is entirely within the frame.
    Input:
        x, y, w, h: int, coordinates and dimensions of the bounding box
        frame_width, frame_height: int, dimensions of the frame
    Output:
        True if the bounding box is within the frame, False otherwise
    '''

    if x >= 0 and y >= 0:
        if w <= frame_width and h <= frame_height:
            return True
    return False

def list_images(root_path:str):
    '''
    Retrieve a list of every image path in a folder.
    Input:
        root_path: str, folder path 
    Output:
        A list of images path
    '''

    image_path_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            image_path = os.path.join(root, file)
            if is_image(image_path):
                image_path_list.append(image_path)
    return image_path_list

def list_videos(root_path:str):
    '''
    Retrieve a list of every video path in a folder.
    Input:
        root_path: str, folder path 
    Output:
        A list of video paths
    '''

    video_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            video_path = os.path.join(root, file)
            if not is_image(video_path):
                video_paths.append(video_path)

    return video_paths

def copy_files(root_path:str):
    '''
    Copy files from a given folder to a new folder with filenames converted to ASCII.
    Input:
        root_path: str, source folder path 
    '''

    for root, dirs, files in os.walk(root_path):
        for file in files:
            image_path = os.path.join(root, file)
            new_image_path = 'new_' + unidecode(image_path)
            root_unidecode = 'new_' + unidecode(root)
            if not os.path.exists(root_unidecode):
                os.makedirs(root_unidecode)
            copy2(image_path, new_image_path)

def clear_folder(folder_path:str):
    '''
    Clear all files in a given folder.
    Input:
        folder_path: str, path to the folder to be cleared
    '''

    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Iterate over the files and remove each one
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def create_training_data(folder_path: str, detect_method = 'mediapipe'):
    '''
    Create training data for face recognition by processing images and videos in a given folder.
    Input:
        folder_path: str, path to the folder containing images and videos
        detect_method: str, face detection method ('mediapipe', etc.)
    '''

    dict_image = {}
    for root, dirs, files in os.walk(folder_path):
        new_root = 'training_data'
        if not os.path.exists(new_root):
            os.makedirs(new_root)
        for file in files:
            file_path = os.path.join(root, file)
            name = file_path.split('/')[1]
            name_folder_path = os.path.join(new_root, name)
            if not os.path.exists(name_folder_path):
                os.makedirs(name_folder_path)
            if is_image(file_path):
                image = cv.imread(file_path)
                if len(detect_frame(frame, detect_method)) == 1:
                    if name not in dict_image:
                        dict_image[name] = 1
                    else:
                        dict_image[name] += 1
                    cv.imwrite(name_folder_path + f'/image_{dict_image[name]}.jpg', image)
            else:
                cap = cv.VideoCapture(file_path)
                rotate = False
                if file_path.endswith('MOV'):
                    rotate = True
                while True:
                    ret, frame = cap.read()
                    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

                    if rotate:
                        frame = cv.rotate(frame, cv.ROTATE_180)
                    if not ret:
                        break
                    
                    face_bouding_boxes = detect_frame(frame, detect_method)
                    if len(face_bouding_boxes) != 1:
                        continue
                    for x, y, w, h in face_bouding_boxes:
                        if not is_bbox_within_frame(x, y, w, h, frame_width, frame_height):
                            continue
                        if name not in dict_image:
                            dict_image[name] = 1
                        else:
                            dict_image[name] += 1
                        cv.imwrite(name_folder_path + f'/image_{dict_image[name]}.jpg', frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
                cv.destroyAllWindows()

def create_dataset(folder_path: str) -> dict:
    '''
    Create a dataset by organizing images into a dictionary with names as keys and lists of image paths as values.
    Input:
        folder_path: str, path to the folder containing images
    Output:
        A dictionary containing names as keys and lists of image paths as values
    '''

    dataset = {}
    for name in os.listdir(folder_path):
        dataset[name] = []
        sub_folder = os.path.join(folder_path, name)
        for image_idx in os.listdir(sub_folder):
            image_path = os.path.join(sub_folder, image_idx)
            dataset[name].append(image_path)
    return dataset

def create_database_faiss(folder_path, model, faiss_path, names_path, detect_method: str):
    '''
    Create a face database using Faiss for fast similarity search.
    Input:
        folder_path: str, path to the folder containing images
        model: pretrained face detection model
        faiss_path: str, path to save the Faiss index
        names_path: str, path to save the list of names
        detect_method: str, face detection method ('mediapipe', etc.)
    '''

    # Create dataset;
    dataset = create_dataset(folder_path)

    database_names = []
    database_face_vectors = []

    for name in dataset:
        for image_path in dataset[name]:
            image = cv.imread(image_path)
            height, width, _ = image.shape
            boxes = detect_frame(image, detect_method)
            if len(boxes) != 1:
                continue
            x, y, w, h = boxes[0]
            if not is_bbox_within_frame(x, y, w, h, width, height):
                continue
            try:
                image_face = preprocess_face(image, boxes[0])
            except:
                continue
            image_face = torch.unsqueeze(image_face, 0)
            face_embedding = model(image_face).detach().cpu().numpy()
            face_embedding = face_embedding.reshape(-1)
            database_face_vectors.append(face_embedding)
            database_names.append(name)

    database_face_vectors = np.array(database_face_vectors).astype('float32')
    vector_dimension = database_face_vectors.shape[1]
    index = faiss.IndexFlatIP(vector_dimension)
    index.add(database_face_vectors)
    with open(names_path, 'wb') as file:
        pickle.dump(database_names, file)
    faiss.write_index(index, faiss_path)
    print('Save faiss')

def create_database_SVM(folder_path, model, svm_path_file, detect_method: str):
    '''
    Create a face database using a Support Vector Machine (SVM) classifier.
    Input:
        folder_path: str, path to the folder containing images
        model: pretrained face detection model
        svm_path_file: str, path to save the trained SVM model
        detect_method: str, face detection method ('mediapipe', etc.)
    '''

    # Create dataset;
    dataset = create_dataset(folder_path)

    database_names = []
    database_face_vectors = []

    for name in dataset:
        for image_path in dataset[name]:
            image = cv.imread(image_path)
            height, width, _ = image.shape
            boxes = detect_frame(image, detect_method)
            if len(boxes) != 1:
                continue
            x, y, w, h = boxes[0]
            if not is_bbox_within_frame(x, y, w, h, width, height):
                continue
            try:
                image_face = preprocess_face(image, boxes[0])
            except:
                continue
            image_face = torch.unsqueeze(image_face, 0)
            face_embedding = model(image_face).detach().cpu().numpy()
            face_embedding = face_embedding.reshape(-1)
            database_face_vectors.append(face_embedding)
            database_names.append(name)

    database_face_vectors = np.array(database_face_vectors).astype('float32')

    # Train an SVM classifier
    svm_classifier = SVC(kernel='linear', C=1.0, probability=True)
    svm_classifier.fit(database_face_vectors, database_names)

    # Save the trained SVM model to a file
    joblib.dump(svm_classifier, svm_path_file)
    print('Save SVM')
