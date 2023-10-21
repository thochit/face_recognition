import time
import random
from utils_pytorch import *
from recognize_face import *
from facenet_pytorch import InceptionResnetV1

def add_identity(folder_path: str, name: str, detect_method: str, num_image=5):
    '''
    Collect a specified number of images for a new identity using the webcam.
    Input:
        folder_path: str, path to the folder to save images
        name: str, name of the person
        detect_method: str, face detection method ('mediapipe', etc.)
        num_image: int, number of images to collect (default is 5)
    '''

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    person_dir = os.path.join(folder_path, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    image_count = 0
    cap = cv.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        face = detect_frame(frame, detect_method)

        if cv.waitKey(1) & 0xFF == ord('c'):
            if len(face) == 1:
                image_count += 1
                filename = os.path.join(person_dir, f'{name}_{image_count}.jpg') 
                print(filename)
                cv.imwrite(filename, frame)
            else:
                cv.putText(frame, "More than one face in frame", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.putText(frame, "Press c to collect 1 image", (10, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.putText(frame, f"Number of images: {image_count}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow('Capture image', frame)

        if cv.waitKey(1) & 0xFF == ord('q') or image_count == num_image:
             break
        
    cap.release()
    cv.destroyAllWindows()

def webcam_demo_SVM(load_clf: bool, dataset_path:str, threshold:str, svm_filename: str, detect_method: str):
    '''
    Perform real-time face recognition using a Support Vector Machine (SVM) classifier and a webcam.
    Input:
        load_clf: bool, whether to load a pretrained classifier or create a new one
        dataset_path: str, path to the folder containing training images
        threshold: float, threshold for face recognition confidence
        svm_filename: str, path to save or load the trained SVM model
        detect_method: str, face detection method ('mediapipe', etc.)
    '''

    # Load model
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    if not load_clf:
        create_database_SVM(dataset_path, facenet, svm_filename, detect_method)
    
    svm_classifier = joblib.load(svm_filename)

    cap = cv.VideoCapture(0)
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        if not ret:
            break
        
        face_bouding_boxes = detect_frame(frame, detect_method)
        for i, (x, y, w, h) in enumerate(face_bouding_boxes):
            if not is_bbox_within_frame(x, y, w, h, frame_width, frame_height):
                continue
            face = preprocess_face(frame, face_bouding_boxes[i])
            face_name, score = recognize_face_SVM(face, facenet, svm_classifier, threshold)
            cv.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv.putText(frame, f"Name: {face_name}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.putText(frame, f"Score: {score}", (x, y - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.imshow('Webcam', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def webcam_demo_faiss(load_clf: bool, dataset_path:str, threshold:str, faiss_path: str, names_path:str, detect_method: str):
    '''
    Perform real-time face recognition using Faiss for fast similarity search and a webcam.
    Input:
        load_clf: bool, whether to load a pretrained index or create a new one
        dataset_path: str, path to the folder containing training images
        threshold: float, threshold for face recognition confidence
        faiss_path: str, path to save or load the Faiss index
        names_path: str, path to save or load the list of names
        detect_method: str, face detection method ('mediapipe', etc.)
    '''

    # Load model
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    if not load_clf:
        create_database_faiss(dataset_path, facenet, faiss_path, names_path, detect_method)

    index = faiss.read_index(faiss_path)
    with open(names_path, 'rb') as file:
            names = pickle.load(file)

    cap = cv.VideoCapture(0)
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        if not ret:
            break
        
        face_bouding_boxes = detect_frame(frame, detect_method)
        for i, (x, y, w, h) in enumerate(face_bouding_boxes):
            if not is_bbox_within_frame(x, y, w, h, frame_width, frame_height):
                continue
            face = preprocess_face(frame, face_bouding_boxes[i])
            face_name, score = recognize_face_faiss(face, facenet, names, index, threshold)
            cv.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv.putText(frame, f"Name: {face_name}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.putText(frame, f"Score: {score:.2f}", (x, y - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.imshow('Webcam', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def video_demo_svm(video_path:str, facenet, svm_classifier, threshold:float, detect_method:str):
    '''
    Perform face recognition on a video using a Support Vector Machine (SVM) classifier.
    Input:
        video_path: str, path to the video file
        facenet: InceptionResnetV1, pretrained face detection model
        svm_classifier: trained SVM classifier
        threshold: float, threshold for face recognition confidence
        detect_method: str, face detection method ('mediapipe', etc.)
    '''

    cap = cv.VideoCapture(video_path)
    rotate = False
    if video_path.lower().endswith('mov') or video_path.endswith('MOV'):
        print('MOV')
        rotate = True
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()

    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0

    if frame_height > 1080:
        # Define the new height you want
        new_height = 1080  # You can set it to any value you prefer

        # Calculate the aspect ratio to maintain the original aspect ratio
        aspect_ratio = frame_width / frame_height
        new_width = int(new_height * aspect_ratio)
    
    while True:
        ret, frame = cap.read()
        if rotate:
            frame = cv.rotate(frame, cv.ROTATE_180)

        if frame_height > 1080:
            frame = cv.resize(frame, (new_width, new_height))

        if not ret:
            break
        
        face_bouding_boxes = detect_frame(frame, detect_method)
        for i, (x, y, w, h) in enumerate(face_bouding_boxes):
            if not is_bbox_within_frame(x, y, w, h, frame_width, frame_height):
                continue
            face = preprocess_face(frame, face_bouding_boxes[i])
            face_name, score = recognize_face_SVM(face, facenet, svm_classifier, threshold)
            cv.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv.putText(frame, f"Name: {face_name}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.putText(frame, f"Score: {score}", (x, y - 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.imshow('Webcam', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def video_demo_faiss(video_path:str, facenet, index, names, threshold:float, detect_method:str):
    '''
    Perform face recognition on a video using Faiss for fast similarity search.
    Input:
        video_path: str, path to the video file
        facenet: InceptionResnetV1, pretrained face detection model
        index: Faiss index for fast similarity search
        names: list, list of names corresponding to embeddings in the Faiss index
        threshold: float, threshold for face recognition confidence
        detect_method: str, face detection method ('mediapipe', etc.)
    '''

    cap = cv.VideoCapture(video_path)
    rotate = False
    if video_path.lower().endswith('mov') or video_path.endswith('MOV'):
        print('MOV')
        rotate = True
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()

    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0

    if frame_height > 1080:
        # Define the new height you want
        new_height = 1080  # You can set it to any value you prefer

        # Calculate the aspect ratio to maintain the original aspect ratio
        aspect_ratio = frame_width / frame_height
        new_width = int(new_height * aspect_ratio)
    
    while True:
        ret, frame = cap.read()
        if rotate:
            frame = cv.rotate(frame, cv.ROTATE_180)

        if frame_height > 1080:
            frame = cv.resize(frame, (new_width, new_height))

        if not ret:
            break
        
        face_bouding_boxes = detect_frame(frame, detect_method)
        for i, (x, y, w, h) in enumerate(face_bouding_boxes):
            if not is_bbox_within_frame(x, y, w, h, frame_width, frame_height):
                continue
            face = preprocess_face(frame, face_bouding_boxes[i])
            face_name, score = recognize_face_faiss(face, facenet, names, index, threshold)
            cv.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv.putText(frame, f"Name: {face_name}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.putText(frame, f"Score: {score:.2f}", (x, y - 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.imshow('Webcam', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def folder_demo(dataset_path, load_clf:bool, clf_method:str, threshold:float, svm_path_file:str, faiss_path:str, names_path:str, detect_method:str):
    '''
    Perform face recognition on videos in a folder using either SVM or Faiss for fast similarity search.
    Input:
        dataset_path: str, path to the folder containing training images
        load_clf: bool, whether to load a pretrained classifier/index or create a new one
        clf_method: str, classifier method ('svm' or 'faiss')
        threshold: float, threshold for face recognition confidence
        svm_path_file: str, path to save or load the trained SVM model
        faiss_path: str, path to save or load the Faiss index
        names_path: str, path to save or load the list of names
        detect_method: str, face detection method ('mediapipe', etc.)
    '''

    facenet = InceptionResnetV1(pretrained='vggface2').eval()

    if not load_clf:
        if clf_method == 'svm':
            create_database_SVM(dataset_path, facenet, svm_path_file, detect_method)
        else:
            create_database_faiss(dataset_path, facenet, faiss_path, names_path, detect_method)

    if clf_method == 'svm':
        svm_classifier = joblib.load(svm_path_file)
    else:
        index = faiss.read_index(faiss_path)
        with open(names_path, 'rb') as file:
                names = pickle.load(file)

    video_paths = list_videos(dataset_path)
    for _ in video_paths:
        video_path = random.choice(video_paths)
        if clf_method == 'svm':
            video_demo_svm(video_path, facenet, svm_classifier, threshold, detect_method)
        else:
            video_demo_faiss(video_path, facenet, index, names, threshold, detect_method)
