import mediapipe as mp
import os
from facenet_pytorch import MTCNN
import cv2 as cv
import numpy as np
from facenet_pytorch import extract_face, fixed_image_standardization

# Initialize MediaPipe Face Detection and Facial Landmarks
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
    
file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
threshold = 0.5  # human face's confidence threshold

# Load model ResnetSSD
prototxt_file = file_path + 'model/Resnet_SSD_deploy.prototxt'
caffemodel_file = file_path + 'model/Res10_300x300_SSD_iter_140000.caffemodel'
net = cv.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file)
print('MobileNetSSD caffe model loaded successfully')
detector_mtcnn = MTCNN()

def detect_frame_mp(image) -> list:
    '''
    Detect faces in an image using MediaPipe Face Mesh.
    Input:
        image: numpy array, input image
    Output:
        List of bounding boxes [x, y, w, h] for detected faces
    '''

    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:

        ih, iw, _ = image.shape
        # Convert the frame to RGB format (MediaPipe expects RGB images)
        frame_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Perform facial landmark detection
        results_landmarks = face_mesh.process(frame_rgb)
        if results_landmarks.multi_face_landmarks:
            for landmarks in results_landmarks.multi_face_landmarks:
                landmark_points = []
                for landmark in landmarks.landmark:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    landmark_points.append((x, y, z))

                # Calculate bounding box around the facial landmarks
                min_x, max_x, min_y, max_y = iw, 0, ih, 0
                for x, y, z in landmark_points:
                    x_pixel, y_pixel = int(x * iw), int(y * ih)
                    if x_pixel < min_x:
                        min_x = x_pixel
                    if x_pixel > max_x:
                        max_x = x_pixel
                    if y_pixel < min_y:
                        min_y = y_pixel
                    if y_pixel > max_y:
                        max_y = y_pixel

    try:
        return [[min_x, min_y, max_x, max_y]]
    except:
        return []

def detect_frame_ssd(image) -> list:
    '''
    Detect faces in an image using a Single Shot Multibox Detector (SSD).
    Input:
        image: numpy array, input image
    Output:
        List of bounding boxes [x, y, w, h] for detected faces
    '''

    origin_h, origin_w = image.shape[:2]

    blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
            x_start, y_start, x_end, y_end = bounding_box.astype('int')
            faces.append((x_start, y_start, x_end, y_end))

    return faces

def detect_frame_MTCNN(image) -> list:
    '''
    Detect faces in an image using MTCNN (Multi-task Cascaded Convolutional Networks).
    Input:
        image: numpy array, input image
    Output:
        List of bounding boxes [x, y, w, h] for detected faces
    '''

    faces, _ = detector_mtcnn.detect(image)
    if faces is None:
        faces = []
    faces = [face.astype('int') for face in faces]
    return faces

def detect_frame(image, method) -> list:
    '''
    Detect faces in an image using specified method.
    Input:
        image: numpy array, input image
        method: str, face detection method ('mediapipe', 'ssd', 'mtcnn')
    Output:
        List of bounding boxes [x, y, w, h] for detected faces
    '''

    if method == 'mediapipe':
        return detect_frame_mp(image)
    elif method == 'ssd':
        return detect_frame_ssd(image)
    else:
        return detect_frame_MTCNN(image)

def preprocess_face(image, box):
    '''
    Extract and preprocess the face from the image using specified bounding box.
    Input:
        image: numpy array, input image
        box: list, bounding box [x, y, w, h] around the face
    Output:
        preprocessed face
    '''

    face = extract_face(image, box, 160, 0)
    face = fixed_image_standardization(face)
    return face
