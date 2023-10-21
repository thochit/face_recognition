import numpy as np

def recognize_face_SVM(faces, facenet, svm_classifier, threshold):
    '''
    Recognize faces using a Support Vector Machine (SVM) classifier.
    Input:
        faces: torch.Tensor, face images
        facenet: InceptionResnetV1, pretrained face detection model
        svm_classifier: trained SVM classifier
        threshold: float, threshold for face recognition confidence
    Output:
        predicted_identity: str, predicted identity
        prediction_prob: float, prediction probability
    '''

    face_embedding = facenet(faces.unsqueeze(0)).detach().cpu().numpy()

    # Predict the label using the SVM classifier
    predicted_identities = svm_classifier.predict(face_embedding)
    probabilities = svm_classifier.predict_proba(face_embedding)

    prediction_prob = np.round(np.max(probabilities[0]), 2)
    predicted_identity = predicted_identities[0]

    # Apply a threshold to face recognition
    if prediction_prob < threshold:
        predicted_identity = 'Unknown'

    return predicted_identity, prediction_prob

def recognize_face_faiss(faces, facenet, names, index, threshold):
    '''
    Recognize faces using Faiss for fast similarity search.
    Input:
        faces: torch.Tensor, face images
        facenet: InceptionResnetV1, pretrained face detection model
        names: list, list of names corresponding to embeddings in the Faiss index
        index: Faiss index for fast similarity search
        threshold: float, threshold for face recognition confidence
    Output:
        predicted_identity: str, predicted identity
        distance: float, distance to the recognized face
    '''

    face_embedding = facenet(faces.unsqueeze(0)).detach().cpu().numpy()

    distances, indices = index.search(face_embedding, 1)

    predicted_identity = names[indices[0][0]]

    # Apply a threshold to face recognition
    if distances[0][0] < threshold:
        predicted_identity = 'Unknown'

    return predicted_identity, distances[0][0]
