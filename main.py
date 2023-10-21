import argparse
from facenet_pytorch import InceptionResnetV1
from utils_pytorch import *
from demo import *

parser = argparse.ArgumentParser(description="Demo Model")
parser.add_argument('-f', "--dataset-path", type=str, help="Enter your dataset path")
parser.add_argument('-s', "--image-size", type=int, help="Enter your image size", default=224)
parser.add_argument('-v', "--video-path", type=str, help="Enter your video path")
parser.add_argument('-n', '--name', type=str, help='Enter name')
parser.add_argument('-i', '--num-image', type=int, help='Enter number of images to collect', default=5)
parser.add_argument('-m', "--mode", type=str, help='Enter which mode you wnat(webcam_demo or add_identity or folder_demo or create_database)', default='webcam_demo')
parser.add_argument('-lc', '--load-clf', type=str, help = 'Load trained file classify or not', default=True)
parser.add_argument('-svm', '--svm-model', type=str, help='Path to svm model file', default='model/svm_model.joblib')
parser.add_argument('-faiss', '--faiss-path', type=str, help='Path to faiss model file', default='model/vector_search.index')
parser.add_argument('-ns', '--names-path', type=str, help='Path to names file', default='model/names.pkl')
parser.add_argument('-clf', '--classify-method', type=str, help='Name of the classify method(svm or faiss)', default='svm')
parser.add_argument('-dm', '--detect-method', type=str, help = 'Detect method using(mediapipe, mtcnn, ssd)', default='mediapipe')
args = parser.parse_args()



if __name__ == '__main__':
    # #TEST;
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    if args.mode == 'webcam_demo':
        if args.classify_method == 'svm':
            webcam_demo_SVM(args.load_clf, args.dataset_path, 0.5, args.svm_model, args.detect_method)
        else:
            webcam_demo_faiss(args.load_clf, args.dataset_path, 0.5, args.faiss_path, args.names_path, args.detect_method)
    elif args.mode == 'folder_demo':
        folder_demo(args.dataset_path, args.load_clf, args.classify_method, 0.5, args.svm_model, args.faiss_path, args.names_path, args.detect_method)
    elif args.mode == 'add_identity':
        add_identity(args.dataset_path, args.name, svm_path_file=args.svm_model, detect_method=args.detect_method)
    elif args.mode == 'create_database':
        if args.classify_method == 'svm':
            create_database_SVM(args.dataset_path, facenet, args.svm_model, args.detect_method)
        else:
            create_database_faiss(args.dataset_path, facenet, args.faiss_path, args.names_path, args.detect_method)
        
