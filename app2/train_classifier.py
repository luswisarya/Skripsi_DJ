import os
import pickle
from PIL import Image
import numpy as np
import torch
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
from sklearn.svm import SVC


def load_dataset(dataset_dir):
    """
    Load images from the dataset directory and extract faces using MTCNN.
    Returns a list of face images and corresponding labels.
    """
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    elif not os.listdir(dataset_dir):
        raise ValueError("Dataset directory is empty.")

    face_images = []
    labels = []

    detector = MTCNN()

    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                img_path = os.path.join(label_dir, filename)
                img = Image.open(img_path).convert('RGB')

                # Detect faces and extract bounding boxes
                result = detector.detect_faces(np.array(img))
                if result:
                    box = result[0]['box']
                    # Crop the face using the bounding box
                    face = img.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
                    face_images.append(face)
                    labels.append(label)
                else:
                    raise Exception("No face detected in {img_path}")

    if not face_images:
        raise ValueError("No images found in the dataset directory.")

    return face_images, labels


def extract_embeddings(face_images):
    """
    Extract facial embeddings using FaceNet (InceptionResnetV1).
    Returns a numpy array of facial embeddings.
    """
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    embeddings = []
    for face_image in face_images:
        # Convert PIL Image to PyTorch tensor and normalize
        face_tensor = torch.tensor(np.array(face_image)).float() / 255.0
        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)

        # Extract facial embeddings
        with torch.no_grad():
            face_embedding = resnet(face_tensor).numpy()[0]

        embeddings.append(face_embedding)

    return np.array(embeddings)


def train_classifier(face_embeddings, labels):
    """
    Train a classifier using the extracted facial embeddings and labels.
    Returns the trained classifier.
    """
    # Ensure there are at least two unique classes for training
    # TODO: Move error check when clicking "Train" button // count folder in dataset & check if empty inside it
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Insufficient number of classes for training. At least two classes are required.")

    # Use SVC classifier
    clf = SVC(kernel='linear', probability=True)
    clf.fit(face_embeddings, labels)
    return clf


def train_and_save(dataset_dir="dataset", save_path='classifier.pkl'):

    face_images, labels = load_dataset(dataset_dir)
    print("Dataset loaded successfully.")

    face_embeddings = extract_embeddings(face_images)
    print("Facial embeddings extracted successfully.")

    # Print unique labels and their counts
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print("Unique Labels:", unique_labels)
    print("Label Counts:", label_counts)

    classifier = train_classifier(face_embeddings, labels)

    # Save the classifier directly within the train_and_save function
    with open(save_path, 'wb') as f:
        pickle.dump(classifier, f)
    print("Classifier saved successfully.")


class FaceRecognitionModel:
    def __init__(self):
        pass


if __name__ == "__main__":
    model = FaceRecognitionModel()
    train_and_save()
