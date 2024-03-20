import os
import pickle
from PIL import Image
import numpy as np
import torch
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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


def train_and_save(dataset_dir="dataset", save_path='classifier.pkl', test_size=0.2, random_state=42):
    # Load dataset
    face_images, labels = load_dataset(dataset_dir)
    print("Dataset loaded successfully.")

    # Extract facial embeddings
    face_embeddings = extract_embeddings(face_images)
    print("Facial embeddings extracted successfully.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(face_embeddings, labels, test_size=test_size, random_state=random_state)

    # Train the classifier
    classifier = train_classifier(X_train, y_train)
    print("Classifier trained successfully.")

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

    # Save the classifier
    with open(save_path, 'wb') as f:
        pickle.dump(classifier, f)
    print("Classifier saved successfully.")

# Example usage
train_and_save(dataset_dir="dataset", save_path='your_classifier.pkl')


