import os
import face_recognition


def load_dataset_names():
    dataset_path = "dataset"
    os.makedirs(dataset_path, exist_ok=True)
    return [entry.name for entry in os.scandir(dataset_path) if entry.is_dir()]


def encode_dataset(dataset_names):
    dataset_encodings = []

    for name in dataset_names:
        directory = os.path.join("dataset", name)
        for filename in os.listdir(directory):
            image_path = os.path.join(directory, filename)
            face_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(face_image)
            if len(face_encoding) > 0:
                dataset_encodings.append(face_encoding[0])

    return dataset_encodings
