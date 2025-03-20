import cv2
import dlib
import numpy as np
import torch
import os
from facenet_pytorch import InceptionResnetV1
from numpy.linalg import norm

# Load FaceNet Model for face embeddings
facenet = InceptionResnetV1(pretrained="vggface2").eval()

# Load Face Detector (HOG or CNN)
use_cnn = False  # Set True for CNN-based detection

detector = dlib.cnn_face_detection_model_v1(
    "./face_models/mmod_human_face_detector.dat") if use_cnn else dlib.get_frontal_face_detector()

# Load the Dlib landmark predictor
predictor = dlib.shape_predictor("./face_models/shape_predictor_68_face_landmarks.dat")


def get_embedding_from_face(face_image):
    """Extracts facial embeddings using FaceNet model."""
    img = cv2.resize(face_image, (160, 160)).astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    return facenet(img).detach().numpy()


def extract_faces(image_path):
    """Detects faces in an image and extracts their embeddings."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Error: Image not found - {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if use_cnn:
        faces = [face.rect for face in faces]

    face_embeddings = []
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_crop = img[y:y + h, x:x + w]
        if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
            embedding = get_embedding_from_face(face_crop)
            face_embeddings.append((embedding, (x, y, w, h)))

    return face_embeddings, img


def load_id_bank(id_bank_folder):
    """Loads and extracts embeddings for known identities."""
    id_embeddings = {}
    for filename in os.listdir(id_bank_folder):
        if filename.endswith((".jpg", ".png")):
            img = cv2.imread(os.path.join(id_bank_folder, filename))
            if img is not None:
                id_embeddings[filename] = get_embedding_from_face(img)
    return id_embeddings


def cosine_similarity(emb1, emb2):
    """Computes cosine similarity between two embeddings."""
    return np.dot(emb1, emb2.T) / (norm(emb1) * norm(emb2))


def match_faces(detected_faces, id_embeddings, threshold=0.5, metric="cosine"):
    """Matches detected faces with known identities using similarity metrics."""
    assert metric in ["cosine", "euclidean"], "Invalid metric. Choose 'cosine' or 'euclidean'."
    matched_faces = []

    for emb1, bbox in detected_faces:
        best_match, best_similarity = None, -1
        for id_name, emb2 in id_embeddings.items():
            similarity = cosine_similarity(emb1, emb2)[0][0] if metric == "cosine" else 1 - norm(emb1 - emb2)
            if similarity > best_similarity:
                best_similarity, best_match = similarity, id_name

        if best_similarity > threshold:
            matched_faces.append((bbox, best_match, best_similarity))

    return matched_faces


# Main Code
id_bank_folder = "./ID_bank/"
detected_faces, img = extract_faces("./sample/test.jpg")
id_embeddings = load_id_bank(id_bank_folder)
matched_faces = match_faces(detected_faces, id_embeddings, threshold=0.5, metric="cosine")

# Draw results
for (x, y, w, h), id_name, similarity in matched_faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, f"{id_name} ({similarity:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Face Recognition", img)
cv2.imwrite("output.jpg", img)

# Wait for key press or window close
while True:
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
