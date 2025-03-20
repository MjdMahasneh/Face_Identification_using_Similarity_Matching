from deepface import DeepFace
import cv2
import os


def recognize_faces(image_path, id_bank_folder, model_name="VGG-Face", distance_metric="cosine", threshold=0.5):
    """Extracts faces from an image and recognizes them using DeepFace."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Error: Image not found - {image_path}")

    # Detect faces using MTCNN backend
    faces = DeepFace.extract_faces(img_path=image_path, detector_backend="mtcnn")

    # Load ID bank images and their embeddings
    id_embeddings = {}
    for filename in os.listdir(id_bank_folder):
        if filename.endswith((".jpg", ".png")):
            id_path = os.path.join(id_bank_folder, filename)
            id_embeddings[filename] = DeepFace.represent(img_path=id_path, model_name=model_name)[0]["embedding"]

    matched_faces = []

    # Compare each detected face with the ID bank
    for face in faces:
        x, y, w, h = int(face["facial_area"]["x"]), int(face["facial_area"]["y"]), int(face["facial_area"]["w"]), int(
            face["facial_area"]["h"])
        face_img = img[y:y + h, x:x + w]  # Crop face

        if face_img.shape[0] > 0 and face_img.shape[1] > 0:
            face_embedding = DeepFace.represent(img_path=face_img, model_name=model_name)[0]["embedding"]

            best_match, best_similarity = None, float("inf")
            for id_name, id_embedding in id_embeddings.items():
                similarity = DeepFace.find(img_path=face_img, db_path=id_bank_folder, model_name=model_name,
                                           distance_metric=distance_metric)
                if similarity and similarity[0]["VGG-Face_cosine"].min() < threshold:
                    best_match = id_name
                    best_similarity = similarity[0]["VGG-Face_cosine"].min()

            if best_match:
                matched_faces.append((x, y, w, h, best_match, best_similarity))

    # Draw results
    for (x, y, w, h, id_name, similarity) in matched_faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"{id_name} ({similarity:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", img)

    # Wait for key press or window close
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


# Run the face recognition
id_bank_folder = "./ID_bank/"
recognize_faces("./sample/test.jpg", id_bank_folder)
