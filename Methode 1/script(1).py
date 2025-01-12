
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Fonction pour recadrer une image au centre
def crop_center(image, crop_size):
    h, w = image.shape[:2]
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    return image[start_y:start_y + crop_size, start_x:start_x + crop_size]

# Fonction pour extraire les indices U, M, L et leurs résidus
def extract_residues(image):
    # Normalisation des canaux RGB
    image = image.astype(np.float32) / 255.0
    r, g, b = cv2.split(image)

    # Calcul des indices U (max), M (médian), L (min)
    max_channel = np.maximum(np.maximum(r, g), b)
    min_channel = np.minimum(np.minimum(r, g), b)
    median_channel = r + g + b - max_channel - min_channel

    # Application d'un filtre passe-haut (Laplacien diagonal)
    kernel = np.array([[1, 0, 1], [0, -4, 0], [1, 0, 1]])
    u_res = cv2.filter2D(np.log1p(max_channel), -1, kernel)
    m_res = cv2.filter2D(np.log1p(median_channel), -1, kernel)
    l_res = cv2.filter2D(np.log1p(min_channel), -1, kernel)

    # Création des vecteurs de caractéristiques
    features = np.hstack([u_res.flatten(), m_res.flatten(), l_res.flatten()])
    return features

# Fonction pour extraire un maximum de 5 I-frames d'une vidéo avec des métadonnées

def extract_iframes(video_path, max_iframes=5):
    cap = cv2.VideoCapture(video_path)
    iframes = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Identifier les I-frames à l'aide des métadonnées
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == current_frame + 1:
            iframes.append(crop_center(frame, 512))

        # Limiter le nombre d'I-frames
        if len(iframes) >= max_iframes:
            break

        current_frame += 1

    cap.release()
    return iframes

# Fonction pour entraîner le modèle
def train_model(real_videos, synthetic_videos):
    X = []
    y = []

    # Extraire les caractéristiques des vidéos réelles
    for video_path in real_videos:
        iframes = extract_iframes(video_path)
        for frame in iframes:
            features = extract_residues(frame)
            X.append(features)
            y.append(0)  # Label pour les vidéos réelles

    # Extraire les caractéristiques des vidéos synthétiques
    for video_path in synthetic_videos:
        iframes = extract_iframes(video_path)
        for frame in iframes:
            features = extract_residues(frame)
            X.append(features)
            y.append(1)  # Label pour les vidéos synthétiques

    # Diviser les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle Random Forest
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Évaluer le modèle
    y_pred = model.predict(X_test)
    print(f"Precision du modèle : {accuracy_score(y_test, y_pred)}")

    # Sauvegarder le modèle
    joblib.dump(model, 'synthetic_image_detector.pkl')
    print("Modèle sauvegardé sous 'synthetic_image_detector.pkl'.")

# Traitement des vidéos
def process_video(video_path):
    iframes = extract_iframes(video_path)
    results = []

    for frame in iframes:
        features = extract_residues(frame)
        model = joblib.load('synthetic_image_detector.pkl')
        prediction = model.predict([features])[0]
        results.append(prediction)

    return results

# Parcourir un dossier pour traiter toutes les vidéos
def process_videos_in_folder(folder_path):
    video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mkv'))]
    for video in video_files:
        results = process_video(video)
        print(f"Resultats pour {video} : {results}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Étape 1 : Entraînement du modèle
    real_videos_folder = "videos_ressources/Originaux"
    synthetic_videos_folder = "videos_ressources/Fakes"

    real_videos = [os.path.join(real_videos_folder, f) for f in os.listdir(real_videos_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]
    synthetic_videos = [os.path.join(synthetic_videos_folder, f) for f in os.listdir(synthetic_videos_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]

    train_model(real_videos, synthetic_videos)

    # Étape 2 : Traitement des vidéos
    process_videos_in_folder("videos_tests")