"""
Le Memotron - Système de reconnaissance de gestes en temps réel avec réseau de neurones
Utilise les modèles hand_landmarker et pose_landmarker pour extraire les features,
puis le modèle memotron_model.pth pour prédire le meme correspondant.
"""

import mediapipe as mp
import cv2
import torch
import numpy as np
import utilities
from pyvidplayer2 import Video
import pygame
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Chemins des modèles
HAND_MODEL = "models/hand_landmarker2.task"
POSE_MODEL = "models/pose_landmarker_full.task"
FACE_MODEL = "models/face_landmarker.task"
NN_MODEL = "models/memotron_model.pth"

# Configuration de la popup
POPUP_NAME = "This you ?"
POPUP_OUVERTE = False
POPUP_SIZE = (800, 600)
POPUP_POS = (500, 100)

# Configuration audio
AUDIO_DIR = "audios"

# Configuration
NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = 33
POSE_LANDMARKS_TO_USE = list(range(11, 25))  # Épaules (11,12) jusqu'aux hanches (23,24)
show_webcam = True


def find_audio_file(meme_name):
    """Cherche un fichier audio correspondant au meme."""
    # Extensions audio supportées
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac']
    
    for ext in audio_extensions:
        audio_path = os.path.join(AUDIO_DIR, f"{meme_name}{ext}")
        if os.path.exists(audio_path):
            return audio_path
    return None


def load_meme_images():
    """Charge les images de mèmes et référence les vidéos."""
    meme_map = {
        "AbsoluteCinema": {"type": "image", "content": utilities.load_and_resize("memes/absolute_cinema.jpg", POPUP_SIZE)},
        "Josh":           {"type": "video", "path": "memes/Josh.mp4"},
        "Nerd":           {"type": "image", "content": utilities.load_and_resize("memes/nerd_emoji.jpg", POPUP_SIZE)},
        "Pouce":          {"type": "image", "content": utilities.load_and_resize("memes/ryan_thumb.jpg", POPUP_SIZE)},
        "rien":           None,  # YA RIEN
        "Silence":        {"type": "image", "content": utilities.load_and_resize("memes/SILENCE.jpg", POPUP_SIZE)},
        "Uwu":            {"type": "image", "content": utilities.load_and_resize("memes/uwu.jpg", POPUP_SIZE)},
        "HellYeah":            {"type": "image", "content": utilities.load_and_resize("memes/HellYeah.jpg", POPUP_SIZE)},

    }
    
    # Ajouter les références audio pour chaque meme
    for key, value in meme_map.items():
        if value is not None:
            audio_path = find_audio_file(key)
            value["audio"] = audio_path
    
    return {k: v for k, v in meme_map.items() if v is not None}


def create_video_player(video_path):
    """Crée un lecteur vidéo pyvidplayer2."""
    try:
        vid = Video(video_path)
        vid.resize(POPUP_SIZE)
        return vid
    except Exception as e:
        print(f"Erreur lors du chargement de la vidéo: {e}")
        return None



def extract_features(hand_result, pose_result):
    """
    Extrait les features normalisées à partir des résultats de détection.
    Retourne un vecteur numpy de la même forme que celui utilisé pour l'entraînement.
    """
    # Récupérer la position du nez pour normalisation
    nose_x, nose_y, nose_z = utilities.get_nose_position(pose_result.pose_landmarks)
    
    # Extraire les landmarks de main (2 mains * 21 points * 3 coords = 126 valeurs)
    hand_values = []
    for hand_idx in range(2):
        if hand_idx < len(hand_result.hand_landmarks):
            for landmark in hand_result.hand_landmarks[hand_idx]:
                # x, y, z = utilities.normalize_landmark(landmark, nose_x, nose_y, nose_z)
                x, y, z = utilities.normalize_landmark2(landmark, utilities.get_shoulders_distance(pose_result.pose_landmarks))
                hand_values.extend([x, y, z])
        else:
            # Remplir avec des zéros si main non détectée
            hand_values.extend([0.0] * (NUM_HAND_LANDMARKS * 3))
    
    # Extraire les landmarks de pose (uniquement épaules à hanches: 14 points * 3 coords = 42 valeurs)
    pose_values = []
    if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
        pose_landmarks = pose_result.pose_landmarks[0]
        for idx in POSE_LANDMARKS_TO_USE:
            if idx < len(pose_landmarks):
                landmark = pose_landmarks[idx]
                # x, y, z = utilities.normalize_landmark(landmark, nose_x, nose_y, nose_z)
                x, y, z = utilities.normalize_landmark2(landmark, utilities.get_shoulders_distance(pose_result.pose_landmarks))
                pose_values.extend([x, y, z])
            else:
                pose_values.extend([0.0, 0.0, 0.0])
    else:
        # Remplir avec des zéros si pose non détectée
        pose_values.extend([0.0] * (len(POSE_LANDMARKS_TO_USE) * 3))
    
    # Concatener et convertir en numpy array
    features = np.array(hand_values + pose_values, dtype=np.float32)
    return features


def load_neural_network(model_path):
    """Charge le modèle de réseau de neurones."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Récupérer les métadonnées
    label_encoder = checkpoint['label_encoder']
    num_features = checkpoint['num_features']
    num_classes = checkpoint['num_classes']
    
    # Recréer le modèle
    from memotronTrainer import GestureClassifier
    model = GestureClassifier(num_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Modèle chargé: {model_path}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Précision de validation: {checkpoint['val_acc']:.2f}%")
    
    return model, label_encoder


def predict_gesture(model, label_encoder, features):
    """Prédit le geste à partir des features."""
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
        confidence_value = confidence.item()
        
        return predicted_label, confidence_value


def draw_landmarks(frame, hand_result, pose_result):
    """Dessine les landmarks sur la frame."""
    h, w, _ = frame.shape
    
    # Dessiner les landmarks de main
    if hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:
            points = []
            for lm in hand_landmarks:
                points.append((int(lm.x * w), int(lm.y * h)))
            
            # Dessiner les connexions
            for connection in utilities.HAND_CONNEXIONS:
                p1 = points[connection[0]]
                p2 = points[connection[1]]
                cv2.line(frame, p1, p2, (0, 255, 0), 2)  # Vert
            
            # Dessiner les points
            for pt in points:
                cv2.circle(frame, pt, 5, (0, 0, 255), cv2.FILLED)  # Rouge
    
    # Dessiner les landmarks de pose
    if pose_result.pose_landmarks:
        for pose_landmarks in pose_result.pose_landmarks:
            points = []
            for lm in pose_landmarks:
                points.append((int(lm.x * w), int(lm.y * h)))
            
            # Dessiner les connexions
            for connection in utilities.POSE_CONNEXIONS:
                p1 = points[connection[0]]
                p2 = points[connection[1]]
                cv2.line(frame, p1, p2, (255, 0, 0), 2)  # Bleu
            
            # Dessiner les points
            for pt in points:
                cv2.circle(frame, pt, 3, (255, 255, 0), cv2.FILLED)  # Cyan


def play_audio(audio_path):
    """Joue un fichier audio."""
    if audio_path and os.path.exists(audio_path):
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Erreur lors de la lecture de l'audio: {e}")


def stop_audio():
    """Arrête la lecture audio."""
    pygame.mixer.music.stop()


def main():
    """Fonction principale."""
    print("="*60)
    print("DÉMARRAGE DU MEMOTRON")
    print("="*60)
    
    # Initialiser pygame mixer pour l'audio
    pygame.mixer.init()
    
    # Charger le modèle de réseau de neurones
    print("\nChargement du modèle de réseau de neurones...")
    model, label_encoder = load_neural_network(NN_MODEL)
    
    # Charger les images de mèmes
    print("\nChargement des images et vidéos de mèmes...")
    meme_map = load_meme_images()
    
    # Créer les détecteurs MediaPipe
    print("\nInitialisation des détecteurs MediaPipe...")
    
    # Hand landmarker
    hand_base_options = python.BaseOptions(model_asset_path=HAND_MODEL)
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    
    # Pose landmarker
    pose_base_options = python.BaseOptions(model_asset_path=POSE_MODEL)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
    
    print("\n✓ Memotron prêt!")
    print("Appuyez sur 'q' pour quitter\n")
    
    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)
    previous_meme = None
    POPUP_OUVERTE = False
    current_video_player = None
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        h, w, _ = frame.shape
        
        # Convertir pour MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Détecter les landmarks
        hand_result = hand_detector.detect(mp_image)
        pose_result = pose_detector.detect(mp_image)
        
        # Extraire les features
        features = extract_features(hand_result, pose_result)
        
        # Prédire le geste
        predicted_label, confidence = predict_gesture(model, label_encoder, features)
        
        # Afficher la prédiction sur la frame
        text = f"{predicted_label} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Dessiner les landmarks
        draw_landmarks(frame, hand_result, pose_result)
        
        # Gérer l'affichage du meme si la confiance est suffisante
        if confidence > 0.5 and predicted_label != "rien":
            if predicted_label in meme_map and previous_meme != predicted_label:
                meme_data = meme_map[predicted_label]
                
                # Jouer l'audio si disponible
                if meme_data.get("audio"):
                    play_audio(meme_data["audio"])
                
                if meme_data["type"] == "video":
                    # Créer un nouveau lecteur vidéo
                    if current_video_player is not None:
                        current_video_player.close()
                    current_video_player = create_video_player(meme_data["path"])
                    previous_meme = predicted_label
                    
                elif meme_data["type"] == "image":
                    # Fermer la vidéo si elle est en cours
                    if current_video_player is not None:
                        current_video_player.close()
                        current_video_player = None
                    # Afficher l'image dans une fenêtre
                    cv2.imshow(POPUP_NAME, meme_data["content"])
                    cv2.moveWindow(POPUP_NAME, POPUP_POS[0], POPUP_POS[1])
                    POPUP_OUVERTE = True
                    previous_meme = predicted_label
        else:
            # Fermer la popup image si confiance faible ou "rien"
            if POPUP_OUVERTE:
                cv2.destroyWindow(POPUP_NAME)
                POPUP_OUVERTE = False
                previous_meme = None
            # Fermer la vidéo si elle est en cours
            if current_video_player is not None:
                current_video_player.close()
                current_video_player = None
            # Arrêter l'audio si un meme n'est plus détecté
            stop_audio()
        
        # Afficher la frame vidéo si elle existe
        if current_video_player is not None:
            if current_video_player.active:
                try:
                    # Mettre à jour le lecteur vidéo
                    current_video_player.update()
                    # Obtenir la frame actuelle et la convertir en numpy array pour OpenCV
                    pygame_surface = current_video_player.frame_surf
                    if pygame_surface is not None:
                        # Convertir la surface pygame en array numpy BGR pour OpenCV
                        frame_array = pygame.surfarray.array3d(pygame_surface)
                        frame_bgr = cv2.cvtColor(frame_array.swapaxes(0, 1), cv2.COLOR_RGB2BGR)
                        cv2.imshow(POPUP_NAME, frame_bgr)
                        cv2.moveWindow(POPUP_NAME, POPUP_POS[0], POPUP_POS[1])
                except Exception as e:
                    print(f"Erreur lors de l'affichage de la vidéo: {e}")
            else:
                # La vidéo est terminée, fermer la fenêtre et le lecteur
                cv2.destroyWindow(POPUP_NAME)
                current_video_player.close()
                current_video_player = None
                previous_meme = None
        
        # Afficher la webcam
        if show_webcam:
            cv2.imshow("Memotron - Webcam", frame)
        
        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Nettoyage
    cap.release()
    if current_video_player is not None:
        current_video_player.close()
    stop_audio()
    pygame.mixer.quit()
    hand_detector.close()
    pose_detector.close()
    cv2.destroyAllWindows()
    
    print("\nMemotron arrêté.")


if __name__ == "__main__":
    main()
