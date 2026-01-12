import cv2
import mediapipe as mp
import os

def load_and_resize(path, size):
    img = cv2.imread(path)
    if img is None:
        print(f"ça marche pas bozo {path}")
        # On retourne une image noire carrée en cas d'erreur pour ne pas crash
        return cv2.resize(cv2.imread('default_error.jpg'), size) if os.path.exists('default_error.jpg') else None
    return cv2.resize(img, size)

HAND_CONNEXIONS = [(0,1), (1,2), (2,3), (3,4),             # Pouce
                   (0,5), (5,6), (6,7), (7,8),             # Index
                   (0,9), (9,10), (10,11), (11,12),       # Majeur
                   (0,13), (13,14), (14,15), (15,16),     # Annulaire
                   (0,17), (17,18), (18,19), (19,20),     # Auriculaire
                   (5,9), (9,13), (13,17)                 # Connexions entre les bases des doigts
                   ]

POSE_CONNEXIONS = [(11,12), (11,13), (13,15), (12,14),
                    (14,16), (16, 18), (15,17), (18,20), (17,19), (20,16), (19,15), (16,22), (15,21),
                    (11,23), (12,24), (23,24),
                    (23,25), (24,26), (25,27), (26,28),
                    (27,31), (28,32), 
                    (0,1), (1,2), (2,3), (3,7),
                    (0,4), (4,5), (5,6), (6,8),
                    (9,10)]

FACE_CONNEXIONS = [(336,296), (296,334), (334,293), (293,300), # Haut sourcil gauche
                   (285,295), (295,282), (282,283), (283,276), # Bas sourcil gauche
                   (70,156), (156,46), (300,383), (383,276), # Pointe des sourcils
                   (107,66), (66,105), (105,63), (63,70), # Haut sourcil droit
                   (55,65), (65,52), (52,53), (53,46), # Bas sourcil droit
                   (78,191), (191,80), (80,81), (81,82), (82,13), (13,312), (312,311), (311,310), (310, 415), (415,308), # Sous lèvre supérieure
                   (78,95), (95,88), (88,178), (178,87), (87,14), (14, 317), (317,402), (402,318), (318,324), (324,308) # Sous lèvre inférieure
                   ]
# FACE_USED_LANDMARKS1 = [k[0] for k in FACE_CONNEXIONS]
# FACE_USED_LANDMARKS2 = [k[1] for k in FACE_CONNEXIONS]
# FACE_USED_LANDMARKS = [*set(FACE_USED_LANDMARKS1 + FACE_USED_LANDMARKS2)]
# FACE_USED_LANDMARKS.sort()
# print(FACE_USED_LANDMARKS)
# HEHEHE
FACE_USED_LANDMARKS = [13, 14, 46, 52, 53, 55, 63, 65, 66, 70, 78, 80, 81, 82, 
                       87, 88, 95, 105, 107, 156, 178, 191, 276, 282, 283, 285, 
                       293, 295, 296, 300, 308, 310, 311, 312, 317, 318, 324, 
                       334, 336, 383, 402, 415]

def get_nose_position(pose_landmarks):
    """Récupère la position du nez (point 0) pour la normalisation."""
    if pose_landmarks and len(pose_landmarks) > 0:
        nose = pose_landmarks[0][0]  # Premier pose, premier landmark (nez)
        return nose.x, nose.y, nose.z
    return 0.5, 0.5, 0.0  # Position par défaut si pas de pose détectée

def get_shoulders_distance(pose_landmarks):
    """Calcule la distance entre les épaules gauche et droite."""
    if pose_landmarks and len(pose_landmarks) > 0:
        left_shoulder = pose_landmarks[0][11]  # Épaule gauche
        right_shoulder = pose_landmarks[0][12]  # Épaule droite
        distance = ((left_shoulder.x - right_shoulder.x) ** 2 + 
                    (left_shoulder.y - right_shoulder.y) ** 2 + 
                    (left_shoulder.z - right_shoulder.z) ** 2) ** 0.5
        return distance
    return 1.0  # Valeur par défaut si pas de pose détectée


def normalize_landmark(landmark, nose_x, nose_y, nose_z):
    """Normalise un landmark par rapport à la position du nez."""
    return (
        landmark.x - nose_x,
        landmark.y - nose_y,
        landmark.z - nose_z
    )

def normalize_landmark2(landmark, distance):
    """Normalise un landmark par rapport à la distance entre les épaules."""
    if distance == 0:
        return landmark.x, landmark.y, landmark.z
    return (
        landmark.x / distance,
        landmark.y / distance,
        landmark.z / distance
    )

def normalize_landmark3(landmark, center_x, center_y, center_z, scale):
    """Normalise un landmark par rapport à un centre et une échelle."""
    if scale == 0:
        return landmark.x, landmark.y, landmark.z
    return (
        (landmark.x - center_x) / scale,
        (landmark.y - center_y) / scale,
        (landmark.z - center_z) / scale
    )
def get_eyes_distance(pose_landmarks):
    """Calcule la distance entre les yeux gauche et droit."""
    if pose_landmarks and len(pose_landmarks) > 0:
        left_eye = pose_landmarks[0][1]  # Oeil gauche
        right_eye = pose_landmarks[0][4]  # Oeil droit
        distance = ((left_eye.x - right_eye.x) ** 2 + 
                    (left_eye.y - right_eye.y) ** 2 + 
                    (left_eye.z - right_eye.z) ** 2) ** 0.5
        return distance
    return 1.0  # Valeur par défaut si pas de pose détectée

def normalize_face_landmarks(face_landmarks, nose_x, nose_y, nose_z, eyes_distance):
    """Normalise les landmarks du visage.
       Besoin d'une normalisation spécifique car les distances sont très petites.
    """
    normalized = []
    for landmark in face_landmarks:
        norm_landmark = normalize_landmark(landmark, nose_x, nose_y, nose_z)
        norm_landmark = (
            norm_landmark[0] / eyes_distance,
            norm_landmark[1] / eyes_distance,
            norm_landmark[2] / eyes_distance
        )
        normalized.append((norm_landmark[0]*2, norm_landmark[1]*2, norm_landmark[2]*2))
    return normalized
