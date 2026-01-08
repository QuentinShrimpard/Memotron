"""
Script pour créer un fichier CSV à partir des landmarks de main et de pose.
Utilise les modèles MediaPipe hand_landmarker.task et pose_landmarker_full.task.
Les points sont normalisés par rapport à la position du nez (point 0 de pose_landmarker).
"""

import os
import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np

import utilities

# Chemins des modèles
HAND_MODEL_PATH = "models/hand_landmarker2.task"
POSE_MODEL_PATH = "models/pose_landmarker_full.task"
PHOTOS_ROOT = "LesPhotos"
OUTPUT_CSV = "dataset_landmarks.csv"

# Nombre de landmarks
NUM_HAND_LANDMARKS = 21  # Par main
NUM_POSE_LANDMARKS = 33
POSE_LANDMARKS_TO_USE = list(range(11, 25))  # Épaules (11,12) jusqu'aux hanches (23,24)

# Configuration du filtrage par nombre de mains détectées
# Format: {"label": nombre_de_mains_requis}
# Si None ou absent, aucun filtrage n'est appliqué
HAND_COUNT_FILTER = {
    # "Uwu": 2,      # Uwu nécessite 2 mains détectées
    "Pouce": 1,    # Pouce nécessite exactement 1 main détectée
    # "AbsoluteCinema":2,
    "Nerd": 1,
    "Silence":1,
}


def create_hand_landmarker():
    """Crée le détecteur de landmarks de main."""
    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2  # Détecter jusqu'à 2 mains
    )
    return vision.HandLandmarker.create_from_options(options)


def create_pose_landmarker():
    """Crée le détecteur de landmarks de pose."""
    base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=1
    )
    return vision.PoseLandmarker.create_from_options(options)


def add_noise(value, noise_level=0.02):
    """
    Ajoute du bruit gaussien à une valeur.
    
    Args:
        value: La valeur à bruiter
        noise_level: L'écart-type du bruit gaussien (par défaut 0.02)
    
    Returns:
        La valeur bruitée
    """
    noise = np.random.normal(0, noise_level)
    return value + noise


def augment_csv_data(csv_path, output_path=None, target_samples=None, noise_level=0.02):
    """
    Duplique et ajoute du bruit aux données du CSV pour augmenter le dataset.
    
    Args:
        csv_path: Chemin du fichier CSV original
        output_path: Chemin du fichier CSV de sortie (par défaut: csv_path)
        target_samples: Dict {label: nombre_cible} pour équilibrer les classes.
                       Si None, double simplement toutes les données.
                       Exemple: {"Pouce": 100, "Silence": 150}
        noise_level: Niveau de bruit à ajouter (écart-type, par défaut 0.02)
    
    Returns:
        Statistiques de l'augmentation
    """
    if output_path is None:
        output_path = csv_path
    
    print(f"\n{'='*60}")
    print("AUGMENTATION DES DONNÉES")
    print(f"{'='*60}")
    
    # Charger le CSV
    df = pd.read_csv(csv_path)
    print(f"Dataset original: {len(df)} échantillons")
    
    # Afficher la distribution actuelle
    print("\nDistribution actuelle:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} échantillons")
    
    # Préparer les nouvelles lignes
    augmented_rows = []
    stats = {}
    
    # Pour chaque label
    for label in df['label'].unique():
        label_data = df[df['label'] == label]
        current_count = len(label_data)
        
        # Déterminer combien d'échantillons augmentés créer
        if target_samples and label in target_samples:
            target = target_samples[label]
            num_to_generate = max(0, target - current_count)
        else:
            # Par défaut, doubler les données
            num_to_generate = current_count
        
        stats[label] = {
            'original': current_count,
            'augmented': num_to_generate,
            'total': current_count + num_to_generate
        }
        
        print(f"\n{label}:")
        print(f"  Échantillons originaux: {current_count}")
        print(f"  Échantillons à générer: {num_to_generate}")
        
        # Générer les échantillons augmentés
        for i in range(num_to_generate):
            # Choisir un échantillon aléatoire de cette classe
            random_sample = label_data.sample(n=1).iloc[0].copy()
            
            # Ajouter du bruit à tous les features (sauf le label)
            for col in df.columns:
                if col != 'label':
                    random_sample[col] = add_noise(random_sample[col], noise_level)
            
            augmented_rows.append(random_sample)
    
    # Créer le DataFrame augmenté
    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        # Combiner avec les données originales
        final_df = pd.concat([df, augmented_df], ignore_index=True)
    else:
        final_df = df
    
    # Mélanger les données
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Sauvegarder
    final_df.to_csv(output_path, index=False)
    
    # Afficher le résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ DE L'AUGMENTATION")
    print(f"{'='*60}")
    print(f"Dataset final: {len(final_df)} échantillons")
    print(f"Augmentation: +{len(final_df) - len(df)} échantillons")
    print(f"Fichier sauvegardé: {output_path}")
    
    print("\nDistribution finale:")
    for label, stat in stats.items():
        print(f"  {label}: {stat['total']} échantillons ")
        print(f"    (original: {stat['original']}, augmenté: {stat['augmented']})")
    
    return stats




def extract_landmarks(image_path, hand_landmarker, pose_landmarker):
    """
    Extrait les landmarks de main et de pose d'une image.
    Retourne une liste de valeurs normalisées par rapport au nez.
    Retourne également le nombre de mains détectées.
    """
    # Charger l'image
    image = mp.Image.create_from_file(image_path)
    
    # Détecter les landmarks de pose
    pose_result = pose_landmarker.detect(image)
    
    # Récupérer la position du nez pour la normalisation
    # en fait non HEHEHE
    nose_x, nose_y, nose_z = utilities.get_nose_position(pose_result.pose_landmarks)
    
    # Détecter les landmarks de main
    hand_result = hand_landmarker.detect(image)
    
    # Compter le nombre de mains détectées
    num_hands_detected = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
    
    # Extraire les landmarks de main (2 mains * 21 points * 3 coords = 126 valeurs)
    hand_values = []
    for hand_idx in range(2):  # Pour 2 mains
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
    
    return hand_values + pose_values, num_hands_detected


def should_keep_image(label, num_hands_detected):
    """
    Détermine si une image doit être conservée en fonction du nombre de mains détectées.
    
    Args:
        label: Le label de l'image (catégorie)
        num_hands_detected: Le nombre de mains détectées dans l'image
    
    Returns:
        True si l'image doit être conservée, False sinon
    """
    # Si aucun filtre n'est défini pour ce label, conserver l'image
    if label not in HAND_COUNT_FILTER:
        return True
    
    # Vérifier si le nombre de mains correspond au filtre
    required_hands = HAND_COUNT_FILTER[label]
    return num_hands_detected == required_hands


def generate_header():
    """Génère l'en-tête du CSV."""
    header = ["label"]
    
    # Colonnes pour les landmarks de main (2 mains)
    for hand_idx in range(2):
        for i in range(NUM_HAND_LANDMARKS):
            header.extend([
                f"hand{hand_idx}_point{i}_x",
                f"hand{hand_idx}_point{i}_y",
                f"hand{hand_idx}_point{i}_z"
            ])
    
    # Colonnes pour les landmarks de pose (uniquement épaules à hanches)
    for i in POSE_LANDMARKS_TO_USE:
        header.extend([
            f"pose_point{i}_x",
            f"pose_point{i}_y",
            f"pose_point{i}_z"
        ])
    
    return header


def process_photos(photos_root, output_csv):
    """Traite toutes les photos et génère le fichier CSV."""
    # Créer les détecteurs
    hand_landmarker = create_hand_landmarker()
    pose_landmarker = create_pose_landmarker()
    
    # Préparer les données
    rows = []
    
    # Statistiques de filtrage
    stats = {}
    
    # Parcourir les dossiers (chaque dossier = un label)
    for label in os.listdir(photos_root):
        label_path = os.path.join(photos_root, label)
        
        if not os.path.isdir(label_path):
            continue
        
        print(f"Traitement du label: {label}")
        
        # Initialiser les statistiques pour ce label
        stats[label] = {"total": 0, "kept": 0, "filtered": 0}
        
        # Afficher le filtre s'il existe
        if label in HAND_COUNT_FILTER:
            print(f"  Filtre actif: conserver uniquement les images avec {HAND_COUNT_FILTER[label]} main(s)")
        
        # Parcourir les images du dossier
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            
            # Vérifier que c'est une image
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                continue
            
            try:
                stats[label]["total"] += 1
                
                # Extraire les landmarks et le nombre de mains
                landmarks, num_hands_detected = extract_landmarks(image_path, hand_landmarker, pose_landmarker)
                
                # Vérifier si l'image doit être conservée
                if should_keep_image(label, num_hands_detected):
                    # Ajouter la ligne avec le label
                    row = [label] + landmarks
                    rows.append(row)
                    stats[label]["kept"] += 1
                    print(f"  ✓ Traité: {image_file} ({num_hands_detected} main(s))")
                else:
                    stats[label]["filtered"] += 1
                    print(f"  ✗ Ignoré: {image_file} ({num_hands_detected} main(s) - requis: {HAND_COUNT_FILTER[label]})")
                
            except Exception as e:
                print(f"  - Erreur pour {image_file}: {e}")
    
    # Fermer les détecteurs
    hand_landmarker.close()
    pose_landmarker.close()
    
    # Écrire le fichier CSV
    header = generate_header()
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"\n{'='*60}")
    print(f"Fichier CSV créé: {output_csv}")
    print(f"Nombre total de lignes: {len(rows)}")
    print(f"Nombre de colonnes: {len(header)}")
    print(f"\n{'='*60}")
    print("Statistiques de filtrage par catégorie:")
    print(f"{'='*60}")
    for label, stat in stats.items():
        print(f"{label}:")
        print(f"  - Total d'images: {stat['total']}")
        print(f"  - Images conservées: {stat['kept']}")
        print(f"  - Images filtrées: {stat['filtered']}")
        if stat['total'] > 0:
            percentage = (stat['kept'] / stat['total']) * 100
            print(f"  - Taux de conservation: {percentage:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Créer un CSV de landmarks à partir de photos.")
    parser.add_argument("--photos-root", type=str, default=PHOTOS_ROOT,
                        help="Dossier racine contenant les sous-dossiers de labels")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV,
                        help="Chemin du fichier CSV de sortie")
    parser.add_argument("--augment", action="store_true",
                        help="Activer l'augmentation de données après la création du CSV")
    parser.add_argument("--noise-level", type=float, default=0.02,
                        help="Niveau de bruit pour l'augmentation (défaut: 0.02)")
    
    args = parser.parse_args()
    
    process_photos(args.photos_root, args.output)
    
    # Augmentation optionnelle
    if args.augment:
        # Exemple: équilibrer toutes les classes à 100 échantillons
        # Vous pouvez modifier ce dictionnaire selon vos besoins
        target_samples = {
            # "Pouce": 100,
            # "Silence": 100,
            # "Nerd": 100,
            # etc...
        }
        augment_csv_data(
            csv_path=args.output,
            target_samples=target_samples if target_samples else None,
            noise_level=args.noise_level
        )
