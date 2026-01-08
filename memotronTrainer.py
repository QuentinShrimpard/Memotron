"""
Script d'entraînement d'un réseau de neurones pour la classification de gestes.
Utilise le dataset_landmarks.csv généré par createCSV.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


# Configuration
CSV_PATH = "dataset_landmarks.csv"
MODEL_PATH = "models/memotron_model.pth"
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.2  # 20% pour la validation


class GestureDataset(Dataset):
    """Dataset personnalisé pour les landmarks de gestes."""
    
    def __init__(self, csv_path):
        """
        Args:
            csv_path: Chemin vers le fichier CSV contenant les landmarks
        """
        # Charger le CSV
        self.data = pd.read_csv(csv_path)
        
        # Séparer les labels et les features
        self.labels = self.data['label'].values
        self.features = self.data.drop('label', axis=1).values.astype(np.float32)
        
        # Encoder les labels en entiers
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        print(f"Dataset chargé: {len(self.data)} échantillons")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"Nombre de features: {self.features.shape[1]}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        return features, label
    
    def get_num_classes(self):
        return len(self.label_encoder.classes_)
    
    def get_num_features(self):
        return self.features.shape[1]


class GestureClassifier(nn.Module):
    """Réseau de neurones pour la classification de gestes."""
    
    def __init__(self, input_size, num_classes):
        super(GestureClassifier, self).__init__()
        
        # Architecture du réseau
        self.network = nn.Sequential(
            # Couche 1
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Couche 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Couche 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Couche de sortie
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entraîne le modèle pour une epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Évalue le modèle sur l'ensemble de validation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Statistiques
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc


def compute_class_accuracy(model, val_loader, device, label_encoder):
    """Calcule l'accuracy par classe sur l'ensemble de validation."""
    model.eval()
    
    # Dictionnaires pour stocker les stats par classe
    class_correct = {}
    class_total = {}
    
    # Initialiser les compteurs pour chaque classe
    for class_name in label_encoder.classes_:
        class_correct[class_name] = 0
        class_total[class_name] = 0
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            
            # Compter les prédictions correctes par classe
            for i in range(labels.size(0)):
                label_idx = labels[i].item()
                class_name = label_encoder.classes_[label_idx]
                class_total[class_name] += 1
                if predicted[i] == labels[i]:
                    class_correct[class_name] += 1
    
    # Calculer l'accuracy par classe
    class_accuracies = {}
    for class_name in label_encoder.classes_:
        if class_total[class_name] > 0:
            acc = 100 * class_correct[class_name] / class_total[class_name]
            class_accuracies[class_name] = {
                'accuracy': acc,
                'correct': class_correct[class_name],
                'total': class_total[class_name]
            }
        else:
            class_accuracies[class_name] = {
                'accuracy': 0.0,
                'correct': 0,
                'total': 0
            }
    
    return class_accuracies


def train_model():
    """Fonction principale d'entraînement."""
    
    # Vérifier la disponibilité du GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device: {device}")
    
    # Charger le dataset
    print("\nChargement du dataset...")
    full_dataset = GestureDataset(CSV_PATH)
    
    # Diviser en train et validation
    train_size = int((1 - VALIDATION_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"\nTaille du dataset d'entraînement: {train_size}")
    print(f"Taille du dataset de validation: {val_size}")
    
    # Créer les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Créer le modèle
    num_features = full_dataset.get_num_features()
    num_classes = full_dataset.get_num_classes()
    model = GestureClassifier(num_features, num_classes).to(device)
    
    print(f"\nModèle créé avec {num_features} features et {num_classes} classes")
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters())}")
    
    # Loss et optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Variable pour suivre le meilleur modèle
    best_val_acc = 0.0
    
    # Créer le dossier models s'il n'existe pas
    os.makedirs("models", exist_ok=True)
    
    # Entraînement
    print("\n" + "="*60)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("="*60)
    
    for epoch in range(NUM_EPOCHS):
        # Entraînement
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Affichage
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%", end="")
        
        # Sauvegarder le modèle si c'est le meilleur
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'label_encoder': full_dataset.label_encoder,
                'num_features': num_features,
                'num_classes': num_classes
            }, MODEL_PATH)
            print(" ✓ Meilleur modèle sauvegardé!")
        else:
            print()
    
    print("\n" + "="*60)
    print("FIN DE L'ENTRAÎNEMENT")
    print("="*60)
    print(f"Meilleure précision de validation: {best_val_acc:.2f}%")
    print(f"Modèle sauvegardé dans: {MODEL_PATH}")
    
    # Afficher l'accuracy par classe
    print("\n" + "="*60)
    print("ACCURACY PAR CLASSE (VALIDATION)")
    print("="*60)
    class_accuracies = compute_class_accuracy(model, val_loader, device, full_dataset.label_encoder)
    
    # Trier par nom de classe
    for class_name in sorted(class_accuracies.keys()):
        stats = class_accuracies[class_name]
        print(f"{class_name:20s} | Accuracy: {stats['accuracy']:6.2f}% | "
              f"({stats['correct']}/{stats['total']} correct)")
    
    # Calculer la moyenne
    avg_acc = np.mean([stats['accuracy'] for stats in class_accuracies.values()])
    print("-" * 60)
    print(f"{'Moyenne':20s} | Accuracy: {avg_acc:6.2f}%")


if __name__ == "__main__":
    train_model()
