"""
Script d'entrainement avec les hyperparamètres optimisés.
Utilise l'architecture et les paramètres trouvés par getMetalearnedBozo.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import json


# Configuration
CSV_PATH = "dataset_landmarks.csv"
MODEL_PATH = "memotron_models/memotron_model_optimized.pth"
HYPEROPT_RESULTS_PATH = "memotron_models/hyperopt_results.json"
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.2


class GestureDataset(Dataset):
    """Dataset personnalisé pour les landmarks de gestes."""
    
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.labels = self.data['label'].values
        self.features = self.data.drop('label', axis=1).values.astype(np.float32)
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


class DynamicGestureClassifier(nn.Module):
    """
    Réseau de neurones avec architecture dynamique pour la classification de gestes.
    L'architecture est configurée par les hyperparamètres trouvés par Optuna.
    """
    
    def __init__(self, input_size, num_classes, hidden_layers, dropout_rates, use_batch_norm=True, activation='relu'):
        super(DynamicGestureClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Construire les hidden layers
        for i, (hidden_size, dropout_rate) in enumerate(zip(hidden_layers, dropout_rates)):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Fonction d'activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes)) # Couche de sortie
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_hyperparameters():
    """Charge les hyperparamètres optimisés depuis le fichier JSON."""
    if not os.path.exists(HYPEROPT_RESULTS_PATH):
        raise FileNotFoundError(
            f"Fichier d'hyperparamètres non trouvé: {HYPEROPT_RESULTS_PATH}\n"
            "Lancez d'abord getMetalearnedBozo.py pour optimiser les hyperparamètres."
        )
    
    with open(HYPEROPT_RESULTS_PATH, 'r') as f:
        results = json.load(f)
    
    return results['best_params']


def build_architecture(params, input_size):
    """Construit l'architecture à partir des paramètres."""
    n_layers = params['n_layers']
    initial_size = params['initial_hidden_size']
    
    hidden_layers = []
    dropout_rates = []
    
    for i in range(n_layers):
        if i == 0:
            size = initial_size
        else:
            ratio = params[f'layer_{i}_ratio']
            size = max(32, int(hidden_layers[-1] * ratio))
        hidden_layers.append(size)
        dropout_rates.append(params[f'dropout_{i}'])
    
    return hidden_layers, dropout_rates


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entraîne le modèle pour une epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
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
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
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
    
    class_correct = {}
    class_total = {}
    
    for class_name in label_encoder.classes_:
        class_correct[class_name] = 0
        class_total[class_name] = 0
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(labels.size(0)):
                label_idx = labels[i].item()
                class_name = label_encoder.classes_[label_idx]
                class_total[class_name] += 1
                if predicted[i] == labels[i]:
                    class_correct[class_name] += 1
    
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
    """Fonction principale d'entraînement avec les hyperparamètres optimisés."""
    
    params = load_hyperparameters()
    
    print("\n  Hyperparamètres utilisés:")
    print(f"   - Batch size: {params['batch_size']}")
    print(f"   - Learning rate: {params['learning_rate']:.6f}")
    print(f"   - Weight decay: {params['weight_decay']:.6f}")
    print(f"   - Optimizer: {params['optimizer']}")
    print(f"   - Activation: {params['activation']}")
    print(f"   - Batch norm: {params['use_batch_norm']}")
    print(f"   - Scheduler: {params.get('scheduler', 'None') if params['use_scheduler'] else 'None'}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Device: {device}")
    
    # Charger le dataset
    print("\n Chargement du dataset...")
    full_dataset = GestureDataset(CSV_PATH)
    
    num_features = full_dataset.get_num_features()
    num_classes = full_dataset.get_num_classes()
    
    # Construire l'architecture
    hidden_layers, dropout_rates = build_architecture(params, num_features)
    print(f"\n Architecture: {num_features} → {' → '.join(map(str, hidden_layers))} → {num_classes}")
    
    # Split train/val
    train_size = int((1 - VALIDATION_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n📊 Train: {train_size} | Val: {val_size}")
    
    # DataLoaders
    batch_size = params['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Créer le modèle
    model = DynamicGestureClassifier(
        input_size=num_features,
        num_classes=num_classes,
        hidden_layers=hidden_layers,
        dropout_rates=dropout_rates,
        use_batch_norm=params['use_batch_norm'],
        activation=params['activation']
    ).to(device)
    
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    lr = params['learning_rate']
    wd = params['weight_decay']
    
    if params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif params['optimizer'] == 'SGD':
        momentum = params.get('sgd_momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    elif params['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    
    # Scheduler
    scheduler = None
    if params['use_scheduler']:
        scheduler_name = params['scheduler']
        if scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=params['step_size'], 
                gamma=params['gamma']
            )
        elif scheduler_name == 'CosineAnnealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max', 
                patience=params['patience']
            )
    
    # Variables pour suivre le meilleur modèle
    best_val_acc = 0.0
    
    # os.makedirs("models", exist_ok=True)

    print("\n" + "="*60)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("="*60)
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%", end="")
        
        # Sauvegarde le meilleur modèle
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
                'num_classes': num_classes,
                'hyperparameters': params,
                'architecture': {
                    'hidden_layers': hidden_layers,
                    'dropout_rates': dropout_rates,
                    'use_batch_norm': params['use_batch_norm'],
                    'activation': params['activation']
                }
            }, MODEL_PATH)
            print(" ✓ Meilleur modèle sauvegardé!")
        else:
            print()
        
        # Scheduler step
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
    
    print("\n" + "="*60)
    print("FIN DE L'ENTRAÎNEMENT")
    print("="*60)
    print(f"Meilleure précision de validation: {best_val_acc:.2f}%")
    print(f"Modèle sauvegardé dans: {MODEL_PATH}")
    
    # Affiche l'accuracy par classe
    print("\n" + "="*60)
    print("ACCURACY PAR CLASSE (VALIDATION)")
    print("="*60)
    
    # Recharge le meilleur modèle pour l'évaluation finale
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    class_accuracies = compute_class_accuracy(model, val_loader, device, full_dataset.label_encoder)
    
    for class_name in sorted(class_accuracies.keys()):
        stats = class_accuracies[class_name]
        print(f"{class_name:20s} | Accuracy: {stats['accuracy']:6.2f}% | "
              f"({stats['correct']}/{stats['total']} correct)")
    
    avg_acc = np.mean([stats['accuracy'] for stats in class_accuracies.values()])
    print("-" * 60)
    print(f"{'Moyenne':20s} | Accuracy: {avg_acc:6.2f}%")


if __name__ == "__main__":
    train_model()
