"""
Meta Learning pour l'optimisation des hyperparamètres.
Utilise Optuna pour trouver automatiquement les meilleurs hyperparamètres.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import optuna
from optuna.trial import TrialState
import json
from datetime import datetime


# Configuration de base
CSV_PATH = "dataset_landmarks.csv"
MODEL_PATH = "memotron_models/memotron_model_optimized.pth"
N_TRIALS = 100  # Nombre d'essais pour l'optimisation
TIMEOUT = 3600  # Timeout en secondes
VALIDATION_SPLIT = 0.2
N_EPOCHS_PER_TRIAL = 60  # Le nom est plutot explicite


class GestureDataset(Dataset):
    """Dataset personnalisé pour les landmarks de gestes."""
    
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.labels = self.data['label'].values
        self.features = self.data.drop('label', axis=1).values.astype(np.float32)
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
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
        
        # fonction d'activation
        activation_fn = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }.get(activation, nn.ReLU())
        
        # couches cachées
        for i, (hidden_size, dropout_rate) in enumerate(zip(hidden_layers, dropout_rates)):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Cloner l'activation pour chaque couche
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
        
        # couche de sortie
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def create_model(trial, input_size, num_classes):
    """
    Crée un modèle avec des hyperparamètres suggérés par Optuna.
    """
    # Nombre de couches cachées (entre 2 et 5)
    n_layers = trial.suggest_int('n_layers', 2, 5)
    
    hidden_layers = []
    dropout_rates = []
    
    # Taille initiale (entre 128 et 1024)
    initial_size = trial.suggest_int('initial_hidden_size', 128, 1024, step=64)
    
    for i in range(n_layers):
        # Chaque couche peut réduire ou maintenir la taille
        if i == 0:
            size = initial_size
        else:
            # Ratio de réduction entre couches (0.3 à 1.0)
            ratio = trial.suggest_float(f'layer_{i}_ratio', 0.3, 1.0)
            size = max(32, int(hidden_layers[-1] * ratio))
        
        hidden_layers.append(size)
        
        # Dropout pour cette couche (0.1 à 0.5)
        dropout = trial.suggest_float(f'dropout_{i}', 0.1, 0.5)
        dropout_rates.append(dropout)
    
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False]) # Batch normalization
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'gelu', 'silu']) # Fonction d'activation
    
    model = DynamicGestureClassifier(
        input_size=input_size,
        num_classes=num_classes,
        hidden_layers=hidden_layers,
        dropout_rates=dropout_rates,
        use_batch_norm=use_batch_norm,
        activation=activation
    )
    
    return model


def objective(trial, train_dataset, val_dataset, input_size, num_classes, device):
    """
    Fonction objectif pour Optuna.
    Retourne la meilleure accuracy de validation pour un ensemble d'hyperparamètres.
    """
    
    # Hyperparamètres d'entraînement
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD', 'RMSprop'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = create_model(trial, input_size, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss() # Loss

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = trial.suggest_float('sgd_momentum', 0.0, 0.99)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    use_scheduler = trial.suggest_categorical('use_scheduler', [True, False])
    scheduler = None
    if use_scheduler:
        scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealing', 'ReduceLROnPlateau'])
        if scheduler_name == 'StepLR':
            step_size = trial.suggest_int('step_size', 5, 20)
            gamma = trial.suggest_float('gamma', 0.1, 0.9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'CosineAnnealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_PER_TRIAL)
        elif scheduler_name == 'ReduceLROnPlateau':
            patience = trial.suggest_int('patience', 3, 10)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience)
    
    best_val_acc = 0.0
    
    # Entraînement
    for epoch in range(N_EPOCHS_PER_TRIAL):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Scheduler step
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # Pruning (arret précoce, pas obligé mais plus rapide)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_acc


def train_with_best_params(best_params, train_dataset, val_dataset, full_dataset, device, n_epochs=100):
    """
    Entraîne le modèle final avec les meilleurs hyperparamètres trouvés.
    """
    input_size = full_dataset.get_num_features()
    num_classes = full_dataset.get_num_classes()
    
    # Reconstruire l'architecture à partir des paramètres
    n_layers = best_params['n_layers']
    initial_size = best_params['initial_hidden_size']
    
    hidden_layers = []
    dropout_rates = []
    
    for i in range(n_layers):
        if i == 0:
            size = initial_size
        else:
            ratio = best_params[f'layer_{i}_ratio']
            size = max(32, int(hidden_layers[-1] * ratio))
        hidden_layers.append(size)
        dropout_rates.append(best_params[f'dropout_{i}'])
    
    model = DynamicGestureClassifier(
        input_size=input_size,
        num_classes=num_classes,
        hidden_layers=hidden_layers,
        dropout_rates=dropout_rates,
        use_batch_norm=best_params['use_batch_norm'],
        activation=best_params['activation']
    ).to(device)
    
    batch_size = best_params['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    lr = best_params['learning_rate']
    wd = best_params['weight_decay']
    
    if best_params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif best_params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif best_params['optimizer'] == 'SGD':
        momentum = best_params.get('sgd_momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    elif best_params['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    
    scheduler = None
    if best_params['use_scheduler']:
        scheduler_name = best_params['scheduler']
        if scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=best_params['step_size'], 
                gamma=best_params['gamma']
            )
        elif scheduler_name == 'CosineAnnealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max', 
                patience=best_params['patience']
            )
    
    best_val_acc = 0.0
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT AVEC LES MEILLEURS HYPERPARAMÈTRES")
    print("="*60)
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{n_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%", end="")
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'label_encoder': full_dataset.label_encoder,
                'num_features': input_size,
                'num_classes': num_classes,
                'hyperparameters': best_params,
                'architecture': {
                    'hidden_layers': hidden_layers,
                    'dropout_rates': dropout_rates,
                    'use_batch_norm': best_params['use_batch_norm'],
                    'activation': best_params['activation']
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
    
    return best_val_acc, model


def print_best_params(study):
    """Affiche les meilleurs hyperparamètres trouvés."""
    print("\n" + "="*60)
    print("MEILLEURS HYPERPARAMÈTRES TROUVÉS")
    print("="*60)
    
    best_trial = study.best_trial
    
    print(f"\nMeilleure accuracy de validation: {best_trial.value:.2f}%")
    print(f"Numéro du trial: {best_trial.number}")
    print("\nHyperparamètres:")
    
    # Grouper les paramètres par catégorie
    arch_params = {}
    train_params = {}
    
    for key, value in best_trial.params.items():
        if key in ['n_layers', 'initial_hidden_size', 'use_batch_norm', 'activation'] or key.startswith('layer_') or key.startswith('dropout_'):
            arch_params[key] = value
        else:
            train_params[key] = value
    
    print("\n📐 Architecture:")
    for key, value in sorted(arch_params.items()):
        print(f"  - {key}: {value}")
    
    print("\n🎯 Entraînement:")
    for key, value in sorted(train_params.items()):
        if isinstance(value, float):
            print(f"  - {key}: {value:.6f}")
        else:
            print(f"  - {key}: {value}")


def run_meta_learning():
    """Fonction principale de Meta Learning."""
    
    print("="*60)
    print("META LEARNING - OPTIMISATION DES HYPERPARAMÈTRES")
    print("="*60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Device: {device}")
    
    # Charger le dataset
    print("\nChargement du dataset...")
    full_dataset = GestureDataset(CSV_PATH)
    
    input_size = full_dataset.get_num_features()
    num_classes = full_dataset.get_num_classes()
    
    print(f"   - Nombre d'échantillons: {len(full_dataset)}")
    print(f"   - Nombre de features: {input_size}")
    print(f"   - Nombre de classes: {num_classes}")
    
    # Split train/val
    train_size = int((1 - VALIDATION_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Seed fixe pour reproductibilité
    )
    
    print(f"   - Train: {train_size} | Val: {val_size}")
    
    # Créer l'étude Optuna
    print(f"\nDébut de l'optimisation ({N_TRIALS} trials, timeout: {TIMEOUT}s)")
    print("-"*60)
    
    # Utiliser MedianPruner pour arreter tot les essais non prometteurs
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        direction='maximize',  # Maximiser l'accuracy
        pruner=pruner,
        study_name='memotron_hyperopt'
    )
    
    # Callback pour afficher la progression
    def callback(study, trial):
        if trial.state == TrialState.COMPLETE:
            print(f"Trial {trial.number:3d} | Val Acc: {trial.value:.2f}% | "
                  f"Best: {study.best_value:.2f}%")
        elif trial.state == TrialState.PRUNED:
            print(f"Trial {trial.number:3d} | Pruned (non prometteur)")
    
    # Lancer l'optimisation
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, input_size, num_classes, device),
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        callbacks=[callback],
        show_progress_bar=True
    )
    print_best_params(study)
    
    # Sauvegarde des résultats dans un fichier JSON
    results = {
        'best_params': study.best_trial.params,
        'best_value': study.best_trial.value,
        'n_trials': len(study.trials),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = 'models/hyperopt_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats sauvegardés dans: {results_path}")
    
    print("\n" + "="*60)
    print("ENTRAINEMENT DU MODELE FINAL")
    print("="*60)
    
    final_acc, final_model = train_with_best_params(
        study.best_trial.params,
        train_dataset,
        val_dataset,
        full_dataset,
        device,
        n_epochs=100
    )
    
    print("\n" + "="*60)
    print("FINITO")
    print("="*60)
    print(f"Meilleure accuracy finale: {final_acc:.2f}%")
    print(f"Modèle optimisé sauvegardé dans: {MODEL_PATH}")
    
    return study, final_model


if __name__ == "__main__":
    run_meta_learning()
