import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from configs import *
from data_generator import *
from model import *

def create_train_val_datasets(repo_id,wsi_files_with_metadata,filtered_metadata,test_size=0.2,random_state=42,num_patches=32,patch_size=224,patch_level=0,verbose=False):
    from sklearn.model_selection import train_test_split
    from collections import Counter

    # Create label list for stratification
    file_labels = []
    for file_path in wsi_files_with_metadata:
        case_id = file_path.split('/')[0]
        icd10_code = filtered_metadata.get(case_id, {}).get('icd10', 'Unknown')
        file_labels.append(icd10_code)

    if verbose:
        print("Label distribution in full dataset:")
        label_counts = Counter(file_labels)
        for label, count in sorted(label_counts.items()):
            print(f"   {label}: {count} files")

    # Attempt stratified split to maintain class balance
    try:
        train_files, val_files, train_labels, val_labels = train_test_split(
            wsi_files_with_metadata,
            file_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=file_labels
        )

        if verbose:
            print(f"\n Stratified split successful:")
            print(f"   Train: {len(train_files)} files")
            print(f"   Validation: {len(val_files)} files")

            print(f"\n Train label distribution:")
            train_counts = Counter(train_labels)
            for label, count in sorted(train_counts.items()):
                print(f"   {label}: {count} files")

            print(f"\n Validation label distribution:")
            val_counts = Counter(val_labels)
            for label, count in sorted(val_counts.items()):
                print(f"   {label}: {count} files")

    except ValueError as e:
        if verbose:
            print(f"Stratification failed: {e}")
            print("Falling back to random split...")

        train_files, val_files = train_test_split(
            wsi_files_with_metadata,
            test_size=test_size,
            random_state=random_state
        )

        if verbose:
            print(f"Random split completed:")
            print(f"   Train: {len(train_files)} files")
            print(f"   Validation: {len(val_files)} files")

    # Create datasets
    if verbose:
        print(f"\n Creating train dataset...")

    train_dataset = WSI_MONAI_Dataset(
        repo_id=repo_id,
        file_list=train_files,
        metadata=filtered_metadata,
        num_patches=num_patches,
        patch_size=patch_size,
        patch_level=patch_level
    )

    if verbose:
        print(f"\n Creating validation dataset...")

    val_dataset = WSI_MONAI_Dataset(
        repo_id=repo_id,
        file_list=val_files,
        metadata=filtered_metadata,
        num_patches=num_patches,
        patch_size=patch_size,
        patch_level=patch_level
    )

    if verbose:
        print(f"\n Datasets created successfully!")
        print(f"   Train dataset: {len(train_dataset)} samples")
        print(f"   Val dataset: {len(val_dataset)} samples")
        print(f"   Classes: {train_dataset.icd10_labels}")

    return train_dataset, val_dataset


def setup_data_loaders(train_dataset, val_dataset, batch_size=16, num_workers=2):
    """
    Setup data loaders for training and validation
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

def train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip=1.0):
    """
    Train the model for one epoch
    """
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for patches, labels in tqdm(train_loader, desc="Training", leave=False):
        patches, labels = patches.to(device), labels.to(device)

        # Forward pass
        logits = model(patches)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Accumulate loss and predictions
        running_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for patches, labels in tqdm(val_loader, desc="Validation", leave=False):
            patches, labels = patches.to(device), labels.to(device)

            # Forward pass
            logits = model(patches)
            loss = criterion(logits, labels)

            # Accumulate loss and predictions
            running_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, num_classes, device, num_epochs=25, learning_rate=1e-4):
    """
    Train the EXAONEPath classifier
    """
    # Loss function for classification
    criterion = nn.CrossEntropyLoss()

    # Optimizer only for trainable parts
    optimizer, scheduler = setup_optimizer(model, learning_rate=learning_rate)

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate after each epoch
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Logging
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_{num_classes}_classes.pth")
            print(f"Best model saved with val_acc: {val_acc:.4f}")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")

def run_training_pipeline(train_dataset, val_dataset, num_classes, num_epochs=5, batch_size=16):
    """
    End-to-end training pipeline
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup data loaders
    train_loader, val_loader = setup_data_loaders(train_dataset, val_dataset, batch_size=batch_size)

    # Initialize model
    model = create_wsi_model(num_classes=num_classes)
    model = model.to(device)

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device,
        num_epochs=num_epochs,
        learning_rate=1e-4
    )