import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

def supervised_training(model, train_loader, val_loader, test_loader, optimizer, num_epochs=10, device='cpu'):
    model = model.to(device)
    criterion = nn.BCELoss()

    pbar = tqdm(range(num_epochs), desc='Epochs')
    best_val_loss = float('inf')
    best_val_epoch = 0
    best_state = None
    train_losses = []
    val_losses = []
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
        pbar.set_description(f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_state = model.state_dict()
        
    model.load_state_dict(best_state)
    
    model = model.eval()
    # evaluate on test set
    with torch.no_grad():
        evaluations = []
        labs = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            evaluations.append(outputs.cpu().numpy())
            labs.append(labels.numpy())
        auc = roc_auc_score(np.concatenate(labs), np.concatenate(evaluations))

    plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(train_losses))+1, train_losses, label='Train Loss')
    plt.plot(np.arange(len(val_losses))+1, val_losses, label='Validation Loss')
    plt.axvline(best_val_epoch+1, color='k', linestyle='--', label='Best Val Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"Test set AUC: {auc:.4f}")

    return model

def contrastive_training(model, train_loader, val_loader, test_loader, optimizer, num_epochs=10, device='cpu'):
    model = model.to(device)
    criterion = nn.BCELoss()

    pbar = tqdm(range(num_epochs), desc='Epochs')
    best_val_loss = float('inf')
    best_val_epoch = 0
    best_state = None
    train_losses = []
    val_losses = []
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
        pbar.set_description(f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_state = model.state_dict()
        
    model.load_state_dict(best_state)
    
    model = model.eval()
    # evaluate on test set
    with torch.no_grad():
        evaluations = []
        labs = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            evaluations.append(outputs.cpu().numpy())
            labs.append(labels.numpy())
        auc = roc_auc_score(np.concatenate(labs), np.concatenate(evaluations))

    plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(train_losses))+1, train_losses, label='Train Loss')
    plt.plot(np.arange(len(val_losses))+1, val_losses, label='Validation Loss')
    plt.axvline(best_val_epoch+1, color='k', linestyle='--', label='Best Val Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"Test set AUC: {auc:.4f}")

    return model