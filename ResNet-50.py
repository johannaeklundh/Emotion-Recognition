# This script uses the ResNet-50 model provided by The MathWorks, Inc.
# Copyright (c) 2019, The MathWorks, Inc.
# Licensed under the terms described in LICENSE-BSD-3-Clause.txt# This script uses the ResNet-50 model provided by The MathWorks, Inc.

import pickle
import random
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.models import ResNet50_Weights # Import ResNet50_Weights
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("PyTorch version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

# Set random seeds for reproducibility of results.
seed=42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# TO DOWNLOAD THE DATASET, RUN THIS IN TERMINAL WHERE YOU WAN TO STORE IT
# kaggle datasets download -d msambare/fer2013
# unzip fer2013zip -d /datasetFer2013/

# Define the paths to the dataset directories
train_dir = '../dataset/train'
test_dir = '../dataset/test'

# Data transformations
transform50 = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize((224, 224)),               # Resize for ResNet
    transforms.ToTensor(),                       # Convert to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

# Load datasets
train_dataset = datasets.ImageFolder(root='../dataset/train', transform=transform50)
test_dataset = datasets.ImageFolder(root='../dataset/test', transform=transform50)

# Print class names
class_names = train_dataset.classes
print("Classes:", class_names)

# Define the validation split ratio
val_size = int(0.2 * len(train_dataset))  # 20% for validation
train_size = len(train_dataset) - val_size
# Randomly split the dataset
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
# Create DataLoaders for the training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Shuffle enabled
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)     # No shuffle for validation
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Load pre-trained ResNet-50
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Modify the final layer for 7 emotion classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Freeze the convolutional and intermediate layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the final fully connected layer (the last layer)
for param in model.fc.parameters():
    param.requires_grad = True

def train(model, train_loader, valid_loader, test_loader, class_names, criterion, opt, epochs, checkpoint_path, result_path):
    # Set device to cuda if it is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device) # Move model to the right device

    # Initialize lists for accuracies, losses, Grad-CAM images, probabilities of predicting class 1
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    true_labels = []
    pred_labels = []
    probs = []

    for epoch in range(epochs):  # Number of epochs
        print(f"Epoch {epoch + 1}/{epochs}") 

        model.train()
        train_running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            opt.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            opt.step()

            train_running_loss += loss.item()
            # Calculate accuracy
            predicted = torch.max(outputs, 1)[1]  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate average loss for the training epoch
        avg_train_loss = train_running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Calculate training accuracy
        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)

        print(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}, Accurcacy: {train_accuracy:.4f}")
    
        ## Validation phase ##
        model.eval()
        val_running_loss = 0.0
        val_correct, val_total = 0, 0

        # Initialize label lists for this epoch
        epoch_true_labels = []
        epoch_pred_labels = []

        # Iterate over the batches in valid_loader
        for val_inputs, val_labels in valid_loader:
            # Move to device
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            with torch.no_grad():  # no need to compute gradients here
                # Forward pass
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                val_running_loss += val_loss.item() # Increase validation loss
                # Calculate accuracy
                val_predicted = torch.max(val_outputs, 1)[1]  # Get predicted class
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

                # Memorize true and predicted labels for explainability
                epoch_true_labels.extend(val_labels.cpu().numpy())
                epoch_pred_labels.extend(val_predicted.cpu().numpy())

            # Calculate probabilities for prediction = class 1 (real) using softmax
            # prob.extend(torch.softmax(val_outputs, dim=1)[:, 1].cpu().detach().numpy())

        print(f"Accuracy: {100 * correct / total}%")
        
        # Calculate average loss for the validation epoch
        avg_val_loss = val_running_loss / len(valid_loader)
        val_losses.append(avg_val_loss)
        # Calculate validation accuracy
        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)
        # Append results to the probs, true_labels, pred_labels lists
        # probs.append(prob)
        true_labels.append(epoch_true_labels)
        pred_labels.append(epoch_pred_labels)

        print(f'Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}')
        
        # Save the checkpoint at the end of each epoch to be able to continue the training later
        torch.save({
            'epoch': epoch +1, # Save the next epoch for proper resumption
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'valid_loss': avg_val_loss  # Save validation loss for reference
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")
    
    print('Finished Training')

    #Confusion Matrix
    # conf_matrix = confusion_matrix((true_labels[-1], pred_labels[-1]))
    
    ## TEST ##
    # Get predictions on the test set
    model.eval()

    test_predictions = []
    test_true_labels = []

    with torch.no_grad(): # no need to compute gradients here
        # Iterate over the batches in test_loader
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) # move to device
            outputs = model(images) # compute output
            _, predicted = torch.max(outputs, 1) # get prediction
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(test_true_labels, test_predictions)
    precision = precision_score(test_true_labels, test_predictions)
    recall = recall_score(test_true_labels, test_predictions)
    f1 = f1_score(test_true_labels, test_predictions)
    roc_auc = roc_auc_score(test_true_labels, test_predictions)

    # Print the results
    print('\nTest Results')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print(f'Test ROC AUC: {roc_auc:.4f}')

    # Test Confusion Matrix
    test_conf_matrix = confusion_matrix(test_true_labels, test_predictions)

    report = classification_report(true_labels, pred_labels, target_names=class_names)
    print("\nClassification Report:\n", report)
    
    # Put results in a dictionary
    results = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'true_labels': true_labels,
        'pred_labels': pred_labels,
        'probs': probs,
        'test_true_labels': test_true_labels,
        'test_predictions': test_predictions,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'conf_matrix': test_conf_matrix,
        'test_conf_matrix': test_conf_matrix
      }
    # Save the efficientnet_results dictionary to a file
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)

    print("Training results saved to ", result_path)
    
    return model, results

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('result', exist_ok=True)
result_path = "result/result_CE_SGD_BASELINE.pkl"
checkpoint_path = "checkpoints/result_CE_SGD_BASELINE.pth"
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
train(model=model, train_loader=train_loader, valid_loader=val_loader, test_loader=test_loader, class_names = class_names, criterion=criterion, opt=optimizer, epochs=3, checkpoint_path=checkpoint_path, result_path=result_path)
