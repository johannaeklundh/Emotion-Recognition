# This script uses the ResNet-50 model provided by The MathWorks, Inc.
# Copyright (c) 2019, The MathWorks, Inc.
# Licensed under the terms described in LICENSE-BSD-3-Clause.txt# This script uses the ResNet-50 model provided by The MathWorks, Inc.

import pickle
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, roc_curve, auc, precision_recall_curve, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from itertools import cycle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.models import ResNet50_Weights # Import ResNet50_Weights
from tqdm import tqdm
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
# unzip fer2013.zip -d /datasetFer2013/

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
print ("hello, number of classes:", len(class_names))
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

        # Initialize tqdm progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)


        for batch_idx, (inputs, labels) in progress_bar:
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

            # Update tqdm progress bar
            progress_bar.set_postfix({
            'Loss': f"{train_running_loss / (batch_idx + 1):.4f}",
            'Accuracy': f"{100. * correct / total:.2f}%"
            })

        # Calculate average loss for the training epoch
        avg_train_loss = train_running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Calculate training accuracy
        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)


        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Training Accurcacy: {train_accuracy:.4f}")
    
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

        print(f"Accuracy: {100 * correct / total}%")
        
        # Calculate average loss for the validation epoch
        avg_val_loss = val_running_loss / len(valid_loader)
        val_losses.append(avg_val_loss)
        # Calculate validation accuracy
        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)
        # Append results to the probs, true_labels, pred_labels lists
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

    ## TEST ##
    # Get predictions on the test set
    model.eval()

    test_predictions = []
    test_true_labels = []
    test_predicted_probs = []

    with torch.no_grad(): # no need to compute gradients here
        # Iterate over the batches in test_loader
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) # move to device
            outputs = model(images) # compute output
            _, predicted = torch.max(outputs, 1) # get prediction
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())
            # Get predicted probabilities
            probs = F.softmax(outputs, dim=1)
            test_predicted_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(test_true_labels, test_predictions)
    precision = precision_score(test_true_labels, test_predictions, average='weighted', zero_division=0)
    recall = recall_score(test_true_labels, test_predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_true_labels, test_predictions, average='weighted')
    roc_auc = roc_auc_score(test_true_labels, test_predicted_probs, multi_class='ovr', average='weighted')

    # Print the results
    print('\nTest Results')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print(f'Test ROC AUC: {roc_auc:.4f}')

    # Test Confusion Matrix
    test_conf_matrix = confusion_matrix(test_true_labels, test_predictions)
    
    # Put results in a dictionary
    results = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'true_labels': test_true_labels,
        'pred_labels': test_predictions,
        'probs': test_predicted_probs,
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

def plot_figures(results, path, num_epochs):
    '''
    Function that, given training, validation and test results, plots and saves images
    Args:
    - results (dict): Dictionary containing training, validation and test results
    - path (str): Path to save images
    - num_epochs (int): Number of epochs
    Returns:
    - None
    '''
    train_losses = results['train_losses']
    val_losses = results['val_losses']
    train_accuracies = results['train_accuracies']
    val_accuracies = results['val_accuracies']
    true_labels = results['true_labels']
    pred_labels = results['pred_labels']
    probs = results['probs']
    conf_matrix = results['conf_matrix']
    test_conf_matrix = results['test_conf_matrix']

    # Ensure the directory exists
    output_dir = os.path.dirname(path[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(15, 16))

    # Training and validation loss curves
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(range(1, num_epochs+1), train_losses, label='Trainining Loss')
    ax1.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid()

    # Training and validation accuracy curves
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
    ax2.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid()

    # ROC curve
    # Ensure test_true_labels is a 1D array
    test_true_labels = np.array(true_labels).ravel()
    # Ensure test_predicted_probs is a 2D array with the same number of samples as test_true_labels
    test_predicted_probs = np.array(probs)
    # Binarize the true labels for multi-class ROC curve
    true_labels_binarized = label_binarize(test_true_labels, classes=range(len(class_names)))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(true_labels_binarized[:,i], test_predicted_probs[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    fig, ax = plt.subplots(figsize=(15, 16))

    ax3 = plt.subplot(3, 2, 3)
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "purple", "yellow"])
    for class_id, color in zip(range(len(class_names)), colors):
        RocCurveDisplay.from_predictions(
        true_labels_binarized[:, class_id],
        test_predicted_probs[:, class_id],
        name=f"ROC curve for {class_names[class_id]} (AUC = {roc_auc[class_id]:.2f})",
        color=color,
        ax=ax3,
        plot_chance_level=False,
        despine=True,
)

    ax3.plot([0, 1], [0, 1], 'k--', label='Chance level')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax3.legend(loc="lower right")
    ax3.grid()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(true_labels_binarized[-1], probs[-1])
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(recall, precision, color='green', label='Precision-Recall Curve')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision-Recall Curve')
    ax4.legend()
    ax4.grid()
    
    # Plot the confusion matrix
    ax5 = plt.subplot(3,2,5)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=True, xticklabels=class_names, yticklabels=class_names, ax=ax5)
    ax5.set_title('Validation Confusion Matrix (last epoch)')
    ax5.set_xlabel('Predicted Labels')
    ax5.set_ylabel('True Labels')

    # Confusion Matrix Testing
    ax6 = plt.subplot(3,2,6)
    sns.heatmap(test_conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=True, xticklabels=class_names, yticklabels=class_names, ax=ax6)
    ax6.set_title('Test Confusion Matrix')
    ax6.set_xlabel('Predicted Labels')
    ax6.set_ylabel('True Labels')

   # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(path[0])
    plt.close()

def print_result (results, num_epochs):

    train_accuracies = results['train_accuracies']
    val_accuracies = results['val_accuracies']
    recall = results['recall']
    f1 = results['f1']
    roc = results['roc_auc']

    train_accuracies_last = train_accuracies[num_epochs -1]
    val_accuracies_last= val_accuracies[num_epochs-1]
    print('Train Accuracy: ', train_accuracies_last, '\nValid Accuracy: ', val_accuracies_last, '\nRecall: ', recall, '\nF1: ', f1, '\nRoc: ', roc)


def load_experiment(result_path):
    with open(result_path, 'rb') as f:
        loaded_results = pickle.load(f)
    print("succsessfully loaded: " + result_path)
    #print("Loaded training results:", loaded_results)
    return loaded_results

# creates folders to save checkpoints and results
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('result', exist_ok=True)
os.makedirs('resnet50_result_Images', exist_ok=True)

# Restarting the training
# Start fresh: Reset model weights and optimizer state
# model.apply(lambda module: module.reset_parameters() if hasattr(module, 'reset_parameters') else None)  # Reset model weights
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Reinitialize optimizer
# start_epoch = 0 # Ensure we start from epoch 0
# print("Starting training from epoch:", start_epoch)

# To conitune training  ---- needs to be after the initialization of optimizer -----
# #Load previous epochs (training)
# checkpoint = torch.load('checkpoint.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# start_epoch = checkpoint['epoch'] + 1  # This should be 1 for the second epoch
# print("Starting training from epoch:", start_epoch)

## BASELINE ## 
result_path = "result/result_CE_SGD_BASELINE.pkl"
checkpoint_path = "checkpoints/result_CE_SGD_BASELINE.pth"
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
train(model=model, train_loader=train_loader, valid_loader=val_loader, test_loader=test_loader, class_names = class_names, criterion=criterion, opt=optimizer, epochs=8, checkpoint_path=checkpoint_path, result_path=result_path)

## ADDING a custom classification layer ##
# Load the baseline model
checkpoint = torch.load('checkpoints/result_CE_SGD_BASELINE.pth')  # Replace with your checkpoint path
model.load_state_dict(checkpoint['model_state_dict'])

# Freeze earlier layers to focus on training the new layers initially
for param in model.parameters():
    param.requires_grad = False

# Wrap the existing model.fc
model.fc = nn.Sequential(
    model.fc,                  # Existing fc layer (already trained during baseline)
    nn.ReLU(),                 # Add ReLU activation for non-linearity
    nn.Dropout(0.5),           # Add dropout for regularization
    nn.Linear(7, 256),         # Additional layer with 256 neurons
    nn.ReLU(),                 # Add ReLU for the new layer
    nn.Dropout(0.5),           # Add another dropout
    nn.Linear(256, 7)          # Output layer for 7 emotion classes
)

# Print the model to verify the changes
print(model)

result_path_added = "result/result_CE_SGD_Added.pkl"
checkpoint_path_added = "checkpoints/result_CE_SGD_Added.pth"
# Set the optimizer to update only the new layers initially
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
train(model=model, train_loader=train_loader, valid_loader=val_loader, test_loader=test_loader, class_names = class_names, criterion=criterion, opt=optimizer, epochs=8, checkpoint_path=checkpoint_path_added, result_path=result_path_added)

results_result_CE_SGD_BASELINE = load_experiment('result/result_CE_SGD_BASELINE.pkl')
result_CE_SGD_BASELINE_paths_plots = ['resnet50_result_Images/result_CE_SGD_BASELINE.png'];
plot_figures(results_result_CE_SGD_BASELINE, result_CE_SGD_BASELINE_paths_plots, 8)
print_result(results_result_CE_SGD_BASELINE, 8)

results_result_CE_SGD_Added = load_experiment('result/result_CE_SGD_Added.pkl')
result_CE_SGD_Added_paths_plots = ['resnet50_result_Images/result_CE_SGD_Added.png'];
plot_figures(results_result_CE_SGD_Added, result_CE_SGD_Added_paths_plots, 8)
print_result(results_result_CE_SGD_Added, 8)