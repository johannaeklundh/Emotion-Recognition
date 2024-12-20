# This script uses the ResNet-50 model provided by The MathWorks, Inc.
# Copyright (c) 2019, The MathWorks, Inc.
# Licensed under the terms described in LICENSE-BSD-3-Clause.txt# This script uses the ResNet-50 model provided by The MathWorks, Inc.

import random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
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

def preprocess_image(img): # ResNet expect 3-channel images with size 224x224
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (224, 224))  # ResNet-50 input size
    return img / 255.0  # Normalize pixel values


# Define the paths to the dataset directories
train_dir = '../dataset/train'
test_dir = '../dataset/test'

# Data transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize((224, 224)),               # Resize for ResNet
    transforms.ToTensor(),                       # Convert to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

# Load datasets
train_dataset = datasets.ImageFolder(root='../dataset/train', transform=transform)
test_dataset = datasets.ImageFolder(root='../dataset/test', transform=transform)

# Print class names
class_names = train_dataset.classes
print("Classes:", class_names)

# Define the validation split ratio
val_size = int(0.2 * len(train_dataset))  # 20% for validation
train_size = len(train_dataset) - val_size
# Randomly split the dataset
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
# Create DataLoaders for the training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Shuffle enabled
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)     # No shuffle for validation
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Load pre-trained ResNet-50
model = resnet50(pretrained=True)

# Modify the final layer for 7 emotion classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# def train(model):
#     for epoch in range(10):  # Number of epochs
#     model.train()
#     running_loss = 0.0

#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# def evaluation():
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     print(f"Accuracy: {100 * correct / total}%")

# train(model=model)
# evaluation