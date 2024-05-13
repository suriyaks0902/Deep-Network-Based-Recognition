# Suriya Kasiyalan Siva
# CS5330 Pattern Recognition and Computer Vision
# Spring 2024
# Date : 4/03/2024
#
# PROJECT NUMBER : 5
# PROJECT NAME : Recognition using Deep Networks
#
# Extension : Build a live video digit recognition application using the trained network
# Description : This Python script conducts live digit recognition using a pre-trained CNN model on the MNIST dataset. 
#               It loads the data, splits it for training and testing, and loads the pre-trained model. 
#               Then, it defines functions for training, testing, preprocessing frames, recognizing digits, and performing majority voting. 
#               Finally, it displays live video feed with predicted digits and terminates upon pressing 'q'.
#

# Import necessary libraries
import sys
from matplotlib import axes
import torch
import cv2
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from task1_main import Mynetwork
from PIL import ImageOps

# Define a function to preprocess the frame
def preprocess_frame(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize frame to 28x28 pixels
    resized_frame = cv2.resize(gray_frame, (28, 28))
    # Invert colors (since MNIST digits are white on black background)
    inverted_frame = cv2.bitwise_not(resized_frame)
    # Convert frame to tensor and add batch dimension
    tensor_frame = ToTensor()(inverted_frame).unsqueeze(0)
    return tensor_frame

# Training function
def train(model, device, train_loader, val_loader, criterion, optimizer, epochs=5):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    return train_losses, val_losses

# Evaluate the model on the test set
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.2%}')
    
# Define a function to perform digit recognition
def recognize_digit(frame, device, model):
    with torch.no_grad():
        output = model(frame.to(device))  # Move frame to device before passing it to the model
        prediction = torch.argmax(F.softmax(output, dim=1)).item()
    return prediction

def main(argv):
    # Load train data:
    # Load MNIST dataset with augmentation
    # Define data transformation with augmentation
    
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True
    )
    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor(),
        download=True
    )
    
    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False
    )
    # Split the training dataset into train and validation sets
    train_size = int(0.8 * len(train_data))
    val_size = len(test_data) - train_size  
    device = torch.device('cpu')
    # Load the trained model
    model = Mynetwork()
    model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device('cpu')))
    model.eval() 
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses, val_losses = train(model, device, train_loader, test_loader, criterion, optimizer, epochs=10)
    test(model, device, test_loader)
    # Start video stream from the webcam
    cap = cv2.VideoCapture(0)
    window_size = 5
    prediction_window = []
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Preprocess frame
        processed_frame = preprocess_frame(frame)
        print("Frame preprocessed.")
        # Perform digit recognition
        digit_prediction = recognize_digit(processed_frame, device, model)
        print("Digit recognized:", digit_prediction)
        # Update prediction window
        if len(prediction_window) < window_size:
            prediction_window.append(digit_prediction)
        else:
            prediction_window.pop(0)
            prediction_window.append(digit_prediction)
        
        # Perform majority voting
        final_prediction = max(set(prediction_window), key=prediction_window.count)
        # Overlay predicted digit on frame
        cv2.putText(frame, str(final_prediction), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Display the resulting frame
        cv2.imshow('Live Digit Recognition', frame)
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video stream and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)