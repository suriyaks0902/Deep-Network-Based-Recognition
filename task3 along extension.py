# Suriya Kasiyalan Siva
# CS5330 Pattern Recognition and Computer Vision
# Spring 2024
# Date : 3/29/2024
#
# PROJECT NUMBER : 5
# PROJECT NAME : Recognition using Deep Networks
#
# TASK 3 : Transfer Learning on Greek Letters
# Description : This code fine-tunes a pre-trained convolutional neural network (CNN) model, originally trained on the MNIST dataset, to recognize Greek letters. 
#               It achieves this by replacing the last layer of the pre-trained model with a new linear layer tailored to classify Greek letters. The script then trains the modified model on a custom dataset containing images of Greek letters. 
#               After training, it evaluates the model's performance on the training set and provides predictions for new input images of Greek letters, along with their corresponding labels. 
#               Finally, it visualizes the predictions alongside the input images.
#

# import dependencies
import sys
import os
from PIL import Image, ImageOps
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from task1_main import Mynetwork  # Ensure this is correctly imported

# Define the Greek letter dataset transform
class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)
    
# Replace the last layer with a new Linear layer with three nodes
def replace_last_layer(model, num_classes=4):
    num_features = model.fc1.out_features  # Adjust this according to your model's layer names
    model.fc2 = nn.Linear(num_features, num_classes)
    print("Number of classes in the model:", num_classes)
    return model

def setup_data_loader(training_set_path):
    # Map class names to integers
    class_to_idx = {'alpha': 0, 'beta': 1, 'delta': 2, 'gamma': 3}
    train_dataset = torchvision.datasets.ImageFolder(
        root=training_set_path,
        transform=transforms.Compose([
            transforms.ToTensor(),
            GreekTransform(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        # Explicitly map class names to integers
        target_transform=lambda label: class_to_idx[train_dataset.classes[label]]
    )

    # Print out transformed labels for debugging
    transformed_labels = [class_to_idx[train_dataset.classes[label]] for label in range(len(train_dataset.classes))]
    print("Transformed labels:", transformed_labels)

    print("Class labels:", train_dataset.classes)
    greek_train = DataLoader(train_dataset, batch_size=5, shuffle=True)
    return greek_train

# Train the model
def train_model(model, train_loader, device, epochs=220):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_history = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    print('Finished Training')
    return loss_history

# Plot training loss history
def plot_loss(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Evaluate the model
def evaluate_model(model, device, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy of the network on the test images: {accuracy*100:.2f}%")

# Preprocess new input images
def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),  # Invert colors
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Predict new inputs
def new_input(model, device, image_paths):
    model.eval()  # Set model to evaluation mode
    predictions = []
    for image_path in image_paths:
        image = preprocess_image(image_path)
        image = image.to(device)
        with torch.no_grad():
            output = model(image)
            prediction = output.argmax(dim=1)
            predictions.append(prediction.item())
    return predictions

def new_input_with_label_mapping(model, device, image_paths):
    class_mapping = {0: 'alpha', 1: 'beta', 2: 'delta', 3: 'gamma'}
    predictions = new_input(model, device, image_paths)  # Use your existing prediction function
    letter_predictions = [class_mapping[pred] for pred in predictions]
    return letter_predictions

def main(argv):
    # Load Greek train dataset
    training_set_path = 'greek_train'
    # Load the pre-trained network created in previous task  
    model = Mynetwork()
    model.load_state_dict(torch.load("mnist_model.pth"))
    print(model)
    input_tensor = torch.randn(1, 1, 28, 28)
    # Pass the input tensor through the model to get the output tensor
    output_tensor = model(input_tensor)
    dot = make_dot(output_tensor, params=dict(model.named_parameters()))
    dot.render("Mynetwork", format="png")
    for param in model.parameters():
        param.requires_grad = False
    model = replace_last_layer(model, num_classes=4)
    output_tensor = model(input_tensor)
    dot = make_dot(output_tensor, params=dict(model.named_parameters()))
    dot.render("New_Mynetwork", format="png")
    device = torch.device("cpu")
    print("Using device:", device)

    # Then, move your model and data loaders to the appropriate device
    model.to(device)
    train_loader = setup_data_loader(training_set_path)
    loss_history = train_model(model, train_loader, device, epochs=120)
    plot_loss(loss_history)
    evaluate_model(model, device, train_loader)  # Ideally, this should be a separate test set

    image_paths = ["GIMG_1.jpg","GIMG_3.jpg", "beta3.png", "beta2.png", "YIMG_7.jpg", "YIMG_8.jpg", "delta_008.png", "delta_002.png"]
    letter_predictions = new_input_with_label_mapping(model, device, image_paths)
    print("Predictions:", letter_predictions)
    # plot to display Greek letter prediction 
    plt.figure(figsize=(9, 5))
    for i in range(len(image_paths)):
        plt.subplot(2, 4, i + 1)
        image = Image.open(image_paths[i])
        plt.imshow(image, cmap='gray')
        plt.legend()
        plt.title(f"Prediction: {letter_predictions[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
