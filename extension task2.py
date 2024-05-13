# Suriya Kasiyalan Siva
# CS5330 Pattern Recognition and Computer Vision
# Spring 2024
# Date : 4/02/2024
#
# PROJECT NUMBER : 5
# PROJECT NAME : Recognition using Deep Networks
#
# Extension : (use pre-trained networks available in the PyTorch package) and (Gabor filter in first layer)
# Description : This Python script visualizes filters and filtered images for convolutional layers in neural networks. 
#               It toggles between a custom CNN model (Mynetwork1) with Gabor filters and a pre-trained ResNet-18 model (models.resnet18) by pressing the 'g' key. 
#               It loads MNIST dataset images, plots the first six images with their labels, and visualizes filters and filtered images for the first and second convolutional layers. 
#               The apply_filters function applies filters to input images, and apply_filters1 applies Gabor filters specifically for the custom CNN model. 
#               Parameters of the Gabor layer in the custom CNN model are frozen to prevent updates during training.

# import dependencies
import sys
import keyboard
from matplotlib import axes
import torch
import cv2
import numpy as np
from torchvision import datasets
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from task1_main import Mynetwork

def apply_filters(image, model, layer_index):
    filtered_images = []
    
    with torch.no_grad():
        for i in range(10):
            # Convert the weight tensor to numpy array
            kernel = model.conv1.weight[i, 0].cpu().numpy()
            # Apply the filter using OpenCV's filter2D function
            filtered_image = cv2.filter2D(image, -1, kernel)
            filtered_images.append(filtered_image)
    return filtered_images

# Define Gabor filters as the first layer weights
class GaborLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(GaborLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)

        # Initialize Gabor filters as the weights of the convolutional layer
        self.conv1.weight.data = torch.tensor(self.get_gabor_filters(out_channels), dtype=torch.float32)
        self.conv1.weight.requires_grad = False  # Freeze the parameters

    def forward(self, x):
        return F.relu(self.conv1(x))

    def get_gabor_filters(self, num_filters):
        filters = []
        for i in range(num_filters):
            theta = np.pi * i / num_filters  # Rotation angle for each filter
            kernel = cv2.getGaborKernel((5, 5), sigma=2, theta=theta, lambd=5, gamma=0.5, psi=0, ktype=cv2.CV_32F)
            filters.append(kernel)
        filters = np.array(filters)
        filters = filters.reshape(num_filters, 1, 5, 5)
        return filters

# Define the modified network
class Mynetwork1(nn.Module):
    def __init__(self):
        super(Mynetwork1, self).__init__()
        self.conv1 = GaborLayer(in_channels=1, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout2d(0.5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=320, out_features=50)  # for in_features 20*4*4 = 320
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.dropout(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
    
def apply_filters1(image, model):
    filtered_images = []
    
    with torch.no_grad():
        for i in range(10):
            # Convert the weight tensor to numpy array
            kernel = model.conv1.conv1.weight[i, 0].cpu().numpy()
            # Apply the filter using OpenCV's filter2D function
            filtered_image = cv2.filter2D(image, -1, kernel)
            filtered_images.append(filtered_image)
    return filtered_images

def main(argv):
    
    # Define whether to use new input or test dataset
    use_new_input = False
    def switch_mode():
        nonlocal use_new_input
        use_new_input = not use_new_input
        print("Switched mode. Now using Gabor filter Model." if use_new_input else "Switched mode. Now using resnet18 model.")
    
    # define the key press event handler to switch mode
    keyboard.add_hotkey('g', switch_mode)
    while True:
        if use_new_input:
            model = Mynetwork1()

            # Load only the convolutional layer weights from the pre-trained model
            pretrained_dict = torch.load("mnist_model.pth")
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'conv1' in k}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)  # Set strict=False to ignore missing keys

            # Adjust sizes of the fully connected layers
            model.fc1 = nn.Linear(128, 128)  # Adjusted size based on the pre-trained model
            model.fc2 = nn.Linear(128, 64)   # Adjusted size based on the pre-trained model
            model.fc3 = nn.Linear(64, 10)    # Adjusted size based on the pre-trained model
            
            # Load MNIST dataset
            train_data = datasets.MNIST(
                        root = 'data',
                        train = True,
                        transform= ToTensor(),
                        download= True
                    )
            images = train_data.data[:6]
            labels = train_data.targets[:6]
            
            # Plot the first six images
            plt.figure(figsize=(10, 5))
            for i in range(6):
                plt.subplot(2, 3, i + 1)
                plt.imshow(images[i], cmap='gray')
                plt.title(f"Label: {labels[i]}")
                plt.axis('off')

            plt.tight_layout()
            plt.show()

            # Print the shape of the first layer weights
            print("Shape of the first layer weights:", model.conv1.conv1.weight.shape)

            # Visualize the ten filters using pyplot
            num_filters = model.conv1.conv1.weight.shape[0]
            num_rows = 3
            num_cols = 4
            plt.figure(figsize=(15, 30))
            
            for i in range(num_filters):
                plt.subplot(num_rows, num_cols, i+1)
                plt.imshow(model.conv1.conv1.weight[i, 0].cpu().detach().numpy(), cmap='viridis')
                plt.xticks([])
                plt.yticks([])
                plt.title('Filter {}'.format(i+1))    
            plt.show()

            # Load the first training example image
            image, _ = train_data[0]  # Extracting the first image and its label
            image = image.squeeze().numpy()  # Convert PyTorch tensor to numpy array and remove the channel dimension

            # Apply filters to the first training example image
            filtered_images = apply_filters1(image, model)
            
            # Plot the filters and filtered images
            num_rows = 5
            num_cols = 4
            plt.figure(figsize=(20, 30))
            
            for i in range(10):
                row = i // 2
                col = i % 2 * 2

                print(f"Filter {i}: (Row: {row}, Col: {col})")

                # Plot filter
                plt.subplot(num_rows, num_cols * 2, row * num_cols * 2 + col * 2 + 1)
                plt.imshow(model.conv1.conv1.weight[i, 0].cpu().detach().numpy(), cmap='gray')
                plt.title(f'Filter {i+1}')   
                plt.axis('off')

                # Plot filtered image
                plt.subplot(num_rows, num_cols * 2, row * num_cols * 2 + col * 2 + 2)
                plt.imshow(filtered_images[i], cmap='gray')
                plt.title(f'Filtered Image {i+1}')
                plt.axis('off')

                # Add empty space between the columns
                plt.subplot(num_rows, num_cols * 2, row * num_cols * 2 + col * 2 + 3)
                plt.axis('off')       
            plt.tight_layout()
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.4, wspace=0.4)
            plt.show()
            
        else:
            # Load the pre-trained ResNet-18 model
            model = models.resnet18(pretrained=True)
            
            # Print model to examine its structure and layer name
            print(model)
            
            train_data = datasets.MNIST(
                        root = 'data',
                        train = True,
                        transform= transforms.ToTensor(),
                        download= True
                    )
            images = train_data.data[:6]
            labels = train_data.targets[:6]
            
            # Plot the first six images
            plt.figure(figsize=(10, 5))
            for i in range(6):
                plt.subplot(2, 3, i + 1)
                plt.imshow(images[i], cmap='gray')
                plt.title(f"Label: {labels[i]}")
                plt.axis('off')

            plt.tight_layout()
            plt.show()

            # Load the first training example image
            image, _ = train_data[0]  # Extracting the first image and its label
            image = image.squeeze().numpy()  # Convert PyTorch tensor to numpy array and remove the channel dimension

            # Print the shape of the first convolutional layer weights
            print("Shape of the first layer weights:", model.conv1.weight.shape)
            print("Shape of the second layer weights:", model.layer1[0].conv1.weight.shape)

            # Visualize the filters of the first and second convolutional layers using pyplot
            num_filters_layer1 = min(model.conv1.weight.shape[0], 10)  # Limit number of filters to visualize
            num_filters_layer2 = min(model.layer1[0].conv1.weight.shape[0], 10)  # Limit number of filters to visualize
            num_rows = max(num_filters_layer1, num_filters_layer2)  # Choose the maximum number of rows needed
            num_cols = 2  # Two columns for two layers

            plt.figure(figsize=(15, 40))
            plt.title("resNet18 model")

            # Visualize filters of the first convolutional layer
            for i in range(num_filters_layer1):
                plt.subplot(num_rows, num_cols, i+1)
                plt.imshow(model.conv1.weight[i, 0].cpu().detach().numpy(), cmap='viridis')
                plt.xticks([])
                plt.yticks([])
                plt.title('Filter {} (Layer 1)'.format(i+1))

            # Visualize filters of the second convolutional layer
            for i in range(num_filters_layer2):
                plt.subplot(num_rows, num_cols, i+1+num_filters_layer1)
                plt.imshow(model.layer1[0].conv1.weight[i, 0].cpu().detach().numpy(), cmap='viridis')
                plt.xticks([])
                plt.yticks([])
                plt.title('Filter {} (Layer 2)'.format(i+1))

            plt.tight_layout()
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.4, wspace=0.4)
            plt.show()

            # Apply filters to the example image for the first convolutional layer and visualize results
            filtered_images_layer1 = apply_filters(image, model, 0)
            plt.figure(figsize=(15, 40))
            plt.title("resNet18 model")
            
            for i in range(num_filters_layer1):
                plt.subplot(num_rows, num_cols, i+1)
                plt.imshow(filtered_images_layer1[i], cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.title('Filtered Image {} (Layer 1)'.format(i+1))

            # Apply filters to the example image for the second convolutional layer and visualize results
            filtered_images_layer2 = apply_filters(image, model, 3)
            for i in range(num_filters_layer2):
                plt.subplot(num_rows, num_cols, i+11)
                plt.imshow(filtered_images_layer2[i], cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.title('Filtered Image {} (Layer 2)'.format(i+1))

            plt.tight_layout()
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.4, wspace=0.4)
            plt.show()
            

if __name__ == "__main__":
    main(sys.argv)