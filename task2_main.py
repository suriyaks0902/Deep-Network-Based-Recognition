# Suriya Kasiyalan Siva
# CS5330 Pattern Recognition and Computer Vision
# Spring 2024
# Date : 3/27/2024
#
# PROJECT NUMBER : 5
# PROJECT NAME : Recognition using Deep Networks
#
# TASK 2 : Examine your network
# Description : This Python script loads a pre-trained convolutional neural network (CNN) model trained on the MNIST dataset. 
#               It visualizes the learned filters (kernels) of the first convolutional layer of the model and applies these filters to example images from the dataset. 
#               The script then plots both the filters and the resulting filtered images, providing insights into how the model processes visual information.
#

# import dependencies
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
from task1_main import Mynetwork

def apply_filters(image, model):
    filtered_images = []
    
    with torch.no_grad():
        for i in range(10):
            # Convert the weight tensor to numpy array
            kernel = model.conv1.weight[i, 0].cpu().numpy()
            # Normalize the kernel
            # kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
            # Apply the filter using OpenCV's filter2D function
            filtered_image = cv2.filter2D(image, -1, kernel)
            filtered_images.append(filtered_image)
    return filtered_images

def main(argv):
    # Read in the trained network
    model = Mynetwork()
    model.load_state_dict(torch.load("mnist_model.pth"))
    # Print model to examine its structure and layer name
    print(model)
    # get weights for layer one
    # weights = model.conv1.weight
    
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

    # load the dataloader for the test dataset
    # Load the first training example image
    image, _ = train_data[0]  # Extracting the first image and its label
    image = image.squeeze().numpy()  # Convert PyTorch tensor to numpy array and remove the channel dimension

    # Step 3b: Print the filter weights and their shape
    print("Shape of the first layer weights:", model.conv1.weight.shape)
    for i in range(model.conv1.weight.shape[0]):
        print("Filter {} weights:".format(i))
        print(model.conv1.weight[i, 0])  # Accessing the ith 5x5 filter

    # Step 3c: Visualize the ten filters using pyplot
    num_filters = model.conv1.weight.shape[0]
    num_rows = 3
    num_cols = 4
    plt.figure(figsize=(10, 8))
    for i in range(num_filters):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(model.conv1.weight[i, 0].cpu().detach().numpy(), cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        plt.title('Filter {}'.format(i+1))
    plt.show()
    # Load the first training example image
    filtered_images = apply_filters(image, model)
    
    # plt.figure(figsize=(10,8))
    # for i in range(num_filters):
    #     plt.subplot(num_rows, num_cols, i+1)
    #     plt.imshow(filtered_images[i], cmap='gray')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.title('Filter {}'.format(i+1))
    # plt.show()
    
    # Create subplots
    num_rows = 5
    num_cols = 4
    plt.figure(figsize=(12, 15))

    # Plot filters and filtered images
    for i in range(10):
        row = i // 2
        col = i % 2 * 2

        print(f"Filter {i}: (Row: {row}, Col: {col})")

        # Plot filter
        plt.subplot(num_rows, num_cols * 2, row * num_cols * 2 + col * 2 + 1)
        plt.imshow(model.conv1.weight[i, 0].cpu().detach().numpy(), cmap='Greys_r') #'gray', 'gray_r', 'grey', 'Greys_r'
        plt.title(f'Filter {i+1}')
        plt.axis('off')

        # Plot filtered image
        plt.subplot(num_rows, num_cols * 2, row * num_cols * 2 + col * 2 + 2)
        plt.imshow(filtered_images[i], cmap='Greys')
        plt.title(f'Filtered Image {i+1}')
        plt.axis('off')

        # Add empty space between the columns
        plt.subplot(num_rows, num_cols * 2, row * num_cols * 2 + col * 2 + 3)
        plt.axis('off')

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main(sys.argv)    