# Suriya Kasiyalan Siva
# CS5330 Pattern Recognition and Computer Vision
# Spring 2024
# Date : 3/27/2024
#
# PROJECT NUMBER : 5
# PROJECT NAME : Recognition using Deep Networks
#
# TASK 1 : Build and train a network to recognize digits
# Description : This script enables interactive testing of a pre-trained convolutional neural network (CNN) model for handwritten digit recognition. 
#               Users can choose between two modes: "New Input Mode" and "Test Dataset Mode". In the "New Input Mode", users can input new images of handwritten digits, which are preprocessed and fed into the model for prediction. 
#               Predictions are then displayed alongside the input images. Alternatively, in the "Test Dataset Mode", the script evaluates the model on the MNIST test dataset, displaying predictions for the first 10 images along with their corresponding true labels. 
#               Users can switch between these modes by pressing the 'n' key.

# import dependencies
import sys
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from task1_main import Mynetwork
from PIL import Image
import keyboard
from PIL import ImageOps


# Define a function to preprocess the image
def preprocess_image(image_path): 
    image = Image.open(image_path).convert("L") # Open the image and Convert to grayscale
    image = ImageOps.invert(image) # Convert to grayscale
    image = image.resize((28, 28)) # Resize to 28x28 pixels
    transform = ToTensor() # Convert to tensor
    # image = (image - 0.1307) / 0.3081  # Normalize
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Define a function to test the network on new inputs
def new_input(model, image_paths):
    predictions = []
    for image_path in image_paths:
        image = preprocess_image(image_path)
        with torch.no_grad():
            output = model(image)
            prediction = output.argmax(dim=1)
            predictions.append(prediction.item())
    return predictions

# Define a function to test the network on the test dataset
def test_dataset(model, test_loader):
    # list to the store prediction and labels 
    predictions = []
    true_labels = []

    # iterate over the first 10 images in the test dataset
    for i, (image, labels) in enumerate(test_loader):
        if i >= 10:
            break
        # Now pass the images through the model for prediction
        with torch.no_grad():
            model_Output = model(image)
            prediction = model_Output.argmax(dim =1)
            predictions.append(prediction.item())
            true_labels.append(labels.item())
            
            # Print output values of network
            print(f"Example {i+1} - Network Output Values: {['%.2f' % elem for elem in model_Output[0]]}")
            print(f"Example {i+1} - Predicted Index: {prediction}, Correct Label: {labels.item()}\n")
    return predictions, true_labels
   
   
def main(argv):
    # Load the trained model
    model = Mynetwork()
    model.load_state_dict(torch.load("mnist_model.pth"))
    # set the model to evaluation mode
    model.eval()
        
    # Define whether to use new input or test dataset
    use_new_input = False
    def switch_mode():
        nonlocal use_new_input
        use_new_input = not use_new_input
        print("Switched mode. Now using new imput images." if use_new_input else "Switched mode. Now using test dataset.")
    
    # define the key press event handler to switch mode
    keyboard.add_hotkey('n', switch_mode)
    
    while True:
        if use_new_input:
            # Paths to handwritten digit images
            image_paths = ["IMG_0.jpg", 
                           "IMG_1.jpg", 
                           "IMG_2.jpg", 
                           "IMG_3.jpg", 
                           "IMG_41.jpg",
                           "IMG_5.jpg", 
                           "IMG_6.jpg",
                           "IMG_71.jpg", 
                           "IMG_8.jpg", 
                           "IMG_9.jpg"]
            
            # Test the network on new input
            predictions = new_input(model, image_paths)
            
            # Plot the handwritten digits along with their predictions
            plt.figure(figsize=(12, 6))
            for i in range(10):
                plt.subplot(2, 5, i + 1)
                image = Image.open(image_paths[i])
                plt.imshow(image, cmap='gray')
                plt.title(f"Prediction: {predictions[i]}")
                plt.axis('off')

            plt.tight_layout()
            plt.show()     
        
        else:
            # Load the test dataset
            test_data = datasets.MNIST(
                root = 'data',
                train = False,
                transform= ToTensor(),
                download= True
            )
            # load the dataloader for the test dataset

            test_loader = DataLoader(
                test_data,
                batch_size= 1,
                shuffle= False
            )
            # Test the network on the test dataset
            predictions, true_labels = test_dataset(model, test_loader)

            # plot the first 9 digits of the test dataset along with the prediction
            plt.figure(figsize=(10,10)) 
            for i in range(9):
                plt.subplot( 3, 3, i+1)
                plt.imshow(test_data.data[i], cmap = 'gray')
                plt.title(f"Prediction: {predictions[i]}, Label: {true_labels[i]}")
                plt.axis('off')
            plt.tight_layout()
            plt.show()
            
if __name__ == "__main__":
    main(sys.argv)
