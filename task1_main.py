# Suriya Kasiyalan Siva
# CS5330 Pattern Recognition and Computer Vision
# Spring 2024
# Date : 3/27/2024
#
# PROJECT NUMBER : 5
# PROJECT NAME : Recognition using Deep Networks
#
# TASK 1 : Build and train a network to recognize digits
# Description : This code defines a convolutional neural network (CNN) model for image classification using PyTorch. 
#               It trains the model on the MNIST dataset, plots some images from the dataset, trains the model, evaluates its performance, 
#               and finally saves the trained model.

  
# import statement
import sys
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Class for CNN model for image classification
class Mynetwork(nn.Module):
    def __init__(self):
        super(Mynetwork, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout2d(0.5)
        self.pool2 = nn.MaxPool2d(2,2)
        # Define fully connected layers
        self.fc1 = nn.Linear(in_features= 320, out_features= 50)      #  for in_features 20*4*4 = 320
        self.fc2 = nn.Linear(in_features= 50, out_features= 10)
        
    def forward(self,x):
        # Define forward pass of the network
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.dropout(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

# training loop
def train_network(model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []  # List to store the loss values
    accuracies = []  # List to store the accuracy values
    train_c = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            accuracy = 100. * batch_idx / len(train_loader)
            accuracies.append(accuracy)
            losses.append(loss.item())
            train_c.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            
    return losses, train_c


def test_network(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy, test_loss

def main(argv):
    # here I am downloading the Train and Test MNIST datas from datasets module and then convert the images into Tensor (3D array)
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

    # here I am creating a dataloader to Load the train and test data into the model later,
    # By using class DataLoader from torch.utils.data
    data_loaders = {
        'train': DataLoader(train_data,
                            batch_size=64,
                            shuffle=True,
                            num_workers=1),
        
        'test': DataLoader(test_data,
                            batch_size=64,
                            shuffle=False,
                            num_workers=1),
    }
    
    
    # Extract the first six images and labels from the test dataset
    images = test_data.data[:6]
    labels = test_data.targets[:6]
    
    # Plot the first six images
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Get cpu device for training.
    device = torch.device('cpu')
    print(f"Using {device} device")
    model = Mynetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train and test the network
    train_c = []
    test_c = []
    train_losses = []
    test_losses = []
    train_l =[]
    train_accuracy_list = []
    test_accuracy_list = []
    # Training loop
    for epoch in range(1, 11):
        train_loss, temp_train_c = train_network(model, device, data_loaders['train'], optimizer, epoch)
        train_accuracy, l = test_network(model, device, data_loaders['train'])
        test_accuracy, test_loss = test_network(model, device, data_loaders['test'])
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        train_l.append(l)
        train_losses.extend(train_loss)
        test_losses.append(test_loss)
        train_c.extend(temp_train_c)
        test_c.append(epoch * len(data_loaders['train'].dataset))
        
    # Plotting training and testing loss curves
    plt.figure()
    plt.plot(train_c, train_losses, color='blue')
    plt.scatter(test_c, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.grid(True)
    plt.show()
    
    # Plotting training and testing accuracies
    plt.figure()
    plt.plot(range(1, 11), train_accuracy_list, marker='o', label='Training Accuracy', color='blue')
    plt.plot(range(1, 11), test_accuracy_list, marker='o', label='Testing Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracies')
    plt.xticks(range(1, 11))
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the trained model
    torch.save(model.state_dict(), "mnist_model.pth")
    
if __name__ == "__main__":
    main(sys.argv)   
