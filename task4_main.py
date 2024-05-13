# Suriya Kasiyalan Siva
# CS5330 Pattern Recognition and Computer Vision
# Spring 2024
# Date : 4/1/2024
#
# PROJECT NUMBER : 5
# PROJECT NAME : Recognition using Deep Networks
#
# TASK 4 : Design your own experiment
# Description : This Python script defines a Convolutional Neural Network (CNN) model called ConvNet and a function to train the model using variations of hyperparameters such as dropout rate, number of convolutional filters, and batch size. 
#               The main function runs an experiment by training the CNN model with different combinations of hyperparameters and records the loss, accuracy, and training time for each configuration. 
#               Finally, it prints the results of the experiment, showing the performance metrics for each configuration.

# import statement
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time

# Define the Convolutional Neural Network model
class ConvNet(nn.Module):
    def __init__(self, dropout_rate, conv_filter_num):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, conv_filter_num, kernel_size=3)
        self.conv2 = nn.Conv2d(conv_filter_num, 2 * conv_filter_num, kernel_size=3)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.conv_filter_num = conv_filter_num
        
        # Calculate output size after convolution layers
        conv_output_size = self._get_conv_output_size()
        self.fc1 = nn.Linear(2 * conv_filter_num * conv_output_size * conv_output_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def _get_conv_output_size(self):
        # Function to calculate output size after convolution layers
        image_size = 28
        conv_output_size = ((image_size - 3 + 1) // 2 - 3 + 1) // 2
        return conv_output_size

    # Forward pass through the network
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_model(dropout_rate, conv_filter_num, batch_size):
    model = ConvNet(dropout_rate, conv_filter_num)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    epoch_loss = 0
    epoch_accuracy = 0

    start_time = time.time()
    for epoch in range(5):
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(trainloader)
            epoch_accuracy = 100 * correct / total

    training_time = time.time() - start_time
    return epoch_loss, epoch_accuracy, training_time

# Main function to run the experiment
def main(argv):
    dropout_rates = [0.1, 0.3, 0.5, 0.8]
    conv_filter_nums = [16, 32, 64, 128]
    batch_sizes = [32, 64, 128, 256]

    num_variations = 50  # number of variations
    variations = [] 
    for dropout_rate in dropout_rates:
        for conv_filter_num in conv_filter_nums:
            for batch_size in batch_sizes:
                variations.append((dropout_rate, conv_filter_num, batch_size))

    selected_variations = variations[:num_variations]
    # Result is stored inside results
    results = {}
    for idx, (dropout_rate, conv_filter_num, batch_size) in enumerate(selected_variations, 1):
        print(f"Evaluating Variation {idx}/{num_variations} - Dropout Rate={dropout_rate}, Conv Filter Num={conv_filter_num}, Batch Size={batch_size}")
        loss, accuracy, training_time = train_model(dropout_rate, conv_filter_num, batch_size)
        results[(dropout_rate, conv_filter_num, batch_size)] = (loss, accuracy, training_time)

    for config, (loss, accuracy, training_time) in results.items():
        print(f"Configuration: Dropout Rate={config[0]}, Conv Filter Num={config[1]}, Batch Size={config[2]}, Loss={loss:.4f}, Accuracy={accuracy:.2f}%, Training Time={training_time:.2f} seconds")

if __name__ == "__main__":
    main(sys.argv)
