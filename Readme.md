# Greek Letter Recognition System with PyTorch



## Overview
This project implements a Greek letter recognition system using convolutional neural networks (CNNs) in PyTorch. It initially recognizes alpha, beta, and gamma and is later extended to include additional Greek letters like delta. Various extensions such as evaluating multiple dimensions in model training, exploring pre-trained networks, and building a live video digit recognition application are included.

## Introduction
The goal of this project is to develop a robust Greek letter recognition system capable of accurately identifying a wide range of Greek characters. Leveraging deep learning techniques, we aim to achieve high accuracy even in challenging real-world scenarios.

## Project Structure
- **Task 1:** Initial implementation for recognizing alpha, beta, and gamma.
- **Task 2:** Exploring pre-trained networks to understand feature extraction mechanisms.
- **Task 3:** Customizing the MNIST network with Gabor filters and evaluating its performance.
- **Task 4:** Evaluating multiple dimensions in model training such as dropout rates, filter layers, batch sizes, and epochs.
- **Extensions:** Additional enhancements and explorations including expanding Greek letter recognition, visualization of network layers, and building a live video recognition application.

## Extensions
- **Greek Letter Expansion:** Enhanced recognition system to include additional Greek letters.
- **Exploration of Pre-trained Networks:** Analysis of pre-trained ResNet-18 model.
- **Custom Filter Implementation:** Replacing the first layer of the MNIST network with Gabor filters.
- **Live Video Digit Recognition:** Development of a live video digit recognition application.

## Install Dependencies
To run this project, install the following dependencies:
- Python 3.x
- PyTorch
- Matplotlib
- OpenCV
- Other libraries as required

## Usage

### Task 1
1. Run `task1_main.py` to execute Build and train a network to recognize digits[Get the MNIST digit data set, Build a network model, Train the model, Save the network to a file].
2. Run `task1_main2.py` and press 'n' to switch between using new input images and the test dataset to execute [Read the network and run it on the test set, Test the network on new inputs].

### Task 2
Run `task2_main.py` to explore pre-trained networks and evaluate their convolutional layers.

### Task 3
Run `task3_along_extension.py` to implement Greek letter recognition with the provided dataset and handwritten letters.

### Task 4
Run `task4_main.py` to evaluate multiple dimensions in model training.

### Extensions
#### Evaluate more dimensions on task 4
Run `Extension_1.py` to assess additional dimensions in model training beyond those in Task 4.

#### Try more Greek letters than alpha, beta, and gamma
Implemented within `task3_along_extension.py`.

#### Pre-trained networks evaluation and Gabor filters
Run `extension_task2.py` and press 'g' to switch between using Gabor filters in the first layer and pre-trained networks evaluation.

#### Live video digit recognition application
Run `extension_task_video.py` to build and test a live video digit recognition application using the trained network.

## Drive Link:
The drive contains datasets like Greek letters, handwritten numbers from 0-9 

link: https://drive.google.com/drive/folders/1YHpR2yw74ZA5m4vqnLdBafhIYs6xUxko?usp=sharing

## Learning Reflection
Through this project, I've gained valuable experience in:
- Designing and training CNNs for pattern recognition tasks.
- Exploring pre-trained networks and understanding their inner workings.
- Implementing custom layers and filters for specific applications.
- Developing real-time applications for computer vision tasks.
- Fine-tuning hyperparameters and evaluating model performance.

## Acknowledgement

I would like to express my gratitude to the following resources for their invaluable support and guidance throughout this project:

- [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html)
- [MNIST Digit Tutorial](https://nextjournal.com/gkoehler/pytorch-mnist)
- [Matplotlib](https://matplotlib.org/stable/tutorials/pyplot.html)
- [nn module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

Special thanks to the developers and contributors of these resources for providing excellent documentation, tutorials, and tools that facilitated the implementation and understanding of various concepts in deep learning and computer vision.


## Contact for More Information

For further inquiries or collaborations, feel free to contact 

Email : k.s.suriya0902@gmail.com
