# datamint

## Overview

Datamint is a python program that creates a neural network to generate data. Datamint functions by using a neural network class and creating a multi layer perceptron using the training class to generate music data. 

### Sub-Package 1 - Models

- Value.py - Just regulars numbers but supports differentiation/gradient calculation

- Neuron.py - A neuron that receives a list of inputs (list of value objects) and produces one output (a value object).

- Layer.py  - A group of neurons working together to produce an output

- MLPnetwork.py - Several layers stacked to form a simple neural network.We plan to add other pre-trained/customized models that can be used for specific tasks like music generation or synthetic generation.

### Sub-Package 2 - Trainer

- Trainer.py - will fit the model using training data, uses helper functions/classes contained in the following sub-folders.
    - Folder 1: Losses Module
        - loss.py - parent class
        - bce_loss.py - binary cross entropy loss
        - ce_loss.py - cross entropy loss function
        - Linear_loss.py  - linear loss function
- Other loss functions
    - Folder 2: Optimizer Module
        - optimizer.py - parent class
        - SGD.py - stochastic gradient descent
        Other optimizers

### Sub-Package 3 - Music Generation

- Input_Trainer.py - takes a dataframe. The user specifies the independent and dependant variables and initializes a trainer

- Data_Generation.py - creates a synthetic dataset similar to the user's     original dataset. 

- Output_statistics.py - compares the initial dataset to the generated  dataset and provides simple statistics
