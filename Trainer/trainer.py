from Models import MLPNetwork
from .losses import *
from .optimizer import *
import random


class Trainer:
    def __init__(self, model: MLPNetwork, learning_rate=0.01, epochs=100, optimizer="SGD"):
        """
        Parameters:
        model: instance of any model (default is MLPNetwork)
        learning_rate: learning rate for optimizer
        epochs: number of training epochs
        optimizer: optimization algorithm to use ("SGD" supported)
        """
        
        #Default model is MLPNetwork
        self.model = model
        
        #Set loss calculation based on model classification type
        if model.classification == "none":
            self.loss_cal = LinearLoss()
        elif model.classification == "sigmoid":
            self.loss_cal = BCELoss()
        elif model.classification == "softmax":
            self.loss_cal = CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown classification mode: {model.classification}")

        self.learning_rate = learning_rate
        self.epochs = epochs

        # Set optimizer
        if optimizer == "SGD":
            self.optimizer = SGD(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        
    def fit(self, X, y, batch_size=1):
        """
        Parameters:
        X: list of input samples, each sample is a list of Values or floats/ints
        y: list of target values (floats/ints)
        batch_size: number of samples per batch for training
        
        Trains the model using mini-batch gradient descent.
        """
        n = len(X)
        
        # Training loop
        for epoch in range(self.epochs):
            indices = list(range(n))
            random.shuffle(indices)

            epoch_loss = 0.0
            
            # Mini-batch training
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                
                # Zero gradients before processing the batch
                self.model.zero_grad()

                batch_losses = []
                
                #Calculate loss for each sample in the batch
                for i in batch_idx:
                    inputs = X[i]
                    target = y[i]

                    outputs = self.model.predict(inputs) 

                    loss = self.loss_cal(outputs, target)
                    batch_losses.append(loss)
                    epoch_loss += loss.data

                # Compute average loss for the batch and backpropagate to get gradients
                batch_loss = sum(batch_losses) / len(batch_losses)
                batch_loss.backward()
                
                # Update model parameters
                self.optimizer.step(self.model.parameters())
                
            # Monitor average loss for the epoch
            epoch_loss /= n

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}")
    
    
        
        
    def test(self, X, y):
        """
        Parameters:
        X: list of input samples, each sample is a list of Values or floats/ints
        y: list of target values (0 or 1)
        
        Returns:
        Test statistics (e.g., accuracy, loss)
        """
        n = len(X)
        
        correct = 0
        total_loss = 0.0
        
        for i in range(n):
            inputs = X[i]
            target = y[i]

            outputs = self.model.predict(inputs)
            
            if self.model.classification == "sigmoid":
                predicted_prob = outputs[0].data
                predicted_label = 1 if predicted_prob >= 0.5 else 0

                if predicted_label == target:
                    correct += 1
                    
            elif self.model.classification == "softmax":
                predicted_label = outputs.index(max(outputs, key=lambda v: v.data))

                if predicted_label == target:
                    correct += 1

            loss = self.loss_cal(outputs, target)
            total_loss += loss.data

        accuracy = correct / n
        avg_loss = total_loss / n
        
        if self.model.classification != "none":
            print(f"Test Accuracy: {accuracy*100:.2f}%, Test Loss: {avg_loss:.4f}")
            return accuracy, avg_loss
        else:
            print(f"Test Loss: {avg_loss:.4f}")
            return avg_loss
        
        
    
    