from Models import MLPNetwork
from .losses import *
from .optimizer import *
import random


class Trainer:
    def __init__(self, model: MLPNetwork, learning_rate=0.01, epochs=100, optimizer="SGD"):
        self.model = model

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

        if optimizer == "SGD":
            self.optimizer = SGD(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        self.model_type = model.classification

    def fit(self, X, y, batch_size=1):
        n = len(X)
        for epoch in range(self.epochs):
            indices = list(range(n))
            random.shuffle(indices)

            epoch_loss = 0.0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]

                self.model.zero_grad()

                batch_losses = []

                for i in batch_idx:
                    inputs = X[i]
                    target = y[i]

                    outputs = self.model.forward(inputs) 

                    loss = self.loss_cal(outputs, target)
                    batch_losses.append(loss)
                    epoch_loss += loss.data

                batch_loss = sum(batch_losses) / len(batch_losses)
                batch_loss.backward()

                self.optimizer.step(self.model.parameters())

            epoch_loss /= n

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}")