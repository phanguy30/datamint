from .loss import Loss

class LinearLoss(Loss):
    def __call__(self, predicted, actual):
        return (predicted - actual) ** 2