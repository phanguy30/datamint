
import random
def build_generator_training_data(X_real, y_labels, latent_dim):
    """
    Take in predictors and labels for real data, and build training data 
    return x_train as (z,y) and y_train as real x.
    Parameters:
    X_real: List of real data samples (predictors)
    y_labels: List of corresponding labels for the real data
    latent_dim: Dimension of the latent noise vector
    
    Returns:
    Tuple (X_train, y_train) for training the generator
    """
    X_train = []   # inputs to generator
    y_train = []   # targets (real x)

    for x, y in zip(X_real, y_labels):
        
        # sample random noise z
        z = [random.uniform(-1, 1) for _ in range(latent_dim)]
        
        # concat [z, y_onehot] as input
        inp = z + y_labels

        X_train.append(inp)
        y_train.append(x)

    return X_train, y_train
