from .MLPnetwork import MLPNetwork
class MLPGenerator(MLPNetwork):
    def __init__(self, latent_dim, cond_dim, output_dim, hidden_sizes, activation_type="tanh"):
        """
        Creating a MLP-based Generator model.
        
        
        :param latent_dim: seed noise dimension
        :param cond_dim: dimension of the dependent variable
        :param output_dim: dimension of the predictors
        :param hidden_sizes: size of the hidden layers as list
        :param activation_type: activation function type
        """
        super().__init__(
            input_dim= latent_dim + cond_dim,
            n_neurons= hidden_sizes + [output_dim],
            activation_type = activation_type,
            classification = "none"
        )
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        

    def _build_input(self, latent_vector, cond_vector):
        """helper to concatenate latent and condition vectors"""
        assert len(latent_vector) == self.latent_dim, f"Expected latent vector of length {self.latent_dim}, got {len(latent_vector)}"
        assert len(cond_vector) == self.cond_dim, f"Expected condition vector of length {self.cond_dim}, got {len(cond_vector)}"
        return latent_vector + cond_vector
    
    def generate(self, latent_vector, cond_vector):
        """
        Parameters:
        latent_vector: list of floats/ints/Values with length equal to latent_dim
        cond_vector: list of floats/ints/Values with length equal to cond_dim
        
        Returns:
        generated output as list of Values
        """
        if len(latent_vector) != self.latent_dim:
            raise ValueError(f"Latent vector/seed length {len(latent_vector)} does not match expected {self.latent_dim}.")
        if len(cond_vector) != self.cond_dim:
            raise ValueError(f"Condition vector length {len(cond_vector)} does not match expected {self.cond_dim}.")
        
        input_vector = self._build_input(latent_vector, cond_vector)
        return self.predict(input_vector)