from .MLPnetwork import MLPNetwork
class MLPGenerator(MLPNetwork):
    def __init__(self, latent_dim, cond_dim, output_dim, hidden_sizes, activation_type="tanh"):
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
        input_vector = self._build_input(latent_vector, cond_vector)
        return self.predict(input_vector)