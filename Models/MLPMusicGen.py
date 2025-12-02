from .MLPnetwork import MLPNetwork
import numpy as np
class MLPMusicGen(MLPNetwork):
    def __init__(self, context_length =10, hidden_sizes = [64,64], activation_type="tanh"):
        super().__init__(
            input_dim= context_length*128,
            n_neurons= hidden_sizes + [128],
            activation_type = activation_type,
            classification = "softmax"
        )
        
        self.context_length = context_length
        
    def _make_onehot(self, indicies, total=128):
        """
        Convert indicies into one-hot vectors by
        first creating an identity matrix of shape [total, total],
        then indexing the appropriate columns of that identity matrix.

        Parameters:
            `indices` - a numpy array of some shape where
                        the value in these arrays should correspond to category
                        indices (e.g. note values between 0-127)
            `total` - the total number of categories (e.g. total number of notes)

        Returns: a numpy array of one-hot vectors
            If the `indices` array is shaped (N,)
            then the returned array will be shaped (N, total)
            If the `indices` array is shaped (N, D)
            then the returned array will be shaped (N, D, total)
            ... and so on.
        """
        I = np.eye(total)
        return I[indicies]

    
    
    def generate_piece(self,seed, max_len=100):
        """
        Generate a piece of music given the model and an initial
        "seed" sequence of notes at the beginning of the piece.

        The piece is generated one note at a time by using, as input
        to the model, the previous 20 notes. The model outputs a
        probability distribution over the next possible note, and we
        will take the most probable note as the next note in our piece.

        Parameters:
            `model` - an instance of MLPModel
            `seed` - a sequence of notes at the beginning of a piece,
                    e.g. generated from calling `get_midi_file_notes`
                    must be at least as long as CONTEXT_LENGTH
            `max_len` - maximum number of total notes in the piece.

            Returns: a list of sequence of notes with length at most `max_len`
            """
        assert(len(seed) >= self.context_length)

        generated = seed #tracking the number of notes
        while len(generated) < max_len:
            # Use the model to predict the next note given the previous CONTEXT_LENGTH notes
            last_n_notes = generated[-self.context_length:]
            x = self._make_onehot(last_n_notes).reshape((1, -1))
            x = x.flatten().tolist()

            y = self.predict(x) # return a list of probabilities for the best next notes

            probabilities = [val.data for val in y]
            next_note = probabilities.index(max(probabilities))

            if next_note == 0:  # Look for the marker for the end of the song
                break
            generated.append(next_note)


        return generated
    
    
    
