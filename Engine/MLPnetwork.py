from Layer import Layer

class MLPNetwork:
    def __init__(self,nin,nouts,label="",activation_type="tanh",classification=False):
        self.classification = classification
        sizes = [nin] + nouts

        self.layers = [
            Layer(
                sizes[i],
                sizes[i+1],
                label=f"{label}_L{i}",
                activation_type=activation_type if i < len(nouts) - 1 else None
            )
            for i in range(len(nouts))
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        if self.classification:
            x = self.softmax(x)

        return x

    def parameters(self):
        """aggregate all parameters from all layers"""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    @staticmethod
    def softmax(vals):
        exps = [v.exp() for v in vals]
        total = exps[0]
        for e in exps[1:]:
            total = total + e
        return [e / total for e in exps]
    
    def zero_grad(self):
        """Set all gradients to zero use to reset after learning step"""
        for p in self.parameters():
            p.grad = 0