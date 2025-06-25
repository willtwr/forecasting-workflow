import torch
from torch import nn


class MLPClassifier(nn.Module):
    """A multilayer-perceptron-based classifier
    
    This model uses multilayer perceptron to perform classification tasks.
    
    Args:
        feature_cats (list): A list of number of categories of each feature
        num_classes (int): Number of output classes
        num_hidden_neurons (int or list[int]): Dimension of hidden layers. Defaults to 256
        num_hidden_layers (int)
    """
    
    def __init__(
            self,
            feature_cats: list,
            num_classes: int,
            num_hidden_neurons: int | list[int] = 256,
            num_hidden_layers: int = 3,
            dropout: float = 0.5
    ) -> None:
        super().__init__()
        
        self.feature_cats = feature_cats
        input_dim = sum([n_cats if n_cats > 2 else 1 for n_cats in feature_cats])

        if not isinstance(num_hidden_neurons, list):
            assert num_hidden_layers > 0, "num_hidden_layers must be > 0."
            num_hidden_neurons = [num_hidden_neurons] * num_hidden_layers

        layers = [
            nn.Linear(input_dim, num_hidden_neurons[0]),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        ]
        if len(num_hidden_neurons) > 1:
            for i in range(len(num_hidden_neurons) - 1):
                layers.append(nn.Linear(num_hidden_neurons[i], num_hidden_neurons[i+1]))
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(num_hidden_neurons[-1], num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x_list = [nn.functional.one_hot(x[:,i].long(), num_classes=n_cats).float() if n_cats > 2 else x[:,i].unsqueeze(1) for i, n_cats in enumerate(self.feature_cats)]
        x = torch.cat(x_list, dim=1)
        output = self.network(x)
        return output
