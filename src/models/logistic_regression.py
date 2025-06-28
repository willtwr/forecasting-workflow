import torch
from torch import nn


class LogisticRegressionClassifier(nn.Module):
    """A standard neural-network-based classifier for sequence data.

    This model uses a standard multi-layer neural network architecture followed by a classification layer
    to perform classification tasks.

    Args:
        feature_cats (list): A list of number of categories of each feature
        num_classes (int): Number of output classes
    """

    def __init__(
        self,
        feature_cats: list,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.feature_cats = feature_cats
        input_dim = sum([n_cats if n_cats > 2 else 1 for n_cats in feature_cats])
        self.network = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x_list = [
            nn.functional.one_hot(x[:, i].long(), num_classes=n_cats).float()
            if n_cats > 2
            else x[:, i].unsqueeze(1)
            for i, n_cats in enumerate(self.feature_cats)
        ]
        x = torch.cat(x_list, dim=1)
        output = self.network(x)
        return output
