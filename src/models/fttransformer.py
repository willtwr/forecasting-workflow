from collections import OrderedDict

import torch
from torch import nn


class FTTransformerClassifier(nn.Module):
    """A transformer-based classifier.

    This model uses a transformer encoder architecture followed by a classification layer
    to perform classification tasks.

    Args:
        feature_cats (list): A list of number of categories of each feature
        num_classes (int): Number of output classes
        d_model (int, optional): Dimension of transformer model. Defaults to 512.
        nhead (int, optional): Number of attention heads. Defaults to 8.
        num_encoder_layers (int, optional): Number of transformer encoder layers. Defaults to 3.
        dim_feedforward (int, optional): Dimension of feedforward network. Defaults to 2048.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """

    def __init__(
        self,
        feature_cats: list,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 192,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.feature_cats = feature_cats

        # Embedding layer for categorical data and continuous data
        self.cat_embeddings_list = []
        self.cont_embeddings_list = []
        for n_cats in feature_cats:
            if n_cats > 1:
                self.cat_embeddings_list.append(nn.Embedding(n_cats, d_model))
            else:
                self.cont_embeddings_list.append(nn.Linear(1, d_model))

        self.cat_embeddings = nn.ModuleList(self.cat_embeddings_list)
        self.cont_embeddings = nn.ModuleList(self.cont_embeddings_list)

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Output classifier
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("norm0", nn.LayerNorm(d_model)),
                    ("relu0", nn.ReLU()),
                    ("linear0", nn.Linear(d_model, num_classes)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x_cat_list = []
        n_cat = 0
        x_cont_list = []
        n_cont = 0
        for i, n_cats in enumerate(self.feature_cats):
            if n_cats > 1:
                x_cat_list.append(self.cat_embeddings[n_cat](x[:, i].long()))
                n_cat += 1
            else:
                x_cont_list.append(self.cont_embeddings[n_cont](x[:, i].unsqueeze(1)))
                n_cont += 1

        # Stack cls
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        if len(x_cat_list) == 0:
            x_cont = torch.stack(x_cont_list, dim=1)
            x = torch.cat([cls_tokens, x_cont], dim=1)
        elif len(x_cont_list) == 0:
            x_cat = torch.stack(x_cat_list, dim=1)
            x = torch.cat([cls_tokens, x_cat], dim=1)
        else:
            x_cat = torch.stack(x_cat_list, dim=1)
            x_cont = torch.stack(x_cont_list, dim=1)
            x = torch.cat([cls_tokens, x_cat, x_cont], dim=1)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Classification layer
        output = self.classifier(x[:, 0])
        return output
