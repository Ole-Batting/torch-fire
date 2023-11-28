from typing import Optional

import torch
import torch.nn as nn


class Simple(nn.Sequential):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            n_hidden: list[int],
            a_hidden: nn.Module,
            a_out: Optional[nn.Module],
            dropout: float = 0,
    ):
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.a_hidden = a_hidden
        self.a_out = a_out
        self.dropout = dropout

        self.layers = nn.ModuleList([
            nn.Linear(self.n_in, self.n_hidden[0]),  # Fully connected layer
            self.a_hidden(),                         # Nonlinearity (activation function)
            nn.Dropout(self.dropout),                # Regularization w dropout
            nn.BatchNorm1d(self.n_hidden[0]),        # Regularization w batch normalization
        ])

        for n0, n1 in zip(self.a_hidden[:-1], self.a_hidden[1:]):
            self.layers.extend([
                nn.Linear(n0, n1),
                self.a_hidden(),
                nn.Dropout(self.dropout),
                nn.BatchNorm1d(n1),
            ])

        super().__init__(self.layers)
