import torch
from torch import nn
from torch.nn import Parameter

from typing import *
from enum import IntEnum


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class CustomGRU(nn.Module):
    """
    Unfortunately the PyTorch GRU/LSTM etc appear to be written in C, possibly to take advantage of CuDNN kernels;
    certainly for speed. Therefore, we must begin with as good a scripted version as possible. A reasonable starting
    point is found in a blog post by keitakurita:
    http://mlexplained.com/2019/02/15/building-an-lstm-from-scratch-in-pytorch-lstms-in-depth-part-1/
    This also contains similar optimisations to Flux.jl. This is substantially adapted from this post in order to
    provide a useful starting point for a GRU cell. Due to the architecture of our target model, this cannot be used
    as a building block in the way that nn.Modules are intended, but it serves as a base for the source code.
    """
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.weight_ih = Parameter(torch.Tensor(input_sz, hidden_sz * 3))
        self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 3))
        self.bias = Parameter(torch.Tensor(hidden_sz * 3))
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

        # open forget gate
        self.bias.data[self.hidden_size:2 * self.hidden_size] = torch.ones_like(
            self.bias.data[self.hidden_size:2*self.hidden_size])

    hidden_seq = []

    def forward(self, x: torch.Tensor,
            init_states: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assumes x is of shape (batch, sequence, feature)"""
        if x.ndimension() == 2:
            return self.forward_2(x, init_states)
        elif x.ndimension() == 3:
            return self.forward_3(x, init_states)
        else:
            Exception("Dimension {:d} of Tensor unsupported".format(x.ndimension()))

    def forward_3(self, x: torch.Tensor,
            init_states: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        h_t = torch.zeros(self.hidden_size).to(x.device) if init_states is None else init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]

            # batch the matmuls
            gx = x_t @ self.weight_ih
            gh = h_t @ self.weight_hh
            _ixz, _ixr, _ixe = slice(0, HS), slice(HS, 2*HS), slice(2*HS, 3*HS)

            z_t = torch.sigmoid(gx[:, _ixz] + gh[:, _ixz] + self.bias[_ixz])  # convex pass-through
            r_t = torch.sigmoid(gx[:, _ixr] + gh[:, _ixr] + self.bias[_ixr])  # forget
            eta_t = torch.tanh( gx[:, _ixe] + r_t * torch.tanh(gh[:, _ixe]) + self.bias[_ixe]) # hidden
            h_t = z_t * h_t + (1 - z_t) * eta_t

            hidden_seq.append(h_t.unsqueeze(Dim.batch))

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, h_t

    def forward_2(self, x: torch.Tensor,
            init_states: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assumes x is of shape (batch, feature)"""

        bs = x.size(0)
        hidden_seq = []
        h_t = torch.zeros(self.hidden_size).to(x.device) if init_states is None else init_states

        HS = self.hidden_size
        x_t = x

        # batch the matmuls
        gx = x_t @ self.weight_ih
        gh = h_t @ self.weight_hh
        _ixz, _ixr, _ixe = slice(0, HS), slice(HS, 2*HS), slice(2*HS, 3*HS)

        z_t = torch.sigmoid(gx[:, _ixz] + gh[:, _ixz] + self.bias[_ixz])  # convex pass-through
        r_t = torch.sigmoid(gx[:, _ixr] + gh[:, _ixr] + self.bias[_ixr])  # forget
        eta_t = torch.tanh( gx[:, _ixe] + r_t * torch.tanh(gh[:, _ixe]) + self.bias[_ixe]) # hidden
        h_t = z_t * h_t + (1 - z_t) * eta_t

        return h_t