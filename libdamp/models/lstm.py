"""Bidirectional LSTM module.

Copied from https://github.com/jongwook/onsets-and-frames/blob/master/onsets_and_frames/lstm.py
"""

import torch
from torch import nn


class BiLSTM(nn.Module):
    inference_chunk_length: int = 512

    def __init__(self, input_features: int, recurrent_features: int, bidirectional: bool = True):
        """Initialize a bidirectional LSTM module.

        Parameters
        ----------
        input_features : int
            Number of input features at each timestep
        recurrent_features : int
            Number of hidden units in the LSTM
        bidirectional : bool
            Whether to use bidirectional LSTM (default: True)
        """
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=bidirectional)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input sequence through the LSTM.

        During training, processes the entire sequence at once. During evaluation,
        processes the sequence in chunks of `inference_chunk_length` to support
        longer sequences while maintaining memory efficiency.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, sequence_length, input_features)

        Returns
        -------
        torch.Tensor
            Output of shape (batch, sequence_length, num_directions * recurrent_features)
            where num_directions is 2 for bidirectional LSTM, 1 for unidirectional
        """
        if self.training:
            return self.rnn(x)[0]
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

            # forward direction
            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.rnn.bidirectional:
                h.zero_()
                c.zero_()

                for start in reversed(slices):
                    end = start + self.inference_chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

            return output
