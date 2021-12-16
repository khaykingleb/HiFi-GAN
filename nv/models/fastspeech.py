from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch

import numpy as np

from typing import *


class DurationPredictor(nn.Module):
    
    def __init__(self, hidden_size: int, kernel_size: int, dropout: float):
        super().__init__()

        self.duration_predictor = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding="same"
            ),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding="same"
            ),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: tensor with shape of (batch_size, seq_len, embed_size)

        Returns: 
            out: tensor with shape of (batch_size, seq_len)
        """
        for idx, operation in enumerate(self.duration_predictor):
            if idx == 0 or idx == 4:
                x = operation(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = operation(x)

        durations = x.squeeze(-1)

        return durations 


class Attention(nn.Module):

    def __init__(self, num_heads: int, hidden_size: int, dropout: float):
        super().__init__()

        attention_size = hidden_size // num_heads

        self.W_Q = nn.Linear(hidden_size, attention_size)
        self.W_K = nn.Linear(hidden_size, attention_size)
        self.W_V = nn.Linear(hidden_size, attention_size)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            query: tensor with shape of (batch_size, seq_len, hidden_size)
            key: tensor with shape of (batch_size, seq_len, hidden_size)
            value: tensor with shape of (batch_size, seq_len, hidden_size)
            mask: tensor with shape of (batch_size, seq_len) ???
        
        Returns: 
            out: tensor with shape of (batch_size, seq_len, hidden_size // 2)
        """
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)

        attention = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.shape[-1])
        
        scores = nn.functional.softmax(attention, dim=-1)
        out = torch.bmm(scores, V)
        out = self.dropout(out)

        return out
        

class MultiHeadAttention(nn.Module):
    """ Based on https://arxiv.org/pdf/1706.03762.pdf """

    def __init__(self, num_heads: int, hidden_size: int, dropout: float):
        super().__init__()

        self.multi_head = nn.Sequential(
            *[Attention(num_heads, hidden_size, dropout) for _ in range(num_heads)]
        )
        
        self.linear = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: tensor with shape of (batch_size, seq_len, hidden_size) 

        Returns:
            out: tensor with shape of (batch_size, seq_len, hidden_size) 
        """
        heads_out = torch.cat([attention(x) for attention in self.multi_head], dim=-1)
        out = self.linear(heads_out)

        return out


class FFTBlock(nn.Module):

    def __init__(
        self, 
        num_heads: int,
        hidden_size: int,  
        kernel_size: int,
        dropout: float
    ):
        super().__init__()
        
        self.multi_head_attention = MultiHeadAttention(
            num_heads, hidden_size, dropout
        )
        
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding="same"
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding="same"
            )
        )

        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: tensor of shape (batch_size, seq_len, emded_size)

        Returns:
            
        """
        out = self.multi_head_attention(self.layer_norm_1(x)) + x
        out = self.conv(self.layer_norm_2(out).permute(0, 2, 1)).permute(0, 2, 1) + out

        return out
        

class PositionalEncoding(nn.Module):
    """ Got from https://pytorch.org/tutorials/beginner/transformer_tutorial.html """

    def __init__(self, embed_size: int, max_len: int, dropout: float):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-np.log(10000.0) / embed_size)
        )
        positional_encoding = torch.zeros(max_len, 1, embed_size)
        positional_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', positional_encoding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: tensor of shape (batch_size, seq_len, emded_size)

        Returns:
            out: tensor of shape (batch_size, seq_len, emded_size)
        """
        out = x + self.positional_encoding[:x.size(0)]
        out = self.dropout(out)

        return out


class FastSpeech(nn.Module):

    def __init__(self, config):
        super().__init__()
        args_config = config["arch"]

        self.phoneme_embedding = nn.Embedding(
            args_config["vocab_size"], 
            args_config["phoneme_embed_size"]
        )

        self.positional_encoding = PositionalEncoding(
            args_config["phoneme_embed_size"], 
            args_config["phoneme_max_len"], 
            args_config["dropout"]
        )

        self.encoder = nn.Sequential(
            *[FFTBlock(
                args_config["num_heads"], args_config["hidden_size"], 
                args_config["kernel_size"], args_config["dropout"]
              ) for _ in range(args_config["phoneme_fft_blocks"])]
        )

        self.duration_predictor = DurationPredictor(
            args_config["hidden_size"],
            args_config["kernel_size"],
            args_config["dropout"]
        )

        self.decoder = nn.Sequential(
            *[FFTBlock(
                    args_config["num_heads"], args_config["hidden_size"], 
                    args_config["kernel_size"], args_config["dropout"]
              ) for _ in range(args_config["melspec_fft_blocks"])]
        )

        self.linear = nn.Linear(
            args_config["hidden_size"], 
            args_config["n_mels"]
        )

    def length_regulator(
        self, 
        x: torch.Tensor, 
        durations: torch.Tensor, 
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Params:
            x: tensor with shape of (batch_size, seq_len, embed_size)
            durations: tensor with shape of (batch_size, seq_len)
            alpha: paramter that makes speech slower or faster

        Returns: 
            x_extended: tensor with shape of (batch_size, seq_len_extended, embed_size)
        """
        durations = torch.round(durations * alpha).int()

        x_extended = []
        for i in range(x.shape[0]):
            x_extended.append(
                x[i, :, :].repeat_interleave(durations[i, :], dim=0)
            )

        x_extended = pad_sequence(x_extended).transpose(0, 1)

        return x_extended

    def forward(self, x: torch.Tensor, durations_teacher: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Params: 
            x: tensor of shape (batch_size, seq_len)
            durations_teacher: tensor with shape of (batch_size, seq_len_teacher)

        Returns:
            durations_pred: tensor with shape of (batch_size, seq_len_student)
            melspec_pred: tensor with shape of (batch_size, n_mels, seq_len_teacher)
        """
        out = self.encoder(self.positional_encoding(self.phoneme_embedding(x)))
        durations_pred = self.duration_predictor(out)

        out = self.length_regulator(out, durations_teacher)
        out = self.decoder(self.positional_encoding(out))
        melspec_pred = self.linear(out).permute(0, 2, 1)

        return durations_pred, melspec_pred

    def inference(self, x: torch.Tensor, alpha=1.0) -> torch.Tensor:
        """
        Params: 
            x: tensor of shape (batch_size, seq_len)
            alpha: parameter that regulates the speed of the speech

        Returns:
            durations_pred: tensor with shape of (batch_size, seq_len_student)
            melspec_pred: tensor with shape of (batch_size, n_mels, seq_len)
        """
        out = self.encoder(self.positional_encoding(self.phoneme_embedding(x)))
        durations_pred = self.duration_predictor(out)

        # Prediction of durations is presented in logarithmic scale
        durations_pred = torch.exp(durations_pred)
        out = self.length_regulator(out, durations_pred, alpha)
        out = self.decoder(self.positional_encoding(out))

        melspec_pred = self.linear(out).permute(0, 2, 1)

        return durations_pred, melspec_pred
