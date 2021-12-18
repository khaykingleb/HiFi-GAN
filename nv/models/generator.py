from torch.nn.utils import weight_norm
import torch.nn as nn
import torch

from typing import *

from nv.utils import get_padding, init_weights


LRELU_SLOPE = 1e-1


class ResSubBlock(nn.Module):

    def __init__(
        self, 
        channels: int,  
        kernel_size: int, 
        dilation: int
    ):
        super(ResSubBlock, self).__init__()

        self.res_sub_block = nn.Sequential(
            nn.LeakyReLU(
                negative_slope=LRELU_SLOPE, 
                inplace=True
            ),
            weight_norm(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=get_padding(kernel_size, dilation)
                )
            )
        )

        self.res_sub_block.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: torch.Tensor with shape of (batch_size, n_mels, seq_len)
        
        Returns: 
            out: torch.Tensor with shape of (batch_size, n_mels, seq_len)
        """
        out = x + self.res_sub_block(x) 
        
        return out
        

class ResBlock(nn.Module):

    def __init__(
        self, 
        channels: int,
        kernel_size: int, 
        dilation_rates: int
    ):
        super(ResBlock, self).__init__()

        self.res_block = nn.Sequential(
            *[
                ResSubBlock(
                    channels, 
                    kernel_size, 
                    dilation_rates[m][ell]
                )
                for m in range(len(dilation_rates)) for ell in range(len(dilation_rates[m]))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: torch.Tensor with shape of (batch_size, n_mels, seq_len)
        
        Returns: 
            out: torch.Tensor with shape of (batch_size, n_mels, seq_len)
        """
        out = self.res_block(x)

        return out


class MultiReceptiveFieldFusion(nn.Module):

    def __init__(
        self, 
        channels: int, 
        resblock_kernel_sizes: int, 
        dilation_rates: int
    ):
        super(MultiReceptiveFieldFusion, self).__init__()
        
        self.mrf = nn.Sequential(
            *[
                ResBlock(
                    channels, 
                    resblock_kernel_sizes[n], 
                    dilation_rates[n]
                ) 
                for n in range(len(resblock_kernel_sizes))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: torch.Tensor with shape of (batch_size, n_mels, seq_len)
        
        Returns: 
            out: torch.Tensor with shape of (batch_size, n_mels, seq_len)
        """
        for i, res_block in enumerate(self.mrf):
            if i == 0:
                out = res_block(x)
            else:
                out += res_block(x)
            
        return out / len(self.mrf)


class UpsamplerBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int, 
        resblock_kernel_sizes: List[int], 
        dilation_rates: List[int]
    ):
        super(UpsamplerBlock, self).__init__()

        self.upsampler_block = nn.Sequential(
            nn.LeakyReLU(
                negative_slope=LRELU_SLOPE, 
                inplace=True
            ),
            weight_norm(
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - stride) // 2,
                )
            ),
            MultiReceptiveFieldFusion(
                out_channels, 
                resblock_kernel_sizes, 
                dilation_rates
            )
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: torch.Tensor with shape of (batch_size, n_mels, seq_len)

        Returns:
            out: torch.Tensor with shape of (batch_size, n_mels // 2, seq_len * 8)
        """
        out = self.upsampler_block(x)

        return out


class HiFiGenerator(nn.Module):

    def __init__(self, config):
        super(HiFiGenerator, self).__init__()
        
        dilation_rates = [config.dilation_rates] * 3

        self.conv_in = weight_norm(
            nn.Conv1d(
                in_channels=config.n_mels,
                out_channels=config.initial_channels,
                kernel_size=7,
                stride=1,
                padding=3
            )
        )

        self.upsampler = nn.Sequential(
            *[
                UpsamplerBlock(
                    in_channels=config.initial_channels // (2 ** i),
                    out_channels=config.initial_channels // (2 ** (i + 1)),
                    kernel_size=config.upsample_kernel_sizes[i],
                    stride=config.upsample_kernel_sizes[i] // 2, 
                    resblock_kernel_sizes=config.resblock_kernel_sizes,
                    dilation_rates=dilation_rates
                )
                for i in range(len(config.upsample_kernel_sizes))
            ]
        )

        self.conv_out = nn.Sequential(
            nn.LeakyReLU(
                negative_slope=LRELU_SLOPE, 
                inplace=True
            ),
            weight_norm(
                nn.ConvTranspose1d(
                    in_channels=config.initial_channels // (2 ** len(config.upsample_kernel_sizes)),
                    out_channels=1,
                    kernel_size=7,
                    padding=3
                )
            ),
            nn.Tanh()
        )

        self.conv_out.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: melspec torch.Tensor with shape of (batch_size, n_mels, seq_len)
        
        Returns: 
            out: wav_fake torch.Tensor with shape of (batch_size, seq_len_wav_real)
        """
        out = self.conv_out(self.upsampler(self.conv_in(x)))

        return out.squeeze()
