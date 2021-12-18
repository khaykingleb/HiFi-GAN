from torch.nn.utils import weight_norm, spectral_norm
import torch.nn as nn
import torch

from typing import *

from nv.utils import get_padding


LRELU_SLOPE = 1e-1


class PeriodSubDiscriminator(nn.Module):

    """
    Operates on disjoint samples of raw waveforms.
    """
    
    def __init__(
        self, 
        period: int, 
        powers: List[int] = [0, 5, 7, 9, 10]
    ):
        super(PeriodSubDiscriminator, self).__init__()
        
        self.period = period

        self.convs = nn.Sequential(
            *[  
                nn.Sequential(
                    weight_norm(
                        nn.Conv2d(
                            in_channels=2 ** powers[ell], 
                            out_channels=2 ** powers[ell + 1], 
                            kernel_size=(5, 1), 
                            stride=(3, 1), 
                            padding=(get_padding(5, 1), 0)
                        )
                    ),
                    nn.LeakyReLU(
                        negative_slope=LRELU_SLOPE, 
                        inplace=True
                    )
                ) 
                for ell in range(4)
            ],
            nn.Sequential(
                weight_norm(
                    nn.Conv2d(
                        in_channels=1024, 
                        out_channels=1024, 
                        kernel_size=(5, 1), 
                        stride=1, 
                        padding=(2, 0)
                    )
                ),
                nn.LeakyReLU(
                    negative_slope=LRELU_SLOPE, 
                    inplace=True
                )
            ),
            nn.Sequential(
                weight_norm(
                    nn.Conv2d(
                        in_channels=1024, 
                        out_channels=1, 
                        kernel_size=(3, 1), 
                        stride=1, 
                        padding=(1, 0)
                    )
                )
            )
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Params:
            x: wav — torch.Tensor with shape of (batch_size, seq_len)
        
        Retruns: 
            out: torch.Tensor with shape of (batch_size, seq_len)
            feature_map: List[tensor.Torch] 
        """
        batch_size, seq_len = x.shape
        if seq_len % self.period != 0:
            pad_len = self.period - seq_len % self.period
            x = F.pad(x, (0, pad_len), "reflect")
            seq_len += pad_len

        # Reshape from 1D to 2D
        out = x.view(batch_size, 1, seq_len // self.period, self.period)

        feature_map = []
        for conv_ell in self.convs:
            out = conv_ell(out)
            feature_map.append(out)
        
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        
        return out, feature_map


class MultiPeriodDiscriminator(nn.Module):

    def __init__(self, config):
        super(MultiPeriodDiscriminator, self).__init__()

        self.sub_discriminators = nn.Sequential(
            *[
                PeriodSubDiscriminator(period)
                for period in config.mpd_periods
            ]
        )
        
    def forward(
        self, 
        wav_real: torch.Tensor, 
        wav_fake: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Params:
            wav_real: torch.Tensor with shape of (batch_size, seq_len)
            wav_fake: generated wav torch.Tensor with shape of (batch_size, seq_len)
        """
        mpd_outs_real, mpd_outs_fake = [], []
        feature_maps_real, feature_maps_fake = [], []

        for sub_discriminator in self.sub_discriminators:
            out_real, feature_map_real = sub_discriminator(wav_real)
            mpd_outs_real.append(out_real)
            feature_maps_real.append(feature_map_real)

            out_fake, feature_map_fake = sub_discriminator(wav_fake)
            mpd_outs_fake.append(out_fake)
            feature_maps_fake.append(feature_map_fake)
        
        return mpd_outs_real, mpd_outs_fake, feature_maps_real, feature_maps_fake


class ScaleSubDiscriminator(nn.Module):

    """
    Operates on smoothed waveforms.
    """
    
    def __init__(
        self,
        powers: List[int] = [0, 7, 7, 8, 9, 10, 10, 10],
        kernel_sizes: List[int] = [15, 41, 41, 41, 41, 41, 5], 
        strides: List[int] = [1, 2, 2, 4, 4, 1, 1],
        groups: List[int] = [1, 4, 16, 16, 16, 16, 1],
        padding: List[int] = [7, 20, 20, 20, 20, 20, 2],
        use_spectral_norm=False
    ):
        super(ScaleSubDiscriminator, self).__init__()

        norm = weight_norm if use_spectral_norm == False else spectral_norm

        self.convs = nn.Sequential(
            *[  
                nn.Sequential(
                    norm(
                        nn.Conv1d(
                            in_channels=2 ** powers[ell], 
                            out_channels=2 ** powers[ell + 1], 
                            kernel_size=kernel_sizes[ell], 
                            stride=strides[ell], 
                            groups=groups[ell],
                            padding=padding[ell]
                        )
                    ),
                    nn.LeakyReLU(
                        negative_slope=LRELU_SLOPE, 
                        inplace=True
                    )
                ) 
                for ell in range(7)
            ], 
            nn.Sequential(
                norm(
                    nn.Conv1d(
                        in_channels=1024, 
                        out_channels=1, 
                        kernel_size=3, 
                        stride=1, 
                        padding=1
                    )
                )
            )
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feature_map = []
        
        for conv_ell in self.convs:
            x = conv_ell(x)
            feature_map.append(x)
    
        out = torch.flatten(x, start_dim=1, end_dim=-1)
    
        return out, feature_map


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, config):
        super(MultiScaleDiscriminator, self).__init__()

        self.sub_discriminators = nn.Sequential(
            *[
                ScaleSubDiscriminator(use_spectral_norm=config.msd_use_spectral_norm[i])
                for i in range(len(config.msd_use_spectral_norm))
            ]
        )
        
        self.mean_pool = nn.AvgPool1d(
            kernel_size=4, 
            stride=2, 
            padding=2
        )
    
    def forward(
        self, 
        wav_real: torch.Tensor, 
        wav_fake: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Params:
            wav_real: real wav torch.Tensor with shape of (batch_size, seq_len)
            wav_fake: fake wav torch.Tensor with shape of (batch_size, seq_len)
        """
    
        wav_real, wav_fake = wav_real.unsqueeze(1), wav_fake.unsqueeze(1)

        msd_outs_real, msd_outs_fake = [], []
        feature_maps_real, feature_maps_fake = [], []

        for i, sub_discriminator in enumerate(self.sub_discriminators):
            
            if i != 0:
                wav_real = self.mean_pool(wav_real)
                wav_fake = self.mean_pool(wav_fake)

            out_real, feature_map_real = sub_discriminator(wav_real)
            msd_outs_real.append(out_real)
            feature_maps_real.append(feature_map_real)

            out_gen, feature_map_fake = sub_discriminator(wav_fake)
            msd_outs_fake.append(out_gen)
            feature_maps_fake.append(feature_map_fake)

        return msd_outs_real, msd_outs_fake, feature_maps_real, feature_maps_fake


class HiFiDiscriminator(nn.Module):
    
    def __init__(self, config): 
        super(HiFiDiscriminator, self).__init__()

        self.multi_period_discriminator = MultiPeriodDiscriminator(config)
        self.multi_scale_discriminator = MultiScaleDiscriminator(config)
    
    def forward(
        self,
        wav_real: torch.Tensor, 
        wav_fake: torch.Tensor
    ) -> Tuple[Tuple[List[torch.Tensor], List[List[torch.Tensor]]]]:

        out_mpd = self.multi_period_discriminator(wav_real, wav_fake)
        out_msd = self.multi_scale_discriminator(wav_real, wav_fake)

        NAMES = [
            "outs_real",
            "outs_fake",
            "feature_maps_real",
            "feature_maps_fake"
        ]

        out_discr = {
            key: v_mpd + v_msd for (key, v_mpd, v_msd) in zip(NAMES, out_mpd, out_msd)
        }

        return out_discr
