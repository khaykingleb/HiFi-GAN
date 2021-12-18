import torch.nn as nn
from librosa.filters import mel 
import torch


def dynamic_range_compression_torch(x, C: int = 1, clip_val: float = 1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)
    

def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    out = dynamic_range_compression_torch(magnitudes)
    return out


class MelSpectrogram(nn.Module):

    def __init__(self, config, for_loss=False):
        super(MelSpectrogram, self).__init__()

        self.f_max = config.f_max if for_loss is False else config.f_max_loss
        self.pad_size = (config.n_fft - config.hop_length) // 2
        self.config = config

    def forward(self, audio: torch.Tensor) -> torch.Tensor:

        spec_1 = mel(
            self.config.sr, self.config.n_fft, self.config.n_mels, 
            self.config.f_min, self.f_max
        )

        audio = torch.nn.functional.pad(
            audio.unsqueeze(1), (self.pad_size, self.pad_size), mode='reflect'
        ).squeeze(1)

        spec_2 = torch.stft(
            audio, 
            self.config.n_fft, 
            hop_length=self.config.hop_length, 
            win_length=self.config.win_length, 
            window=torch.hann_window(self.config.win_length),
            center=self.config.center, 
            pad_mode='reflect', 
            normalized=False, 
            onesided=True
        )

        spec_2 = torch.sqrt(spec_2.pow(2).sum(-1) + (1e-9))
        
        melspec = torch.matmul(torch.from_numpy(spec_1).float(), spec_2)
        melspec = spectral_normalize_torch(melspec)

        return melspec
