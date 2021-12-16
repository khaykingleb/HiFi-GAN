import torch.nn as nn
import torchaudio
import librosa
import torch


class MelSpectrogram(nn.Module):

    def __init__(self, config):
        super(MelSpectrogram, self).__init__()

        mel_config = config["preprocessing"]["spectrogram"]["args"]

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=mel_config["sr"],
            win_length=mel_config["win_length"],
            hop_length=mel_config["hop_length"],
            n_fft=mel_config["n_fft"],
            f_min=mel_config["f_min"],
            f_max=mel_config["f_max"],
            n_mels=mel_config["n_mels"]
        )

        # There is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = mel_config["power"]

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=mel_config["sr"],
            n_fft=mel_config["n_fft"],
            n_mels=mel_config["n_mels"],
            fmin=mel_config["f_min"],
            fmax=mel_config["f_max"]
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Params:
            audio: Expected shape is [B, T]
        Returns: 
            mel: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio) \
                  .clamp_(min=1e-5) \
                  .log_()

        return mel
