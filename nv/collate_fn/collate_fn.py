from typing import *
from dataclasses import dataclass

from torch.nn.utils.rnn import pad_sequence
import torch


@dataclass
class Batch:
    transcript: List[str]
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    melspec: Optional[torch.Tensor] = None
    melspec_loss: Optional[torch.Tensor] = None
        
    def to(self, device: torch.device) -> "Batch":
        batch = Batch(
            self.transcript,
            self.waveform.to(device),
            self.waveform_length,
            self.melspec.to(device),
            self.melspec_loss.to(device)
        )
        return batch
class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Dict:
        transcript, waveform, waveform_length = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)

        waveform_length = torch.cat(waveform_length)

        batch = Batch(
            transcript, waveform, waveform_length
        )

        return batch
