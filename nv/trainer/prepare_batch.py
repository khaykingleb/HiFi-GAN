import random
import torch.nn.functional as F
import torch

from nv.spectrogram import MelSpectrogram
from nv.collate_fn import Batch


def prepare_batch(
    batch: Batch, 
    melspectrogramer: MelSpectrogram, 
    melspectrogramer_for_loss: MelSpectrogram,
    device: torch.device,
    for_training: bool, 
    segment_size: int = 8192
):  
    if for_training: 
        waveform_segment = []

        for idx in range(batch.waveform.shape[0]):
            waveform_length = batch.waveform_length[idx]
            waveform = batch.waveform[idx][:waveform_length]

            if waveform_length >= segment_size:
                difference = waveform_length - segment_size
                waveform_start = random.randint(0, difference - 1)
                waveform_segment.append(
                    waveform[waveform_start:waveform_start + segment_size]
                )
            else:
                waveform_segment.append(
                    F.pad(waveform, (0, segment_size - waveform_length))
                )
        
        batch.waveform = torch.vstack(waveform_segment)
    
    batch.melspec = melspectrogramer(batch.waveform.to(device))
    batch.melspec_loss = melspectrogramer_for_loss(batch.waveform.to(device))
    
    return batch.to(device)
