import torchaudio
import torch


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root: str):
        super().__init__(root=root)
    
    def __getitem__(self, index: int):
        waveform, sr, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()
        
        return transcript, waveform, waveform_length
