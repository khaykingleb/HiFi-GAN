import torchaudio
import torch

import numpy as np
import sys 
sys.path.append("alignments")

import string

class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root: str, use_alignments_folder: bool):
        super().__init__(root=root)
        self.tokenizer = torchaudio.pipelines \
            .TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self.use_alignments_folder = use_alignments_folder

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveforn_length = torch.tensor([waveform.shape[-1]]).int()
        
        if not self.use_alignments_folder:
            transcript = transcript.encode("ascii", "ignore").decode() \
                .translate(str.maketrans('', '', string.punctuation)) \
                .strip()
        
        tokens, token_lengths = self.tokenizer(transcript)

        durations = torch.tensor(
            np.load(f"alignments/{index}.npy")
        ) if self.use_alignments_folder else None
        
        return waveform, waveforn_length, transcript, tokens, token_lengths, durations
    
    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self.tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)

        return result
