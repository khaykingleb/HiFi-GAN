import torch


def prolong_melspecs(
    melspec_pred, 
    melspec_true,
    config,
    device
):
    """ Pad short spectrogramm with silence. """
    
    def prolong_short(
        short_melspec, 
        long_melspec,
        config,
        device
    ) -> torch.Tensor:

        fill_tensor = torch.ones((
            short_melspec.shape[0], 
            short_melspec.shape[1], 
            long_melspec.shape[2] - short_melspec.shape[2]
        )).to(device) * config["preprocessing"]["spectrogram"]["args"]["pad_value"]

        prolonged_melspec = torch.cat([short_melspec, fill_tensor], dim=2)

        return prolonged_melspec
        
    if melspec_pred.shape[2] <= melspec_true.shape[2]:
        melspec_pred = prolong_short(
            melspec_pred, melspec_true, config, device
        )  

    else:
        melspec_true = prolong_short(
            melspec_true, melspec_pred, config, device
        ) 

    return melspec_pred, melspec_true
