from tqdm.notebook import tqdm
import wandb

import torch.nn as nn
import torch

import numpy as np

from tts.trainer.prolong_melspecs import prolong_melspecs
from tts.trainer.prepare_batch import prepare_batch
from tts.utils import get_grad_norm


def train_epoch(
    config, 
    model, 
    optimizer,
    criterion,
    aligner,
    melspectrogramer,
    train_dataloader,
    device
):
    model.train()
    train_loss = 0
    
    for batch in train_dataloader:
        batch = prepare_batch(batch, melspectrogramer, aligner, config, device)
        if config["main"]["use_alignments_folder"] and \
            batch.tokens.shape[1] != batch.durations.shape[1]: 
            continue

        durations_pred, melspec_pred = model(batch.tokens, batch.durations)
        melspec_pred, batch.melspec = prolong_melspecs(
            melspec_pred, batch.melspec, config, device
        )    

        loss = criterion(durations_pred, batch.durations, melspec_pred, batch.melspec)
        train_loss += loss.item()

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), config["trainer"]["grad_norm_clip"])

        if config["logger"]["use_wandb"]:             
            wandb.log({
                "Train Loss on Batch": loss.item(),
                "Gradient Norm": get_grad_norm(model),
                "Learning Rate": optimizer.optimizer.param_groups[0]['lr']
            })
        
        optimizer.step_and_update_lr()
        optimizer.zero_grad()
        
    return train_loss / len(train_dataloader)  


def validate_epoch(
    config, 
    model,
    vocoder,
    aligner,
    melspectrogramer,
    val_dataloader,
    device
):
    model.eval()
    val_loss = 0

    melspec_loss = nn.MSELoss()

    with torch.no_grad():
        for batch in val_dataloader:
            batch = prepare_batch(batch, melspectrogramer, aligner, config, device)
            if config["main"]["use_alignments_folder"] and \
                batch.tokens.shape[1] != batch.durations.shape[1]: 
                continue

            durations_pred, melspec_pred = model.inference(batch.tokens)
            melspec_pred, batch.melspec = prolong_melspecs(
                melspec_pred, batch.melspec, config, device
            )    
            
            loss = melspec_loss(melspec_pred, batch.melspec)
            val_loss += loss.item()

            if config["logger"]["use_wandb"]:             
                wandb.log({"Validation Loss on Batch": loss.item()})

        if config["logger"]["use_wandb"]:
            random_idx = np.random.randint(0, batch.waveform.shape[0])
            wandb.log({
                "Predicted Spectrogram": wandb.Image(
                    melspec_pred[random_idx, :, :].detach().cpu().numpy(), 
                    caption=batch.transcript[random_idx].capitalize()
                ),
                "True Spectrogram": wandb.Image(
                    batch.melspec[random_idx, :, :].detach().cpu().numpy(),
                    caption=batch.transcript[random_idx].capitalize()
                )
            })

            wav_pred = vocoder.inference(
                melspec_pred[random_idx, :, :].unsqueeze(0)
            ).squeeze()
            wandb.log({
                "Predicted Audio": wandb.Audio(
                    wav_pred.detach().cpu().numpy(), 
                    sample_rate=config["preprocessing"]["sr"], 
                    caption=batch.transcript[random_idx].capitalize()
                ),
                "True Audio": wandb.Audio(
                    batch.waveform[random_idx].detach().cpu().numpy(),
                    sample_rate=config["preprocessing"]["sr"], 
                    caption=batch.transcript[random_idx].capitalize()
                )
            })
    
    return val_loss / len(val_dataloader)


def train(
    config, 
    model, 
    optimizer,
    criterion,
    vocoder,
    aligner,
    melspectrogramer,
    train_dataloader,
    val_dataloader,
    device
):    
    history_val_loss = []
    epoch = 0

    #for epoch in tqdm(range(config["trainer"]["num_epoch"])):
    while True:
        epoch += 1

        train_loss = train_epoch(
            config, model, optimizer, criterion, aligner,
            melspectrogramer, train_dataloader, device
        )

        val_loss = validate_epoch(
            config, model, vocoder, aligner,
            melspectrogramer, val_dataloader, device
        )
        
        history_val_loss.append(val_loss)
         
        if config["logger"]["use_wandb"]:             
            wandb.log({
                "Epoch": epoch,
                "Global Train Loss": train_loss,
                "Global Validation Loss": val_loss
            })  
        
        if val_loss <= min(history_val_loss):
            arch = type(model).__name__
            state = {
                "arch": arch,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.optimizer.state_dict(),
                "config": config
            }
            best_path = config["pretrained_model"]["path_to_save"] + "/best.pt"
            torch.save(state, best_path)
