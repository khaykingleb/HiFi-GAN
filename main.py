from pathlib import Path
import argparse
import wandb
import json

import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader, random_split
import torch

from nv.spectrogram import MelSpectrogram
from nv.collate_fn import LJSpeechCollator
from nv.datasets import LJSpeechDataset
from nv.trainer import *
from nv.models import *
from nv.utils import *


def main(config):

    if config.use_wandb:
        wandb.init(project=config.wandb_project_name)

    fix_seed(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config.verbose:
        print(f"The training process will be performed on {device}.")
        print("Downloading and splitting the data.")

    dataset = LJSpeechDataset(config.path_to_data)
    train_size = int(config.train_ratio * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    train_dataloader = DataLoader(
        train_dataset, 
        collate_fn=LJSpeechCollator(),
        batch_size=config.batch_size, 
        #num_workers=config.num_workers
    )

    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=LJSpeechCollator(),
        batch_size=config.batch_size, 
        #num_workers=config.num_workers
    )

    melspectrogramer = MelSpectrogram(config, for_loss=False).to(device)
    melspectrogramer_for_loss = MelSpectrogram(config, for_loss=True).to(device)

    if config.verbose:
        print("Initializing discriminator, generator, optimizers and lr_schedulers.")

    generator = HiFiGenerator(config).to(device)
    trainable_params_generator = filter(
        lambda param: param.requires_grad, generator.parameters()
    )
    optimizer_generator = torch.optim.AdamW(
        trainable_params_generator, 
        betas=(config.adam_beta_1, config.adam_beta_2), 
        weight_decay=config.weight_decay, 
        lr=config.learning_rate
    ) 
    scheduler_generator = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_generator, 
        gamma=config.gamma
    ) 

    discriminator = HiFiDiscriminator(config).to(device) 
    trainable_params_discriminator = filter(
        lambda param: param.requires_grad, discriminator.parameters()
    )
    optimizer_discriminator = torch.optim.AdamW(
        trainable_params_discriminator, 
        betas=(config.adam_beta_1, config.adam_beta_2), 
        weight_decay=config.weight_decay, 
        lr=config.learning_rate
    ) 
    scheduler_discriminator = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_discriminator, 
        gamma=config.gamma
    ) 

    #if config.load_model:
    #    if config.verbose:
    #        print("Downloading the pretrained generator.")
    #    checkpoint = torch.load(config["pretrained_model"]["checkpoint_path"])
    #    model.load_state_dict(checkpoint["state_dict"])
    #    optimizer.optimizer.load_state_dict(checkpoint["optimizer"])  
    
    if config.use_wandb:
        wandb.watch(generator)
        wandb.watch(discriminator)

    train(
        config, train_dataloader, val_dataloader,
        generator, optimizer_generator, scheduler_generator, 
        discriminator, optimizer_discriminator, scheduler_discriminator, 
        melspectrogramer, melspectrogramer_for_loss
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Config file path"
    )

    args = argparser.parse_args()
    config_path = Path(args.config)
    with config_path.open("r") as file:
        config = AttrDict(json.load(file))

    main(config)
