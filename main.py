from pathlib import Path
import argparse
import json
import wandb

import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader, Subset, random_split
import torch

from nv.utils import *


def main(config):
    if config["logger"]["use_wandb"]:
        wandb.init(project=config["logger"]["wandb_project_name"])

    fix_seed(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config["main"]["verbose"]:
        print(f"The training process will be performed on {device}.")

    if config["main"]["verbose"]:
        print("Downloading and splitting the data.")

    dataset = LJSpeechDataset(
        config["data"]["path_to_data"], 
        config["main"]["use_alignments_folder"]
    )
    train_size = int(config["trainer"]["train_ratio"] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(config["main"]["seed"])
    )

    train_dataset = Subset(
        train_dataset, np.arange(config["trainer"]["batch_size"])
    ) if config["main"]["overfit"] is True else train_dataset

    train_dataloader = DataLoader(
        train_dataset, 
        collate_fn=LJSpeechCollator(config["main"]["use_alignments_folder"]),
        batch_size=config["trainer"]["batch_size"], 
        num_workers=config["main"]["num_workers"]
    )

    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=LJSpeechCollator(config["main"]["use_alignments_folder"]),
        batch_size=config["trainer"]["batch_size"],
        num_workers=config["main"]["num_workers"]
    )

    val_dataloader = train_dataloader if config["main"]["overfit"] else val_dataloader

    if config["main"]["verbose"]:
        print("Initializing the vocoder, acoustic model, optimizer and lr_scheduler.")

    vocoder = WaveGlow().eval().to(device)
    aligner = GraphemeAligner(config).eval().to(device)
    melspectrogramer = MelSpectrogram(config).to(device)
    
    model = FastSpeech(config).to(device)
    criterion = FastSpeechMSELoss()

    trainable_params = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = ScheduledOptimizer(
        torch.optim.Adam(
            trainable_params, 
            betas=config["optimizer"]["betas"],
            eps=config["optimizer"]["eps"]
        ),
        config["optimizer"]["lr_mul"],
        config["optimizer"]["d_model"],
        config["optimizer"]["n_warmup_steps"],
    )

    if config["pretrained_model"]["load_model"]:
        if config["main"]["verbose"]:
            print("Downloading the pretrained model.")
        checkpoint = torch.load(config["pretrained_model"]["checkpoint_path"])
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.optimizer.load_state_dict(checkpoint["optimizer"])  
    
    if config["logger"]["use_wandb"]:
        wandb.watch(model)
        
    train(
        config=config, 
        model=model, 
        optimizer=optimizer,
        criterion=criterion,
        vocoder=vocoder,
        aligner=aligner,
        melspectrogramer=melspectrogramer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device
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
        config = json.load(file)

    main(config)
