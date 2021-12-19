import wandb

from nv.trainer.prepare_batch import *
from nv.losses import *
from nv.utils import *


def train_epoch(
    config, 
    train_dataloader,
    generator, 
    optimizer_generator, 
    scheduler_generator, 
    discriminator, 
    optimizer_discriminator, 
    scheduler_discriminator, 
    melspectrogramer, 
    melspectrogramer_for_loss,
    device
):
    generator.train()
    discriminator.train()

    adversarial_loss = AdversarialLoss()
    feature_loss = FeatureMatchingLoss()
    melspec_loss = MelSpectrogramLoss()

    discriminator_loss = DiscriminatorLoss()

    train_melspec_loss = 0
    for batch in train_dataloader:
        batch = prepare_batch(
            batch, melspectrogramer, melspectrogramer_for_loss, 
            device, for_training=True
        )
        wav_real, melspec_real = batch.waveform, batch.melspec

        optimizer_generator.zero_grad()

        wav_fake = generator(melspec_real)
        melspec_fake = melspectrogramer_for_loss(wav_fake)
        out_discr = discriminator(wav_real, wav_fake)

        loss_adv = adversarial_loss(out_discr["outs_fake"])
        loss_fm = feature_loss(out_discr["feature_maps_real"], out_discr["feature_maps_fake"])
        loss_mel = melspec_loss(melspec_real, melspec_fake)
        train_melspec_loss += loss_mel.item()

        loss_gen = loss_adv + 2 * loss_fm + 45 * loss_mel
        loss_gen.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), config.grad_norm_clip)

        if config.use_wandb:             
            wandb.log({
                "Train Adversarial Loss on Batch": loss_adv.item(),
                "Train Feature Matching Loss on Batch": loss_fm.item(),
                "Train Melspec Loss on Batch": loss_mel.item(),
                "Train Generator Loss on Batch": loss_gen.item(),
                "Generator Gradient Norm": get_grad_norm(generator),
                "Generator Learning Rate": optimizer_generator.param_groups[0]['lr']
            })

        optimizer_generator.step()
        
        optimizer_discriminator.zero_grad()

        wav_fake = generator(melspec_real)
        out_discr = discriminator(wav_real, wav_fake)

        loss_discr = discriminator_loss(out_discr["outs_real"], out_discr["outs_fake"])
        loss_discr.backward()
        nn.utils.clip_grad_norm_(discriminator.parameters(), config.grad_norm_clip)

        if config.use_wandb:             
            wandb.log({
                "Train Discriminator Loss on Batch": loss_discr.item(),
                "Discriminator Gradient Norm": get_grad_norm(discriminator),
                "Discriminator Learning Rate": optimizer_discriminator.param_groups[0]['lr']
            })

        optimizer_discriminator.step()

    scheduler_generator.step()
    scheduler_discriminator.step()

    return train_melspec_loss / len(train_dataloader)


def validate_epoch(
    config, 
    val_dataloader,
    generator, 
    discriminator, 
    melspectrogramer, 
    melspectrogramer_for_loss, 
    device
): 
    generator.eval()
    discriminator.eval()

    melspec_loss = MelSpectrogramLoss()

    torch.cuda.empty_cache()

    val_melspec_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = prepare_batch(
                batch, melspectrogramer, melspectrogramer_for_loss, 
                device, for_training=False
            )
            wav_real, melspec_real = batch.waveform, batch.melspec

            wav_fake = generator(melspec_real)
            melspec_fake = melspectrogramer_for_loss(wav_fake)

            loss_mel = melspec_loss(melspec_real, melspec_fake)
            val_melspec_loss += loss_mel.item()

            if config.use_wandb:
                wandb.log({
                    "Validation Melspectrogram Loss on Batch": loss_mel.item()
                })

            melspec_fake = melspectrogramer(wav_fake)

        if config.use_wandb:
            random_idx = np.random.randint(0, wav_real.shape[0])
            wandb.log({
                "Real Spectrogram": wandb.Image(
                    melspec_real[random_idx].detach().cpu(),
                    caption=batch.transcript[random_idx].capitalize()
                ),
                "Faked Spectrogram": wandb.Image(
                    melspec_fake[random_idx].detach().cpu(), 
                    caption=batch.transcript[random_idx].capitalize()
                ),
                "Real Audio": wandb.Audio(
                    wav_real[random_idx].detach().cpu().numpy(),
                    sample_rate=config.sr, 
                    caption=batch.transcript[random_idx].capitalize()
                ),
                "Faked Audio": wandb.Audio(
                    wav_fake[random_idx].detach().cpu().numpy(), 
                    sample_rate=config.sr, 
                    caption=batch.transcript[random_idx].capitalize()
                ),
            })

    return val_melspec_loss / len(val_dataloader)


def train(
    config, 
    train_dataloader, 
    val_dataloader,
    generator, 
    optimizer_generator, 
    scheduler_generator, 
    discriminator, 
    optimizer_discriminator, 
    scheduler_discriminator, 
    melspectrogramer, 
    melspectrogramer_for_loss,
    device
):  
    history_val_melspec_loss = []
    epoch = 0

    #for epoch in tqdm(range(config.num_epoch)):
    while True:
        epoch += 1

        train_melspec_loss = train_epoch(
            config, train_dataloader,
            generator, optimizer_generator, scheduler_generator, 
            discriminator, optimizer_discriminator, scheduler_discriminator, 
            melspectrogramer, melspectrogramer_for_loss, device
        )

        val_melspec_loss = validate_epoch(
            config, val_dataloader,
            generator, discriminator, 
            melspectrogramer, melspectrogramer_for_loss, device
        )

        history_val_melspec_loss.append(val_melspec_loss)

        if config.use_wandb:             
            wandb.log({
                "Epoch": epoch,
                "Global Train Melspectrogram Loss": train_melspec_loss,
                "Global Validation Melspectrogram Loss": val_melspec_loss
            })  
        
        #if val_melspec_loss <= min(history_val_melspec_loss):
        state = {
            "generator": generator.state_dict(),
            "generator_arch": type(generator).__name__,
            "optimizer_generator": optimizer_generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "discriminator_arch": type(discriminator).__name__,
            "optimizer_discriminator": optimizer_discriminator.state_dict(),
            "config": config
        }
        torch.save(state, config.path_to_save + "/best.pt")