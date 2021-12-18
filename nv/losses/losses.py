import torch.nn as nn
import torch

from typing import *


class AdversarialLoss(nn.Module):

    def forward(
        self, 
        outs_fake: List[torch.Tensor]
    ) -> torch.Tensor:

        loss_adv = 0

        for out_fake in outs_fake:
            loss_adv += torch.mean(
                (out_fake - 1) ** 2
            )
        
        return loss_adv


class MelSpectrogramLoss(nn.Module):

    def forward(
        self,
        melspec_real: torch.Tensor,
        melspec_fake: torch.Tensor
    ) -> torch.Tensor:

        loss_mel = torch.mean(
            torch.abs(melspec_real - melspec_fake)
        )

        return loss_mel


class FeatureMatchingLoss(nn.Module):

    def forward(
        self,
        feature_maps_real: List[torch.Tensor],
        feature_maps_fake: List[torch.Tensor]
    ) -> torch.Tensor:

        loss_fm = 0
        for feature_map_real, feature_map_fake \
            in zip(feature_maps_real, feature_maps_fake):

            loss_fm = 0
            for sub_feature_map_real, sub_feature_map_fake \
                in zip(feature_map_real, feature_map_fake):
                loss_fm += torch.mean(
                    torch.abs(sub_feature_map_real - sub_feature_map_fake) 
                )
                loss_fm /= sub_feature_map_real.shape[1]

        return loss_fm


class DiscriminatorLoss(nn.Module):

    def forward(
        self, 
        outs_real: List[torch.Tensor], 
        outs_fake: List[torch.Tensor]
    ) -> torch.Tensor:
    
        discriminator_loss = 0

        for out_real, out_fake in zip(outs_real, outs_fake): 
            discriminator_loss += torch.mean(
                (out_real - 1) ** 2 + out_fake ** 2
            )
        
        return discriminator_loss
