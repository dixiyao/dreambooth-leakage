"""
The customized embedding model for model weights based on CLIP model.
"""

from config import Config
import torch

import einops
from torch import nn
from diffusers import UNet2DConditionModel
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from transformers import CLIPTextModel
from peft import get_peft_model, LoraConfig

CLIP_MAX_LENGTH = 77
UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]

swap = lambda w, h: (h, w)


class HyperNet(torch.nn.Module):
    "A hypernet that takes a matrix and outputs a vector."

    def __init__(self, matrix_height, matrix_width, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layer_num = 4
        self.encoder = nn.ModuleList()
        for i in range(layer_num):
            if_last = False
            if_start = i == 0
            if i == layer_num - 1:
                if_last = True
            self.encoder.append(
                self.build_layer(matrix_height, matrix_height, if_last, if_start)
            )
        self.projection = torch.nn.Linear(matrix_width, 512)

    def build_layer(
        self,
        input_channel,
        output_channel,
        last=False,
        if_start=False,
    ):
        "Build each layer for encoder hypernet."
        layer = nn.Sequential(
            nn.LeakyReLU() if not if_start else nn.Identity(),
            nn.InstanceNorm1d(input_channel),
            nn.Conv1d(
                input_channel,
                input_channel,
                3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(input_channel),
            nn.Conv1d(
                input_channel,
                output_channel if not last else 1,
                3,
                stride=1,
                padding=1,
            ),
            nn.Tanh() if last else nn.Identity(),
        )
        return layer

    def forward(self, x):
        """
        The forward function for HyperNet, dong matrix transformation.
        """
        x = torch.reshape(x, (1, x.shape[0], x.shape[1]))
        for module in self.encoder:
            x = module(x)
        x = self.projection(x)
        return x


class HyperEncoder(torch.nn.Module):
    """Use a frozen CLIP model to embed the LoRA weights."""

    def __init__(
        self, clip_pretrained_model="openai/clip-vit-base-patch32", projection_dim=768
    ) -> None:
        super().__init__()
        self.clip_model = CLIPTextModel.from_pretrained(clip_pretrained_model)
        unet_config = LoraConfig(
            r=Config().training.lora.lora_r,
            lora_alpha=Config().training.lora.lora_alpha,
            target_modules=UNET_TARGET_MODULES,
        )
        pretrained_unet = UNet2DConditionModel.from_pretrained(
            Config().training.pretrained_model_name_or_path,
            subfolder="unet",
            cache_dir=Config().model.cache_path,
            local_files_only=Config().model.local_files_only,
        )
        pretrained_unet = get_peft_model(pretrained_unet, unet_config)
        self.hypernetworks = []
        for param in pretrained_unet.parameters():
            if param.requires_grad == True:
                w, h = param.shape
                if w > h:
                    w, h = swap(w, h)
                self.hypernetworks.append(HyperNet(w, h))
        self.hypernetworks = torch.nn.ModuleList(self.hypernetworks)
        self.task_token = torch.nn.Parameter(torch.randn(1, 1, 512))
        self.first_norm = nn.LayerNorm(512)
        self.mean = torch.nn.Linear(512, projection_dim)
        self.log_var = torch.nn.Linear(512, projection_dim)
        # Timestep embedding
        self.time_proj = Timesteps(512, True, 0)
        self.time_embedding = TimestepEmbedding(512, 512)
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, lora_weights, total_steps):
        """
        The forward function for HyperCLIPModel.
        First, embed all LoRA matrix into a 768 vector.
        Second, put them together with the task token.
        Third, use a CLIP model to embed the whole vector.
        """
        outputs = self.task_token
        timestep = torch.Tensor([total_steps]).to(Config().device())
        time_embed = self.time_proj(timestep)
        time_embedding = self.time_embedding(time_embed)
        time_embedding = einops.rearrange(time_embedding, "b c->b 1 c")
        outputs = torch.concat((outputs, time_embedding), dim=1)
        for index, weights in enumerate(lora_weights):
            if weights.shape[0] > weights.shape[1]:
                weights = torch.transpose(weights, 0, 1)
            weight_token = self.hypernetworks[index](weights)
            outputs = torch.concat((outputs, weight_token), dim=1)
        hidden_states = self.first_norm(outputs)

        encoder_outputs = self.clip_model.text_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=self.clip_model.text_model.config.output_attentions,
            output_hidden_states=self.clip_model.text_model.config.output_hidden_states,
            return_dict=self.clip_model.text_model.config.return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = self.clip_model.text_model.final_layer_norm(last_hidden_state)

        pooled_output = pooled_output[:, :CLIP_MAX_LENGTH, :]

        # variable autoencoder
        mu = self.mean(pooled_output)
        logvar = self.log_var(pooled_output)
        z = self.reparameterize(mu, logvar)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=(1, 2)), dim=0
        )

        return z, kld_loss


if __name__ == "__main__":
    model_updates = []
    unet_config = LoraConfig(
        r=Config().training.lora.lora_r,
        lora_alpha=Config().training.lora.lora_alpha,
        target_modules=UNET_TARGET_MODULES,
    )
    pretrained_unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        cache_dir=Config().model.cache_path,
        local_files_only=Config().model.local_files_only,
    )
    pretrained_unet = get_peft_model(pretrained_unet, unet_config)
    for param in pretrained_unet.parameters():
        if param.requires_grad:
            model_updates.append(param)
            print(param.shape)
    nn_encoder = HyperEncoder()
    nn_encoder(model_updates, 768)
    import pickle
    import sys

    print(sys.getsizeof(pickle.dumps(nn_encoder.clip_model.state_dict())) / 1024**2)
