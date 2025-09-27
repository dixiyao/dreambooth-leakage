"""
The pipeline to invert the trigger words with the method of matching model updates.
"""

import os
from pathlib import Path
from itertools import chain
import itertools

import torch
import numpy as np
from peft import get_peft_model, LoraConfig
from tqdm import tqdm

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)

from finetune_dreambooth import (
    generate_class_images,
    finetune,
)
from valid_attack import validation_process, validation_one_step
from matching_updates_closure import weight_closure
from config import Config


UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]
TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]


def get_unet():
    """
    Get a new pre-trained Unet without any fine-tuning.
    """
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
    pretrained_unet.print_trainable_parameters()
    return pretrained_unet


class MatchingPipeline:
    """
    The pipeline for inverting the trigger words.

    To initialize the pipeline, users need to provide a pre-trained diffusion model.
    """

    def __init__(self, diffusion_model_name, device, valid_dataloader) -> None:
        super().__init__()
        if not os.path.exists(os.path.join(Config().model.save_path, "results")):
            os.mkdir(os.path.join(Config().model.save_path, "results"))
        self.pre_trained_diffusion_model_name = diffusion_model_name
        self.device = device
        self.pretrained_unet = None
        self.valid_dataloader = valid_dataloader

        self.unet = None
        self.pretrained_text_encoder = None
        if hasattr(Config().training, "seed"):
            seed = Config().training.seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        self.weight_dtype = torch.float32
        if Config().training.training_precesion == "fp16":
            self.weight_dtype = torch.float16
        elif Config().training.training_precesion == "bf16":
            self.weight_dtype = torch.bfloat16

        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pre_trained_diffusion_model_name,
            subfolder="tokenizer",
            cache_dir=Config().model.cache_path,
            local_files_only=Config().model.local_files_only,
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.pre_trained_diffusion_model_name,
            subfolder="vae",
            cache_dir=Config().model.cache_path,
            local_files_only=Config().model.local_files_only,
        )
        self.vae.requires_grad_(False)
        self.vae.to(self.device, dtype=self.weight_dtype)

        self.last_class_prompt = None

    def fine_tune_process(
        self,
        instance_prompt,
        class_prompt,
        instance_images,
    ):
        """
        A wrapped up fine-tune process for the diffusion model with LoRA.
        """
        text_encoder_config = LoraConfig(
            r=Config().training.lora.lora_text_encoder_r,
            lora_alpha=Config().training.lora.lora_text_encoder_alpha,
            target_modules=TEXT_ENCODER_TARGET_MODULES,
        )
        unet_config = LoraConfig(
            r=Config().training.lora.lora_r,
            lora_alpha=Config().training.lora.lora_alpha,
            target_modules=UNET_TARGET_MODULES,
        )
        self.pretrained_unet = UNet2DConditionModel.from_pretrained(
            self.pre_trained_diffusion_model_name,
            subfolder="unet",
            cache_dir=Config().model.cache_path,
            local_files_only=Config().model.local_files_only,
        )
        self.pretrained_unet = get_peft_model(self.pretrained_unet, unet_config)
        self.pretrained_unet.print_trainable_parameters()
        self.pretrained_text_encoder = CLIPTextModel.from_pretrained(
            self.pre_trained_diffusion_model_name,
            subfolder="text_encoder",
            cache_dir=Config().model.cache_path,
            local_files_only=Config().model.local_files_only,
        )
        self.pretrained_text_encoder = get_peft_model(
            self.pretrained_text_encoder, text_encoder_config
        )
        self.pretrained_text_encoder.print_trainable_parameters()
        # Remove the class images
        if class_prompt != self.last_class_prompt:
            self.last_class_prompt = class_prompt
            for image in Path(Config().training.class_data_dir).iterdir():
                image.unlink()
        # Generate class images
        generate_class_images(self.pre_trained_diffusion_model_name, class_prompt)
        # Prepare for the fine-tune process
        # Optimizer creation
        params_to_optimize = (
            itertools.chain(
                self.pretrained_unet.parameters(),
                self.pretrained_text_encoder.parameters(),
            )
            if Config().training.train_text_encoder
            else self.pretrained_unet.parameters()
        )
        dreambooth_optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=Config().training.optim.learning_rate,
            betas=(
                Config().training.optim.adam_beta1,
                Config().training.optim.adam_beta2,
            ),
            weight_decay=Config().training.optim.adam_weight_decay,
            eps=Config().training.optim.adam_epsilon,
        )
        lr_scheduler = get_scheduler(
            Config().training.optim.lr_scheduler,
            optimizer=dreambooth_optimizer,
            num_warmup_steps=0,
            num_training_steps=0,
            num_cycles=Config().training.optim.lr_num_cycles,
            power=Config().training.optim.lr_power,
        )

        self.pretrained_text_encoder = self.pretrained_text_encoder.to(
            self.device, dtype=self.weight_dtype
        )
        self.pretrained_unet = self.pretrained_unet.to(
            self.device, dtype=self.weight_dtype
        )
        # Step 3: Start the fine-tune process
        # Get the model weights embeddings
        lora_updates = finetune(
            Config().training.max_train_steps,
            self.vae,
            unet=self.pretrained_unet,
            text_encoder=self.pretrained_text_encoder,
            instance_prompt=instance_prompt,
            class_prompt=class_prompt,
            instance_images=instance_images,
            optimizer=dreambooth_optimizer,
            lr_scheduler=lr_scheduler,
            noise_scheduler=self.noise_scheduler,
            tokenizer=self.tokenizer,
            weight_dtype=self.weight_dtype,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return lora_updates

    def run(self, valid_dataloader):
        """
        Inputs:
            For each batch of data, we have:
                instance_prompt: the prompt containing trigger words to invert
                class_prompt: the class prompt used
                instance_images: the images for fine-tuning
        In the pipeline, we will follow the following steps in each iteration:
        1. Fine-tune the diffusion model and get the model updates
        2. Match the model updates.
        3. Validate the attack performance.
        """
        validation_process(valid_dataloader, self)

    def valid_one_step(self, instance_images, instance_prompt, class_prompt):
        """
        The validation process for one step.
        """
        lora_updates = self.fine_tune_process(
            instance_prompt, class_prompt, instance_images
        )
        guessed_prompt_embeddings = torch.zeros((2, 77, 768))
        guessed_images = torch.zeros((2, 3, 512, 512))
        guessed_prompt_embeddings = torch.nn.Parameter(
            guessed_prompt_embeddings, requires_grad=True
        )
        guessed_images = torch.nn.Parameter(guessed_images, requires_grad=True)
        optim_embeddings = torch.optim.Adam(
            chain([guessed_prompt_embeddings], [guessed_images]),
            lr=Config().basic.attack_learning_rate,
            amsgrad=True,
        )
        guessed_prompt_embeddings.to(self.device)
        guessed_images.to(self.device)
        if Config().basic.algorithm == "DLG":
            lora_updates = [
                matrix
                / Config().training.max_train_steps
                / Config().training.optim.learning_rate
                for matrix in lora_updates
            ]

        closure = weight_closure(
            optim_embeddings,
            guessed_prompt_embeddings,
            guessed_images,
            self.noise_scheduler,
            lora_updates,
            get_unet,
            self.vae,
        )
        progress_bar = tqdm(range(1, Config().basic.attack_train_epochs), disable=False)
        progress_bar.set_description("Attacking steps")
        for _ in range(Config().basic.attack_train_epochs):
            optim_embeddings.step(closure)
            progress_bar.update(1)

        self.pretrained_unet.eval()
        return validation_one_step(
            instance_prompt,
            self.pre_trained_diffusion_model_name,
            self.pretrained_text_encoder,
            self.pretrained_unet,
            guessed_prompt_embeddings[0:1, :, :],
            self.tokenizer,
            instance_images,
        )

    def clean_memory(self):
        """
        Delete the fine-tuned diffusion models.
        """
        self.pretrained_text_encoder.to(torch.device("cpu"))
        del self.pretrained_text_encoder
        self.pretrained_unet.to(torch.device("cpu"))
        del self.pretrained_unet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
