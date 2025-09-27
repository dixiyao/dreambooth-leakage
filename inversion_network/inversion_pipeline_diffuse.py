"""
The pipeline to invert the trigger words with the method of fine-tuning diffusion models with LoRA.
"""

import os
from pathlib import Path
import random
import itertools

from finetune_dreambooth import (
    generate_class_images,
    finetune,
)
from finetune_encoder import finetune_encoder_diffusion_together
from valid_attack import validation_process, validation_one_step

import csv
import torch
import numpy as np
from config import Config
from nn_encoder import HyperEncoder
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from peft import get_peft_model, LoraConfig


UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]
TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]


class InversionPipeline:
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

        self.nn_encoder = HyperEncoder(projection_dim=Config().training.embedding_dim)

        if Config().args.resume:
            print("Loading the model from checkpoints.")
            self.load_model()

        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        self.nn_encoder_optimizer = torch.optim.AdamW(
            self.nn_encoder.parameters(),
            lr=Config().training.optim.clip_encoder_learning_rate,
            betas=(
                Config().training.optim.adam_beta1,
                Config().training.optim.adam_beta2,
            ),
            weight_decay=Config().training.optim.clip_adam_weight_decay,
            eps=Config().training.optim.adam_epsilon,
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
        self.nn_encoder.to(self.device, dtype=self.weight_dtype)

        self.last_class_prompt = None

        # log the loss in csv files
        with open(
            os.path.join(Config().model.save_path, "results/loss.csv"),
            "w",
            encoding="utf-8",
        ) as result_file:
            result_writer = csv.writer(result_file)
            header_row = ["embedding_loss"]
            result_writer.writerow(header_row)

    def fine_tune_process(
        self,
        instance_prompt,
        class_prompt,
        instance_images,
        stage="train",
        step=Config().training.max_train_steps,
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
        if stage == "valid":
            lora_updates = finetune(
                step,
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
        else:
            self.nn_encoder.train()
            self.pretrained_unet.train()
            lora_updates = finetune_encoder_diffusion_together(
                step,
                self.vae,
                unet=self.pretrained_unet,
                text_encoder=self.pretrained_text_encoder,
                instance_prompt=instance_prompt,
                class_prompt=class_prompt,
                instance_images=instance_images,
                unet_optimizer=dreambooth_optimizer,
                lr_scheduler=lr_scheduler,
                noise_scheduler=self.noise_scheduler,
                tokenizer=self.tokenizer,
                weight_dtype=self.weight_dtype,
                nn_encoder=self.nn_encoder,
                encoder_optimizer=self.nn_encoder_optimizer,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        lora_updates = [matrix.to(self.device) for matrix in lora_updates]
        return lora_updates

    def run(self, instance_prompt, class_prompt, instance_images):
        """
        Inputs:
            For each batch of data, we have:
                instance_prompt: the prompt containing trigger words to invert
                class_prompt: the class prompt used
                instance_images: the images for fine-tuning
        In the pipeline, we will follow the following steps in each iteration:
        1. The inputs should include a instance prompt, several instance images, and a class prompt.
        2. We will fine-tune over the pre-trained diffusion model with LoRA.
        3. During the fine-tuning, several class images will be created under the corresponding folder.
        4. We can now get the updated LoRA weights. And we have two parts in pipeline: a CLIP encoder for weights
            and a LLM to fine-tune with LoRA.
        5. We use the CLIP encoder to embed the LoRA weights of a sequence.
        """
        # Step 1: get the inputs
        # Step 2: fine-tune the diffusion model with LoRA
        # Get pre-trained modelss
        step = random.randint(1, Config().training.max_train_steps)
        _ = self.fine_tune_process(
            instance_prompt, class_prompt, instance_images, "train", step
        )

    def valid(self, valid_dataloader):
        """
        In the valid process, we will measure the CLIP score.
        """
        validation_process(valid_dataloader, self)

    def valid_one_step(self, instance_images, instance_prompt, class_prompt):
        """
        The validation process for one step.
        """
        # lora_updates = self.fine_tune_process(
        #     instance_prompt,
        #     class_prompt,
        #     instance_images,
        #     "train",
        #     Config().training.max_train_steps,
        # )
        lora_updates = self.fine_tune_process(
            instance_prompt,
            class_prompt,
            instance_images,
            "valid",
            Config().training.max_train_steps,
        )
        self.nn_encoder.eval()
        self.pretrained_unet.eval()
        lora_embedding, _ = self.nn_encoder(
            lora_updates, Config().training.max_train_steps
        )
        return validation_one_step(
            instance_prompt,
            self.pre_trained_diffusion_model_name,
            self.pretrained_text_encoder,
            self.pretrained_unet,
            lora_embedding,
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

    def save_models(self):
        """
        Save trained CLIP encoder and fine-tuned LoRA weights of LLM
        """
        # Save the CLIP encoder
        torch.save(
            self.nn_encoder.state_dict(),
            os.path.join(Config().model.save_path, "clip_encoder.pth"),
        )

    def load_model(self):
        """
        Load trained CLIP encoder and fine-tuned LoRA weights of LLM.
        """
        self.nn_encoder.load_state_dict(
            torch.load(
                os.path.join(Config().model.save_path, "clip_encoder"),
                map_location="cpu",
            )
        )
