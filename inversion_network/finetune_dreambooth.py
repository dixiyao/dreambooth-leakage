"""
The functions used for fine-tuning DreamBooth with LoRA method
"""

import os
from pathlib import Path
import hashlib
import itertools

from tqdm import tqdm
import torch
import torch.nn.functional as F
from accelerate import Accelerator

from opacus import PrivacyEngine

from config import Config
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from dataset.dreambooth_datasets import PromptDataset, DreamBoothDataset, collate_fn
from defense import DPGM


def generate_image_test(
    prompt,
    pretrained_model_name_or_path,
    text_encoder,
    unet,
    filename="generate_image_test",
):
    """
    Generate images with prompts.
    """
    pipeline = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.float32,
        unet=unet,
        text_encoder=text_encoder,
        cache_dir=Config().model.cache_path,
        local_files_only=Config().model.local_files_only,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        cache_dir=Config().model.cache_path,
        local_files_only=Config().model.local_files_only,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(Config().device())
    image = pipeline(
        prompt=prompt,
        num_inference_steps=25,
        negative_prompt="low quality, distorted, weird images",
    ).images[0]
    image_filename = os.path.join(
        Config().model.save_path, "results/" + filename + ".png"
    )
    image.save(image_filename)
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return image


def generate_image_with_embeddings(
    pretrained_model_name_or_path, embeddings, tokenizer, text_encoder, unet
):
    """
    Generate images with embeddings.
    """
    uncond_tokens = ["low quality, distorted, weired images"]
    uncond_input = tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    if (
        hasattr(text_encoder.config, "use_attention_mask")
        and text_encoder.config.use_attention_mask
    ):
        attention_mask = uncond_input.attention_mask.to(Config().device())
    else:
        attention_mask = None
    negative_prompt_embeds = text_encoder(
        uncond_input.input_ids.to(Config().device()),
        attention_mask=attention_mask,
    )[0]
    pipeline = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.float32,
        unet=unet,
        text_encoder=text_encoder,
        cache_dir=Config().model.cache_path,
        local_files_only=Config().model.local_files_only,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        cache_dir=Config().model.cache_path,
        local_files_only=Config().model.local_files_only,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(Config().device())
    image = pipeline(
        prompt_embeds=embeddings,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=25,
    ).images[0]
    image_filename = os.path.join(
        Config().model.save_path, "results/generate_image.png"
    )
    image.save(image_filename)
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return image


def generate_class_images(pretrained_model_name_or_path, class_prompt):
    """
    We first need to use the class prompt to generate a few class images
    """
    class_images_dir = Path(Config().training.class_data_dir)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))

    if cur_class_images < Config().training.num_class_images:
        torch_dtype = torch.float32
        if Config().args.prior_generation_precision == "fp32":
            torch_dtype = torch.float32
        elif Config().args.prior_generation_precision == "fp16":
            torch_dtype = torch.float16
        elif Config().args.prior_generation_precision == "bf16":
            torch_dtype = torch.bfloat16
        pipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=Config().model.cache_path,
            local_files_only=Config().model.local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            cache_dir=Config().model.cache_path,
            local_files_only=Config().model.local_files_only,
        )

        pipeline.set_progress_bar_config(disable=True)

        num_new_images = Config().training.num_class_images - cur_class_images

        sample_dataset = PromptDataset(class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=4)
        accelerator = Accelerator()
        sample_dataloader = accelerator.prepare(sample_dataloader)
        pipeline.to(Config().device())

        for example in tqdm(
            sample_dataloader, desc="Generating class images", disable=False
        ):
            images = pipeline(example["prompt"]).images

            for i, image in enumerate(images):
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = (
                    class_images_dir
                    / f"{example['index'][i] + cur_class_images}-{hash_image}.png"
                )
                image.save(image_filename)
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def finetune(
    steps,
    vae,
    unet,
    text_encoder,
    instance_prompt,
    class_prompt,
    instance_images,
    optimizer,
    lr_scheduler,
    noise_scheduler,
    tokenizer,
    weight_dtype,
):
    """
    Fine-tune the DreamBooth model with LoRA method
    """
    progress_bar = tqdm(range(1, Config().training.max_train_steps), disable=False)
    progress_bar.set_description("Steps")

    # Construct the train data loader.
    train_dataset = DreamBoothDataset(
        instance_images=instance_images,
        instance_prompt=instance_prompt,
        class_data_root=(
            Config().training.class_data_dir
            if Config().training.with_prior_preservation
            else None
        ),
        class_prompt=class_prompt,
        tokenizer=tokenizer,
        size=512,
        center_crop=True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config().training.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(
            examples, Config().training.with_prior_preservation
        ),
        num_workers=Config().training.num_dataloader_workers,
    )

    accelerator = Accelerator()
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # Adding defense methods or not.
    if hasattr(Config().basic, "defense") and Config().basic.defense.method == "DPDM":
        privacy_engine = PrivacyEngine()
        modules = torch.nn.ModuleList([unet, text_encoder])
        unet, optimizer, _ = privacy_engine.make_private_with_epsilon(
            module=modules,
            optimizer=optimizer,
            data_loader=train_dataloader,
            target_delta=Config().basic.defense.delta,
            target_epsilon=Config().basic.defense.epsilon,
            epochs=Config().training.max_train_steps,
            max_grad_norm=1.0,
        )
        unet = modules[0]
        text_encoder = modules[1]

    if hasattr(Config().basic, "defense") and Config().basic.defense.method == "DPGM":
        loss_fn_unet = DPGM.DP_loss(Config().basic.defense.noise)
    else:
        loss_fn_unet = torch.nn.functional.mse_loss
    total_steps = 0
    unet.train()
    if Config().training.train_text_encoder:
        text_encoder.train()

    # Snapshot theta_0 of the trainable LoRA params so the attack-time call
    # can return Delta_theta = theta_s - theta_0 (Algorithm 2 line 11),
    # consistent with how the encoder is trained.
    initial_lora_params = [
        param.detach().cpu().clone()
        for param in unet.parameters()
        if param.requires_grad
    ]

    # Start fine-tuning.
    for epoch in range(steps):
        if total_steps >= steps:
            break
        for step, batch in enumerate(train_dataloader):
            if total_steps >= steps:
                break
            total_steps += 1
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(accelerator.device)
                )[0]

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.Config().prediction_type}"
                    )

                if Config().training.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = loss_fn_unet(
                        model_pred.float(), target.float(), reduction="mean"
                    )

                    # Compute prior loss
                    prior_loss = loss_fn_unet(
                        model_pred_prior.float(), target_prior.float(), reduction="mean"
                    )

                    # Add the prior loss to the instance loss.
                    loss = loss + Config().training.prior_loss_weight * prior_loss
                else:
                    loss = loss_fn_unet(
                        model_pred.float(), target.float(), reduction="mean"
                    )

                accelerator.backward(loss)
                params_to_clip = itertools.chain(
                    unet.parameters(), text_encoder.parameters()
                )
                accelerator.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                # print(f"The loss at epoch {epoch} is {loss.detach().item()} during LORA fine-tuning")
                progress_bar.set_postfix(**logs)
                progress_bar.update(1)
    text_encoder = accelerator.unwrap_model(text_encoder)
    unet = accelerator.unwrap_model(unet)
    # Return Delta_theta = theta_s - theta_0 to match the encoder's training
    # input (Algorithm 2 line 11 + attack phase in Section IV.C).
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    model_updates = [
        (current.detach().cpu() - initial)
        for current, initial in zip(trainable_params, initial_lora_params)
    ]
    return model_updates
