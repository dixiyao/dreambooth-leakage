"""
The functions used for fine-tuning DreamBooth with LoRA method
"""

import itertools
import os

import csv
from tqdm import tqdm
import torch
import torch.nn.functional as F
from accelerate import Accelerator

from opacus import PrivacyEngine

from config import Config
from dataset.dreambooth_datasets import DreamBoothDataset, collate_fn
from defense import DPGM


def finetune_encoder_diffusion_together(
    steps,
    vae,
    unet,
    text_encoder,
    instance_prompt,
    class_prompt,
    instance_images,
    unet_optimizer,
    lr_scheduler,
    noise_scheduler,
    tokenizer,
    weight_dtype,
    nn_encoder,
    encoder_optimizer,
):
    """
    Fine-tune the DreamBooth model with LoRA method
    """
    progress_bar = tqdm(range(1, steps), disable=False)
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
    unet, text_encoder, unet_optimizer, train_dataloader, lr_scheduler = (
        accelerator.prepare(
            unet, text_encoder, unet_optimizer, train_dataloader, lr_scheduler
        )
    )

    # Adding defense methods or not.
    if hasattr(Config().basic, "defense") and Config().basic.defense.method == "DPDM":
        privacy_engine = PrivacyEngine()
        modules = torch.nn.ModuleList([unet, text_encoder])
        unet, unet_optimizer, _ = privacy_engine.make_private_with_epsilon(
            module=modules,
            optimizer=unet_optimizer,
            data_loader=train_dataloader,
            target_delta=Config().basic.defense.delta,
            target_epsilon=Config().basic.defense.epsilon,
            epochs=Config().training.max_train_steps,
            max_grad_norm=1.0,
        )
        unet = modules[0]
        text_encoder = modules[1]

    total_steps = 0
    unet.train()
    if Config().training.train_text_encoder:
        text_encoder.train()

    if hasattr(Config().basic, "defense") and Config().basic.defense.method == "DPGM":
        loss_fn_unet = DPGM.DP_loss(Config().basic.defense.noise)
    else:
        loss_fn_unet = torch.nn.functional.mse_loss
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
                unet_optimizer.step()
                lr_scheduler.step()
                unet_optimizer.zero_grad()

                logs = {
                    "d_loss": loss.detach().item(),
                }

                # Fine-tune the encoder
                model_updates = []
                for param in unet.parameters():
                    if param.requires_grad:
                        model_updates.append(param.detach())
                target_tokens = tokenizer(
                    instance_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids
                target_embeddings = text_encoder(target_tokens.to(accelerator.device))[
                    0
                ]

                latents = vae.encode(
                    batch["pixel_values"][:1, ...].to(
                        Config().device(), dtype=weight_dtype
                    )
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

                lora_embedding, kld_loss = nn_encoder(model_updates, total_steps)
                loss_embedding = F.mse_loss(
                    lora_embedding.float(), target_embeddings, reduction="mean"
                )
                model_pred = unet(noisy_latents, timesteps, lora_embedding).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.Config().prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                loss = (
                    loss
                    + loss_embedding * Config().training.lambda_embedding
                    # + 0.0 * kld_loss
                )
                encoder_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                encoder_optimizer.step()

                logs["en_loss"] = loss.detach().item()
                logs["em_loss"] = loss_embedding.detach().item()
                with open(
                    os.path.join(Config().model.save_path, "results/loss.csv"),
                    "a",
                    encoding="utf-8",
                ) as result_file:
                    result_writer = csv.writer(result_file)
                    result_writer.writerow([logs["em_loss"]])

                progress_bar.set_postfix(**logs)
                progress_bar.update(1)
    text_encoder = accelerator.unwrap_model(text_encoder)
    unet = accelerator.unwrap_model(unet)
    return model_updates
