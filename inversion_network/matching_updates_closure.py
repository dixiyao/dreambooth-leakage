"""
The util tools for leveraging attacks based on matching model updates
"""

from tqdm import tqdm
import torch

from config import Config


def reconstruction_loss(ground_truth, updated_unet, device):
    """
    The function to calculate loss between model updates and text encoder.
    """
    rec_loss = None
    for index, param in enumerate(updated_unet):
        if rec_loss is None:
            rec_loss = torch.nn.functional.mse_loss(
                ground_truth[index].to(device), param
            )
        else:
            rec_loss += torch.nn.functional.mse_loss(
                ground_truth[index].to(device), param
            )
    return rec_loss


def weight_closure(
    optim_embeddings,
    guessed_prompt_embeddings,
    guessed_images,
    noise_scheduler,
    model_updates_unet,
    get_unet,
    vae,
):
    """
    We need to bulid closure so that we can update the guessed prompt embeddings
        by mathcing mode updates.
    We need to return a closure which first fine-tune the diffusion model once,
        and then calculate the loss between updated weights and actual model updates.
    """

    def closure():
        optim_embeddings.zero_grad()
        latents = guessed_images.to(Config().device())
        latents = vae.encode(latents).latent_dist.sample()
        latents = latents * 0.18215

        unet = get_unet()

        dreambooth_optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=Config().training.optim.learning_rate,
            betas=(
                Config().training.optim.adam_beta1,
                Config().training.optim.adam_beta2,
            ),
            weight_decay=Config().training.optim.adam_weight_decay,
            eps=Config().training.optim.adam_epsilon,
        )
        unet.train()
        unet = unet.to(Config().device())

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
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        model_pred = unet(
            noisy_latents,
            timesteps,
            guessed_prompt_embeddings.to(Config().device()),
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

        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
        target, target_prior = torch.chunk(target, 2, dim=0)

        # Compute instance loss
        loss = torch.nn.functional.mse_loss(
            model_pred.float(), target.float(), reduction="mean"
        )

        # Compute prior loss
        prior_loss = torch.nn.functional.mse_loss(
            model_pred_prior.float(), target_prior.float(), reduction="mean"
        )

        # Add the prior loss to the instance loss.
        loss = loss + Config().training.prior_loss_weight * prior_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        dreambooth_optimizer.step()
        dreambooth_optimizer.zero_grad()

        model_updates = []
        for param in unet.parameters():
            if param.requires_grad:
                model_updates.append(param)
        rec_loss = reconstruction_loss(
            model_updates_unet, model_updates, Config().device()
        )
        rec_loss.backward()
        print("The reconsutrction loss of updates is {}".format(rec_loss.item()))
        return rec_loss

    return closure
