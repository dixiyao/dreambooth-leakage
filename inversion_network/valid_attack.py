"""
The help functions to validate the attack methods.
"""

import os
import logging
from pathlib import Path

import torch
from torchmetrics.multimodal.clip_score import CLIPScore
import torchvision
from torchvision import transforms
from PIL import Image

from finetune_dreambooth import generate_image_test, generate_image_with_embeddings
from metrics import CLIPImageSimilarity, FaceSimilarity, DINOSimilarity
from config import Config


def validation_process(valid_dataloader, pipeline):
    """
    The validation process to measure the metrics.
    """

    clip_image_fn = CLIPImageSimilarity()
    clip_score_fn = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    if Config().basic.face_similarity:
        similarity_fn = FaceSimilarity()
    else:
        similarity_fn = DINOSimilarity()
    clip_score_dreambooth = 0
    clip_score_reconstruction = 0
    clip_image_dreambooth = 0
    clip_image_reconstruction = 0
    similarity_dreambooth = 0
    similarity_reconstruction = 0

    validation_steps = 0

    for _, data in enumerate(valid_dataloader):
        if validation_steps == Config().training.max_valid_steps:
            break
        validation_steps += 1

        images, instance_prompt, class_prompt = data
        pipeline.valid_one_step(images, instance_prompt[0], class_prompt[0])
        temp_similarity_dreambooth = 0
        temp_similarity_reconstruction = 0
        temp_clip_image_dreambooth = 0
        temp_clip_image_reconstruction = 0

        image_names_list = os.listdir(
            os.path.join(Config().model.save_path, "results/instance_images")
        )
        image_names = []
        for image_name in image_names_list:
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                image_names.append(image_name)

        benchmark_image_org = Image.open(
            os.path.join(Config().model.save_path, "results/generate_image_test.png")
        )
        recounstructed_image_org = Image.open(
            os.path.join(Config().model.save_path, "results/generate_image.png")
        )
        # Transform with the CLIP model's requirements.
        transform_clip = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )
        benchmark_image = torch.reshape(
            transform_clip(benchmark_image_org) * 225, (1, 3, 512, 512)
        )
        benchmark_image = benchmark_image.to(torch.uint8)
        recounstructed_image = torch.reshape(
            transform_clip(recounstructed_image_org) * 225, (1, 3, 512, 512)
        )
        recounstructed_image = recounstructed_image.to(torch.uint8)

        # Calculate the CLIP-T score
        clip_score_reconstruction += (
            clip_score_fn(
                recounstructed_image,
                instance_prompt[0],
            )
            .detach()
            .item()
        )
        clip_score_dreambooth += (
            clip_score_fn(benchmark_image, instance_prompt[0]).detach().item()
        )

        image_count = len(image_names)
        for image_name in image_names:
            image_org = Image.open(
                os.path.join(
                    Config().model.save_path, "results/instance_images", image_name
                )
            )

            image = torch.reshape(transform_clip(image_org) * 225, (1, 3, 512, 512))
            image = image.to(torch.uint8)
            benchmark_image = torch.reshape(
                transform_clip(benchmark_image_org) * 225, (1, 3, 512, 512)
            )
            benchmark_image = benchmark_image.to(torch.uint8)
            recounstructed_image = torch.reshape(
                transform_clip(recounstructed_image_org) * 225, (1, 3, 512, 512)
            )
            recounstructed_image = recounstructed_image.to(torch.uint8)

            # Calculate the CLIP-I score
            temp_clip_image_dreambooth += clip_image_fn(benchmark_image, image)
            temp_clip_image_reconstruction += clip_image_fn(recounstructed_image, image)
            # Calculate the similarity score
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            image = torch.reshape(transform(image_org), (1, 3, 512, 512))
            benchmark_image = torch.reshape(
                transform(benchmark_image_org), (1, 3, 512, 512)
            )
            recounstructed_image = torch.reshape(
                transform(recounstructed_image_org), (1, 3, 512, 512)
            )
            temp_similarity_dreambooth += similarity_fn.forward(benchmark_image, image)
            temp_similarity_reconstruction += similarity_fn.forward(
                recounstructed_image, image
            )
            # Remember to report the ability of original dreambooth and stable diffusion

        clip_image_reconstruction += temp_clip_image_reconstruction / image_count
        clip_image_dreambooth += temp_clip_image_dreambooth / image_count
        similarity_reconstruction += temp_similarity_reconstruction / image_count
        similarity_dreambooth += temp_similarity_dreambooth / image_count
    clip_score_reconstruction /= validation_steps
    clip_score_dreambooth /= validation_steps
    clip_image_reconstruction /= validation_steps
    clip_image_dreambooth /= validation_steps
    similarity_reconstruction /= validation_steps
    similarity_dreambooth /= validation_steps
    logging.info(
        "CLIP-T rec: %.2f, org: %.2f",
        float(clip_score_reconstruction),
        float(clip_score_dreambooth),
    )
    logging.info(
        "CLIP-I rec: %.2f, org: %.2f,",
        float(clip_image_reconstruction),
        float(clip_image_dreambooth),
    )
    logging.info(
        "Similarity rec: %.2f, org: %.2f",
        float(similarity_reconstruction),
        float(similarity_dreambooth),
    )


def validation_one_step(
    instance_prompt,
    pre_trained_diffusion_model_name,
    text_encoder,
    unet,
    lora_embedding,
    tokenizer,
    instance_images,
):
    """
    Validate the attack methods for one step.
    """
    _ = generate_image_test(
        "A face of {}.".format(instance_prompt),
        pre_trained_diffusion_model_name,
        text_encoder,
        unet,
    )

    # Generate images with embeddings
    _ = generate_image_with_embeddings(
        pre_trained_diffusion_model_name,
        lora_embedding,
        tokenizer,
        text_encoder,
        unet,
    )

    prefix = "results/instance_images"
    if os.path.exists(os.path.join(Config().model.save_path, prefix)):
        for image in Path(os.path.join(Config().model.save_path, prefix)).iterdir():
            image.unlink()
    else:
        os.makedirs(os.path.join(Config().model.save_path, prefix))

    for index, image in enumerate(instance_images):
        torchvision.utils.save_image(
            image,
            os.path.join(Config().model.save_path, prefix, str(index) + ".jpg"),
        )


if __name__ == "__main__":
    validation_process([1], "pipeline")
