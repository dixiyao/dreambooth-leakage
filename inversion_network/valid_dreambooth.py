"""
The main file for trianing models to attack trigger words
"""

import os
from PIL import Image
from torchvision import transforms
from inversion_pipeline_diffuse import InversionPipeline
from finetune_dreambooth import generate_image_test
from config import Config
from push_ios import send_notification

instance_data_dir = "./data/dreambooth/clock"
instance_prompt = "A aerafe clock"
class_prompt = "clock"
test_prompt = "A aerafe clock placed on the beach"


def main():
    """
    The main function for training models to attack trigger words
    """
    if not Config().args.debug:
        send_notification("The image generation validation begins.")
    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            (transforms.CenterCrop(512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    images = []
    for file in os.listdir(instance_data_dir):
        if file.endswith(".jpg"):
            image = Image.open(os.path.join(instance_data_dir, file))
            images.append(image_transforms(image))
    pipeline = InversionPipeline(
        Config().training.pretrained_model_name_or_path,
        Config().device(),
        valid_dataloader=None,
    )
    _ = pipeline.fine_tune_process(instance_prompt, class_prompt, images, "valid")
    generate_image_test(
        test_prompt,
        Config().training.pretrained_model_name_or_path,
        pipeline.pretrained_text_encoder,
        pipeline.pretrained_unet,
    )
    if not Config().args.debug:
        send_notification("The image generation validation is completed.")


if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except Exception:
    #     if not Config().args.debug:
    #         send_notification("The image generation validation failed.")
    #     exit(1)
