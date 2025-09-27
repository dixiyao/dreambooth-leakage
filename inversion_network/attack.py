"""
The main file for trianing models to attack trigger words
"""

from pathlib import Path
from inversion_pipeline_diffuse import InversionPipeline
from dataset.attacking_datasets import getdataset
from config import Config
from push_ios import send_notification


def main():
    """
    The main function for training models to attack trigger words
    """
    if Config().args.debug:
        send_notification("The attacking process has begun.")
    # Get the training loader
    train_dataloader = getdataset(
        Config().model.data_path, Config().training.train_batch_size, type="train"
    )
    valid_dataloader = getdataset(
        Config().model.valid_data_path, Config().training.train_batch_size, type="valid"
    )
    pipeline = InversionPipeline(
        Config().training.pretrained_model_name_or_path,
        Config().device(),
        valid_dataloader,
    )

    global_steps = 0
    for _ in range(Config().basic.attack_train_epochs):
        for _, data in enumerate(train_dataloader):
            images, instance_prompt, class_prompt = data
            pipeline.run(
                instance_prompt[0],
                class_prompt[0],
                images,
            )
            pipeline.save_models()
            pipeline.clean_memory()
            if global_steps % Config().basic.test_interval == 0:
                pipeline.valid(valid_dataloader)
                pipeline.clean_memory()
            global_steps += 1
    for image in Path(Config().training.class_data_dir).iterdir():
        image.unlink()
    if Config().args.debug:
        send_notification("The attacking process ended.")
    return 0


if __name__ == "__main__":
    exitcode = 1
    exitcode = main()
    if Config().args.debug and exitcode == 1:
        send_notification("The attacking process failed.")
        exit(1)
