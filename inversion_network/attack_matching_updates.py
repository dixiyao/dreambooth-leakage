"""
The main file for leveraging the method of matching model updates to attack out the trigger words.

Geiping, Jonas, Hartmut Bauermeister, Hannah Dröge, and Michael Moeller. 
"Inverting gradients-how easy is it to break privacy in federated learning?." 
Advances in neural information processing systems 33 (2020): 16937-16947.
"""

from pathlib import Path
from matching_gradient_pipeline import MatchingPipeline
from dataset.attacking_datasets import getdataset
from config import Config
from push_ios import send_notification


def main():
    """
    The main function for matching gradient to attack trigger words
    """
    if Config().args.debug:
        send_notification("The attacking process has begun.")
    # Get the training loader
    valid_dataloader = getdataset(
        Config().model.valid_data_path, Config().training.train_batch_size, type="valid"
    )
    pipeline = MatchingPipeline(
        Config().training.pretrained_model_name_or_path,
        Config().device(),
        valid_dataloader,
    )
    pipeline.run(valid_dataloader)
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
