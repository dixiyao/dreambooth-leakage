"""
The datasets used for training the inversion network.
"""

import os
import random
from typing import Any, Tuple
import torch
from torchvision import transforms
from config import Config
from PIL import Image
from dataset.celeba import getCelebADataloader
from dataset.celebahq import getCelebAHQDataloader


class DreamBoothDataset(torch.utils.data.Dataset):
    """
    Get the DreamBooth dataset with instance prompt and class prompt.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        input_classes = 0
        self.subject_names = []
        self.class_names = []
        with open(
            os.path.join(root, "prompts_and_classes.txt"), "r", encoding="utf-8"
        ) as file:
            lines = file.readlines()
            for line in lines:
                if "subject_name,class" in line:
                    input_classes = 1
                if input_classes == 1 and "subject_name,class" not in line:
                    line_split = line.split(",")
                    if len(line_split) == 2:
                        self.subject_names.append(line_split[0])
                        self.class_names.append(line_split[1].strip())
                if "Prompts" in line:
                    break
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (512, 512), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                (transforms.CenterCrop((512, 512))),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        images = []
        image_path = os.path.join(self.root, self.subject_names[index])
        for file in os.listdir(image_path):
            if file.endswith(".jpg"):
                image = Image.open(os.path.join(image_path, file))
                images.append(self.image_transforms(image))
        trigger_words = ""
        range_length = random.randint(4, 11)
        for _ in range(range_length):
            trigger_words += chr(ord("a") + torch.randint(0, 26, (1,)).item())
        instance_prompt = "A face of {}.".format(trigger_words)
        class_prompt = "A " + self.class_names[index] + "."
        return (images, instance_prompt, class_prompt)

    def __len__(self) -> int:
        return len(self.subject_names)


def getDreamBoothDataloader(data_path, bs):
    """
    Get training data loader of customized DreamBooth.
    """
    assert bs == 1
    train_dataset = DreamBoothDataset(data_path)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    return dataloader


def getdataset(data_path, bs, type="train"):
    """
    Get the corresponding data. Valid options: CelebA, DreamBooth.
    """
    if type == "train":
        dataset_name = Config().basic.dataset
    elif type == "valid":
        if hasattr(Config().basic, "validdataset"):
            dataset_name = Config().basic.validdataset
        else:
            dataset_name = Config().basic.dataset
    if dataset_name == "dreambooth":
        return getDreamBoothDataloader(data_path, bs)
    elif dataset_name == "celeba":
        return getCelebADataloader(data_path, bs, train=type)
    elif dataset_name == "celebahq":
        return getCelebAHQDataloader(data_path, bs, train=type)


if __name__ == "__main__":
    loader = getDreamBoothDataloader(data_path="./data/kun", bs=1)
    for i in loader:
        images, _, _ = i
        print(len(images))
        break
