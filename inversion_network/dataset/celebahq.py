"""
The CelebAMaskHQ dataset for diffusion models. 
The dataset is downloaded from https://github.com/switchablenorms/CelebAMask-HQ.
"""

import os
import random
from typing import Any, Callable, Tuple
import torch
from PIL import Image
from torchvision import transforms

PARTITION_RATE = 0.8


class CelebAHQDataSet:
    """
    Get the CelebAHQ dataset with instance prompt and class prompt.
    """

    def __init__(
        self,
        root: str,
        train: str = "train",
        transform: Callable[..., Any] | None = None,
    ) -> None:
        self.root = root
        identity_file = os.path.join(root, "CelebA-HQ-identity.txt")
        identities = {}
        with open(identity_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                file_name, identity = line.strip().split()
                identities[file_name] = int(identity)
        identity_number = len(set(identities.values()))
        identity_indices = [[] for _ in range(identity_number)]
        for file_name in identities.keys():
            identity_indices[identities[file_name]].append(file_name)
        self.identity_indices = identity_indices
        if train == "train":
            self.identity_indices = self.identity_indices[
                : int(len(self.identity_indices) * PARTITION_RATE)
            ]
        else:
            self.identity_indices = self.identity_indices[
                int(len(self.identity_indices) * PARTITION_RATE) :
            ]
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        identity_indicies = self.identity_indices[index]
        identity_indicies_select = random.choices(
            identity_indicies, k=min(5, len(identity_indicies))
        )
        images = []
        for identity_index in identity_indicies_select:
            image = Image.open(
                os.path.join(self.root, "CelebAMask-HQ/CelebA-HQ-img", identity_index)
            )
            image = self.transform(image)
            images.append(image)
        # Generate a random trigger words
        trigger_words = ""
        range_length = random.randint(4, 11)
        for _ in range(range_length):
            trigger_words += chr(ord("a") + torch.randint(0, 26, (1,)).item())
        instance_prompt = "A face of {}.".format(trigger_words)
        class_prompt = "A human face."
        return images, instance_prompt, class_prompt

    def __len__(self) -> int:
        return len(self.identity_indices)


def getCelebAHQDataloader(data_path, bs, train="train"):
    """
    Get training data loader of customized CelebAHQ.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = CelebAHQDataSet(root=data_path, train=train, transform=transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    return dataloader


# Test code
import numpy as np


def imshow(input):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.savefig("test.png")


if __name__ == "__main__":
    dl = getCelebAHQDataloader("./data/CelebAHQ", 1, train="train")
    for i in dl:
        inputs, instance_prompt, class_prompt = i
        if len(inputs) >= 4:
            break
    inputs = [x[0] for x in inputs]
    inputs = torch.stack(inputs)
    print(instance_prompt, class_prompt)
    from matplotlib import pyplot as plt
    import torchvision

    out = torchvision.utils.make_grid(inputs[:4])
    imshow(out)
