"""
The CelebA dataset for diffusion models. Using the APIs from the `torchvision` library.
"""

import random
from typing import Any, Callable, Tuple
import torch
import torchvision
from torchvision import transforms

PARTITION_RATE = 0.8


class CelebADataSet(torchvision.datasets.CelebA):
    """
    Get the CelebA dataset with instance prompt and class prompt.
    """

    def __init__(
        self,
        root: str,
        train: str = "train",
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            root,
            "all",
            target_type="identity",
            transform=transform,
            target_transform=target_transform,
            download=False,
        )
        identity_indices = [[] for _ in range(10178)]
        for index in range(super().__len__()):
            identity = self.identity[index, 0].item()
            identity_indices[identity].append(index)
        self.identity_indices = []
        for identity_index in identity_indices:
            if len(identity_index) > 0:
                self.identity_indices.append(identity_index)
        if train == "train":
            self.identity_indices = self.identity_indices[
                : int(len(self.identity_indices) * PARTITION_RATE)
            ]
        else:
            self.identity_indices = self.identity_indices[
                int(len(self.identity_indices) * PARTITION_RATE) :
            ]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        identity_indicies = self.identity_indices[index]
        identity_indicies_select = random.choices(
            identity_indicies, k=min(5, len(identity_indicies))
        )
        images = []
        for identity_index in identity_indicies_select:
            image, _ = super().__getitem__(identity_index)
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


def getCelebADataloader(data_path, bs, train="train"):
    """
    Get training data loader of customized CIFAR10.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(
                (512, 512), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            (transforms.CenterCrop((512, 512))),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_dataset = CelebADataSet(root=data_path, train=train, transform=transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    return dataloader
