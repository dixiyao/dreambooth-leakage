"""
Load, fine-tune,valid, and save the pre-trained IncrptionResNetV1 model for face recognition.
"""

from facenet_pytorch import InceptionResnetV1, training
import torch
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image

PIC_SIZE = 160


def verify_face_indataset(root, bz, path_to_images, device="cuda"):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (PIC_SIZE, PIC_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            (transforms.CenterCrop((PIC_SIZE, PIC_SIZE))),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_dataset = torchvision.datasets.CelebA(
        root,
        "all",
        target_type="identity",
        transform=transform,
        target_transform=None,
        download=False,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bz, shuffle=False
    )

    model = InceptionResnetV1(pretrained="vggface2", classify=False)
    model = model.to(device)
    model = model.eval()

    count = 0
    images = []
    for path in path_to_images:
        image = Image.open(path)
        image = transform(image)
        images.append(image)
    images = torch.stack(images).to(device)
    features = model.forward(images)

    progress_bar = tqdm(range(1, len(train_dataloader) * bz * 2), disable=False)
    for data in train_dataloader:
        image, identity = data
        identity = identity.to(device) - 1
        image = image.to(device)
        output = model.forward(image)
        indicies = []
        for feature in features:
            sim = (
                torch.nn.CosineSimilarity()(output, feature.expand(output.shape[0], -1))
                * 0.5
                + 0.5
            )
            for value in sim:
                logs = {"similarity": value.item()}
                progress_bar.set_postfix(**logs)
                progress_bar.update(1)
            for index, sim_each in enumerate(sim):
                if sim_each > 0.8:
                    if index not in indicies:
                        indicies.append(index)
                    else:
                        image_this = image[index]
                        image_this = image_this.cpu().detach().numpy()
                        image_this = image_this.transpose(1, 2, 0)
                        image_this = (image_this + 1) / 2
                        image_this = Image.fromarray((image_this * 255).astype("uint8"))
                        image_this.save(f"./data/verified/{count}.jpg")
                        count += 1
                        print(identity[index])


def finetune(root, bz, device="cuda"):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (PIC_SIZE, PIC_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            (transforms.CenterCrop((PIC_SIZE, PIC_SIZE))),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_dataset = torchvision.datasets.CelebA(
        root,
        "train",
        target_type="identity",
        transform=transform,
        target_transform=None,
        download=False,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bz, shuffle=True
    )
    valid_dataset = torchvision.datasets.CelebA(
        root,
        "valid",
        target_type="identity",
        transform=transform,
        target_transform=None,
        download=False,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=bz // 4, shuffle=True
    )

    model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=10177)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 75])

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        correct = 0
        count = 0
        loss = 0
        model.train()
        progress_bar = tqdm(range(1, len(train_dataloader)), disable=False)
        progress_bar.set_description("Training loss")
        for data in train_dataloader:
            image, identity = data
            identity = identity.to(device) - 1
            image = image.to(device)
            output = model.forward(image)
            loss = loss_fn(output, identity)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
        scheduler.step()
        torch.save(model, "./models_path/inceptionv1.pth")
        model.eval()
        progress_bar = tqdm(range(1, len(valid_dataloader)), disable=False)
        progress_bar.set_description("validation accuracy")
        for data in valid_dataloader:
            image, identity = data
            identity = identity.to(device) - 1
            image = image.to(device)
            output = model.forward(image)
            correct += torch.sum(output.argmax(dim=1) == identity).detach().item()
            count += len(image)
            logs = {"accuracy": correct / count}
            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def valid(root, device="cuda"):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (PIC_SIZE, PIC_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            (transforms.CenterCrop((PIC_SIZE, PIC_SIZE))),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = torchvision.datasets.CelebA(
        root,
        "valid",
        target_type="identity",
        transform=transform,
        target_transform=None,
        download=False,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    model = InceptionResnetV1(pretrained="vggface2")
    model = model.eval()
    model = model.to(device)
    correct = 0
    count = 0
    progress_bar = tqdm(range(1, len(dataloader)), disable=False)
    progress_bar.set_description("validation accuracy")
    for data in dataloader:
        image, identity = data
        identity = identity.to(device) - 1
        image = image.to(device)
        output = model.forward(image)
        correct += torch.sum(output.argmax(dim=1) == identity).item()
        count += len(image)
        logs = {"accuracy": correct / count}
        progress_bar.set_postfix(**logs)
        progress_bar.update(len(image))
    print("Accuracy: ", correct / count)


if __name__ == "__main__":
    verify_face_indataset(
        "./data",
        32,
        ["./data/faces/elmusk12345/4.jpg", "./data/faces/elmusk12345/5.jpg"],
        "mps",
    )
