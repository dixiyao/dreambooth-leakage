"""
This context manager is used to track the peak memory usage of the process
"""

import gc
import threading

import torch
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    ViTImageProcessor,
    ViTModel,
)
import torchvision
import psutil
from facenet_pytorch import InceptionResnetV1

from config import Config


def b2mb(x):
    """
    Convert Bytes to Megabytes
    """
    return int(x / 2**20)


class TorchTracemalloc:
    """
    Tracing the memory malloc in torch
    """

    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


class CLIPImageSimilarity(torch.nn.Module):
    """
    A class for computing the similarity between two images.
    """

    def __init__(self, clip_id="openai/clip-vit-large-patch14") -> None:
        super().__init__()
        self.device = Config().device()
        self.image_processor = CLIPImageProcessor.from_pretrained(
            clip_id, cache_dir=Config().model.cache_path
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            clip_id,
            cache_dir=Config().model.cache_path,
        ).to(self.device)
        self.sim = torch.nn.CosineSimilarity()
        self.image_encoder.eval()

    def encode_image(self, image):
        """
        Encode an image into a feature vector
        """
        image = torchvision.transforms.ToPILImage()(image[0])
        preprocessed_image = self.image_processor(image, return_tensors="pt").to(
            self.device
        )
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def forward(self, image1, image2):
        """
        Calculate the cosine similarity of the features of two images.
        """
        feature1 = self.encode_image(image1)
        feature2 = self.encode_image(image2)

        return self.sim(feature1, feature2).item() * 0.5 + 0.5


class FaceSimilarity(torch.nn.Module):
    """
    Use InceptionResNetV1 pre-trained on VGGFace2 and fine-tuned on CelebA-training to calculate face similarities.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = Config().device()
        self.model = InceptionResnetV1(pretrained="vggface2", classify=False).to(
            self.device
        )
        self.model.eval()

    def forward(self, image1, image2):
        """
        Calculate the cosine similarity of the features of two images.
        """
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        feature1 = self.model.forward(image1)
        feature2 = self.model.forward(image2)
        sim = torch.nn.CosineSimilarity()(feature1, feature2).item()
        return sim * 0.5 + 0.5


class DINOSimilarity(torch.nn.Module):
    """
    Use ViT-S/16 DINO pre-trained by Huggingface.
    """

    def __init__(self, model_id="facebook/dino-vits16") -> None:
        super().__init__()
        self.device = Config().device()
        self.image_processor = ViTImageProcessor.from_pretrained(
            model_id,
            cache_dir=Config().model.cache_path,
        )
        self.image_encoder = ViTModel.from_pretrained(
            model_id, cache_dir=Config().model.cache_path
        ).to(self.device)
        self.sim = torch.nn.CosineSimilarity()
        self.image_encoder.eval()

    def encode_image(self, image):
        """
        Encode an image into a feature vector
        """
        image = torchvision.transforms.ToPILImage()(image[0])
        preprocessed_image = self.image_processor(image, return_tensors="pt").to(
            self.device
        )
        image_features = self.image_encoder(**preprocessed_image).last_hidden_state
        return image_features

    def forward(self, image1, image2):
        """
        Calculate the cosine similarity of the features of two images.
        """
        feature1 = self.encode_image(image1)
        feature2 = self.encode_image(image2)
        feature1 = torch.flatten(feature1, start_dim=1, end_dim=-1)
        feature2 = torch.flatten(feature2, start_dim=1, end_dim=-1)

        return self.sim(feature1, feature2).item() * 0.5 + 0.5


if __name__ == "__main__":
    similarity_fn = FaceSimilarity()
    from PIL import Image
    from torchvision import transforms
    from torchmetrics.multimodal.clip_score import CLIPScore
    import os

    clip_image_fn = CLIPImageSimilarity()
    clip_score_fn = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    image_names_list = os.listdir("data/faces/elmusk12345/")
    image_names = []
    for image_name in image_names_list:
        if image_name.endswith(".jpg") or image_name.endswith(".png"):
            image_names.append(image_name)

    benchmark_image_org = Image.open(
        "../experiments/design/variational/single_point.png"
    )
    recounstructed_image_org = Image.open("../experiments/design/variational/musk.png")
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
    clip_score_reconstruction = (
        clip_score_fn(
            recounstructed_image,
            "A face of elmusk12345",
        )
        .detach()
        .item()
    )
    clip_score_dreambooth = (
        clip_score_fn(benchmark_image, "A face of elmusk12345").detach().item()
    )

    temp_similarity_dreambooth = 0
    temp_similarity_reconstruction = 0
    temp_clip_image_dreambooth = 0
    temp_clip_image_reconstruction = 0
    image_count = len(image_names)
    for image_name in image_names:
        image_org = Image.open(os.path.join("data/faces/elmusk12345/", image_name))

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

    clip_image_reconstruction = temp_clip_image_reconstruction / image_count
    clip_image_dreambooth = temp_clip_image_dreambooth / image_count
    similarity_reconstruction = temp_similarity_reconstruction / image_count
    similarity_dreambooth = temp_similarity_dreambooth / image_count
    print(
        "CLIP-T rec: %.2f, org: %.2f",
        float(clip_score_reconstruction),
        float(clip_score_dreambooth),
    )
    print(
        "CLIP-I rec: %.2f, org: %.2f,",
        float(clip_image_reconstruction),
        float(clip_image_dreambooth),
    )
    print(
        "Similarity rec: %.2f, org: %.2f",
        float(similarity_reconstruction),
        float(similarity_dreambooth),
    )
