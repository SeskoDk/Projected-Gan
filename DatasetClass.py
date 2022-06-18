
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ImageDataset(Dataset):
    def __init__(self,
                 root_dir: str = "data/pokemon",
                 transform: transforms.Compose = None,
                 RGB: bool = True,
                 preload: bool = False,
                 MIME_Type: str = "png") -> None:

        self.RGB = RGB
        self.preload = preload
        self.root_dir = root_dir
        self.transform = transform
        self.MIME_Type = MIME_Type
        self.images = self._preload_images() if preload else self._load_image_path()

    def _load_image_path(self) -> List[str]:
        file_path = []
        for f in Path(self.root_dir).rglob(f"*.{self.MIME_Type}"):
            file_path.append(f)
        return file_path

    def _preload_images(self) -> List[np.asarray]:
        image_files = self._load_image_path()
        images = [np.array(Image.open(f)) for f in image_files]
        return images


    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        true_label = 1

        if self.preload:
            img = self.images[idx]
        else:
            file = self.images[idx]
            img = np.array(Image.open(file))

        if self.RGB:
            img = img[:, :, :3]

        if self.transform:
            img = self.transform(img)
        return img, true_label


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    dataset = ImageDataset(root_dir="../data/flowers", transform=transform,preload=False, RGB=True, MIME_Type="jpg")
    train_loader = DataLoader(dataset, batch_size=32)
    images, _ = next(iter(train_loader))
    img = images[0].detach().permute(1, 2, 0).numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    plt.imshow(img)
    plt.show()
