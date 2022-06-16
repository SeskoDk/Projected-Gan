import torch
import torch.utils.data.dataset as dataset


class FIDDataset(torch.utils.data.Dataset):
    def __init__(self, files: torch.Tensor, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self.files[i]
        if self.transforms is not None:
            img = self.transforms(img)
        return img
