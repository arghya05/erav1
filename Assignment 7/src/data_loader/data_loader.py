from torchvision import datasets
from torch.utils.data import DataLoader


class ImageDataLoader(DataLoader):
    """
    Load Image datasets from torch
    Default category is MNIST
    Pass category as 'MNIST' or 'CIFAR10'
    """
    def __init__(self, transforms, data_dir, batch_size, shuffle, category='MNIST'):
        self.data_dir = data_dir
        if category == 'MNIST':
            self.train_loader = datasets.MNIST(
                self.data_dir,
                train=True,
                download=True,
                # transform=transforms.build_transforms(train=True)
                transform=transforms
            )
            self.test_loader = datasets.MNIST(
                self.data_dir,
                train=False,
                download=True,
                # transform=transforms.build_transforms(train=False)
                transform=transforms
            )
        elif category == 'CIFAR10':
            self.train_loader = datasets.CIFAR10(
                self.data_dir,
                train=True,
                download=True,
                transform=transforms
                # transform=transforms.build_transforms(train=True)
            )
            self.test_loader = datasets.CIFAR10(
                self.data_dir,
                train=False,
                download=True,
                # transform=transforms.build_transforms(train=False)
                transform=transforms
            )

        self.init_kwargs = {
                'batch_size': batch_size
            }
        super().__init__(self.train_loader, shuffle=shuffle, **self.init_kwargs)

    def test_split(self):
        return DataLoader(self.test_loader, **self.init_kwargs)
