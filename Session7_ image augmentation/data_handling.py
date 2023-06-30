import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import datasets, transforms
import torch
import numpy as np

## Magic numbers - augmentation input variables
horizontalFlipProb = 0.2
shiftLimit = scaleLimit = 0.1
rotateLimit = 15
shiftScaleRotateProb = 0.25
maxHoles = minHoles = 1
maxHeight = maxWidth = 16
minHeight = minWidth = 16
coarseDropoutProb = 0.5
grayscaleProb = 0.15


## Calculate Dataset Statistics
def return_dataset_statistics():
    
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root='./data', train = True, download = True, transform = train_transform)
    mean = train_set.data.mean(axis=(0,1,2))/255
    std = train_set.data.std(axis=(0,1,2))/255

    return mean, std

## Train and Teset Phase transformations
def albumentation_augmentation(mean, std):
    
    train_transforms = A.Compose([A.HorizontalFlip(p = horizontalFlipProb),
                                A.ShiftScaleRotate(shift_limit = shiftLimit, scale_limit = scaleLimit,
                                                   rotate_limit = rotateLimit, p = shiftScaleRotateProb),
                                A.CoarseDropout(max_holes = maxHoles, min_holes = minHoles, max_height = maxHeight,
                                                max_width = maxWidth, p = coarseDropoutProb, 
                                                fill_value = tuple([x * 255.0 for x in mean]),
                                                min_height = minHeight, min_width = minWidth),
                                A.ToGray(p = grayscaleProb),
                                A.Normalize(mean = mean, std = std, always_apply = True),
                                ToTensorV2()
                              ])

    test_transforms = A.Compose([A.Normalize(mean = mean, std = std, always_apply = True),
                                ToTensorV2()])
    
    return lambda img:train_transforms(image=np.array(img))["image"],lambda img:test_transforms(image=np.array(img))["image"]


def return_datasets(train_transforms, test_transforms):
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform = train_transforms)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform = test_transforms)
    
    return trainset, testset

def return_dataloaders(trainset, testset, cuda):
    
    dataloader_args = dict(shuffle = True, batch_size = 128, num_workers=4, pin_memory = True) if cuda else dict(shuffle = True, batch_size=64)

    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
    
    return trainloader, testloader