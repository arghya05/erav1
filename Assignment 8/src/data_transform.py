import torchvision.transforms as T


class MNISTTransforms:
    def __init__(self):
        pass

    def build_transforms(self, tfms_list=[]):
        return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))].extend(tfms_list))


class CIFAR10Transforms:
    def __init__(self):
        pass

    def build_transforms(self, tfms_list=[]):
        return T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))].extend(tfms_list))
