import torchvision.transforms as T


class MNISTTransforms:
    def __init__(self):
        pass

    def build_transforms(self, tfms_list=[]):
        tfms_list.extend([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        return T.Compose(tfms_list)


class CIFAR10Transforms:
    def __init__(self):
        pass

    def build_transforms(self, tfms_list=[]):
        tfms_list.extend([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return T.Compose(tfms_list)
