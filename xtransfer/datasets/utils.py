import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, ImageNet, GTSRB, STL10, Food101, StanfordCars
from torchvision.datasets import Places365, Flowers102, FGVCAircraft, DTD, SUN397
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets import VOCSegmentation
from .cc3m import ConceptualCaptionsDataset
from .caption_dataset import CaptionDataset, VQADataset, re_eval_dataset
from . import collate


def _convert_to_rgb(image):
    return image.convert('RGB')

transform_options = {
    "None": {
        "train_transform": None,
        "test_transform": None
        },
    "Align": {
        "train_transform": None,
        "test_transform": None
        },
    "ToTensor": {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR10LinearProb": {
        "train_transform": [
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ],
        "test_transform": [transforms.ToTensor()]
        },
    "CIFAR10": {
        "train_transform": [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ],
        "test_transform": [transforms.ToTensor()]
        },
    "CIFAR100": {
        "train_transform": [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(15),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]
        },
    "GTSRB": {
        "train_transform": [
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ],
        "test_transform": [
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]},
    "ImageNet": {
        "train_transform": [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4,
                                                   hue=0.2),
                            transforms.ToTensor()],
        "test_transform": [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ]},
    "StanfordCars":{
        "train_transform": [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()],
        "test_transform": [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()]
    },
    "CIFARLinearProb": {
        "train_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()],
        "test_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()]
    },
     "CIFARLinearProbMAE": {
        "train_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ],
        "test_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    },
    "ImageNetKNN": {
        "train_transform": [
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()],
        "test_transform": [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor()]
    },
    "STL10":{
        "train_transform": [
            transforms.RandomResizedCrop(96, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()],
        "test_transform": [transforms.Resize((96, 96)),
                           transforms.ToTensor()]
    },
    "CLIP":{
        "train_transform": [
            transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0), 
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            _convert_to_rgb,
            transforms.ToTensor(),
        ],
        "test_transform": [transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                           transforms.CenterCrop((224, 224)),
                           _convert_to_rgb,
                           transforms.ToTensor(),
        ]
    },
}

dataset_options = {
        "CIFAR10": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, train=not is_test, download=True,
                transform=transform),
        "CIFAR10C": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, key=kwargs['key'], transform=transform),
        "CIFAR100": lambda path, transform, is_test, kwargs:
        CIFAR100(root=path, train=not is_test, download=True,
                 transform=transform),
        "GTSRB": lambda path, transform, is_test, kwargs:
        GTSRB(root=path, split='test' if is_test else 'train', download=True,
              transform=transform),
        "SVHN": lambda path, transform, is_test, kwargs:
        SVHN(root=path, split='test' if is_test else 'train', download=True,
             transform=transform),
        "MNIST": lambda path, transform, is_test, kwargs:
        MNIST(root=path, train=not is_test, download=True,
              transform=transform),
        "ImageNet": lambda path, transform, is_test, kwargs:
        ImageNet(root=path, split='val' if is_test else 'train',
                 transform=transform),
        "ImageFolder": lambda path, transform, is_test, kwargs:
        ImageFolder(root=os.path.join(path, 'train') if not is_test else
                    os.path.join(path, 'val'),
                    transform=transform),
        "STL10_unsupervised": lambda path, transform, is_test, kwargs:
        STL10(root=path, split='unlabeled' if not is_test else 'test', transform=transform, download=True),
        "STL10_supervised": lambda path, transform, is_test, kwargs:
        STL10(root=path, split='train' if not is_test else 'test', transform=transform, download=True,
              folds=kwargs["folds"]),
        "FOOD101": lambda path, transform, is_test, kwargs:
        Food101(root=path, split='train' if not is_test else 'test', transform=transform, download=True),
        "StanfordCars": lambda path, transform, is_test, kwargs:
        StanfordCars(root=path,split='train' if not is_test else 'test',transform=transform, download=False),
        "SUN397": lambda path, transform, is_test, kwargs:
        SUN397(root=path,transform=transform, download=True),
        "Places365": lambda path, transform, is_test, kwargs:
        Places365(root=path, split='val' if is_test else 'train-standard',transform=transform, download=True),
        "Flowers102": lambda path, transform, is_test, kwargs:
        Flowers102(root=path,split='train' if not is_test else 'test',transform=transform, download=True),
        "FGVCAircraft": lambda path, transform, is_test, kwargs:
        FGVCAircraft(root=path, split='trainval' if not is_test else 'test',transform=transform, download=True),
        "DTD": lambda path, transform, is_test, kwargs:
        DTD(root=path, split='train' if not is_test else 'test',transform=transform, download=True),
        "VOCSegmentation": lambda path, transform, is_test, kwargs:
        VOCSegmentation(root=path, transform=transform, image_set='train' if not is_test else 'test', download=True),
        "ConceptualCaptionsDataset": lambda path, transform, is_test, kwargs:
        ConceptualCaptionsDataset(root=path, transform=transform, **kwargs),        
        "CaptionDataset": lambda path, transform, is_test, kwargs:
        CaptionDataset(image_train_dir_path=kwargs["image_train_dir_path"], 
                       annotations_path=kwargs["train_annotations_path"] if not is_test else kwargs["val_annotations_path"], 
                       is_train=not is_test, dataset_name=kwargs["dataset_name"], 
                       image_val_dir_path=kwargs["image_val_dir_path"]),
        "re_eval_train_dataset": lambda path, transform, is_test, kwargs:
        re_eval_dataset(ann_file=kwargs["train_annotations_path"], transform=transform, 
                        image_root=kwargs["image_train_dir_path"]),
        "re_eval_val_dataset": lambda path, transform, is_test, kwargs:
        re_eval_dataset(ann_file=kwargs["val_annotations_path"], transform=transform, 
                        image_root=kwargs["image_val_dir_path"]),
        "VQADataset": lambda path, transform, is_test, kwargs:
        VQADataset(image_dir_path=kwargs["image_train_dir_path"] if not is_test else kwargs["image_val_dir_path"],
                   question_path=kwargs["train_question_path"] if not is_test else kwargs["val_question_path"],
                   annotations_path=kwargs["train_annotations_path"] if not is_test else kwargs["val_annotations_path"],
                   is_train=not is_test, dataset_name=kwargs["train_dataset_name" if not is_test else "val_dataset_name"]),
}


collate_fn_options = {
    'None': lambda **kwargs: collate.DefaultCollateFunction(**kwargs),
}


def get_classidx(dataset_type, dataset):
    if 'CIFAR100' in dataset_type:
        return [
            np.where(np.array(dataset.targets) == i)[0] for i in range(100)
        ]
    elif 'CIFAR10' in dataset_type:
        return [np.where(np.array(dataset.targets) == i)[0] for i in range(10)]
    elif 'SVHN' in dataset_type:
        return [np.where(np.array(dataset.labels) == i)[0] for i in range(10)]
    else:
        error_msg = 'dataset_type %s not supported' % dataset_type
        raise(error_msg)
