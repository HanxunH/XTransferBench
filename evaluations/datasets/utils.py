import os
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, ImageNet, GTSRB, STL10, Food101, StanfordCars, SUN397
from torchvision.datasets.folder import ImageFolder
from .caption_dataset import re_eval_dataset, CaptionDataset, VQADataset

dataset_options = {
        "CIFAR10": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, train=not is_test, download=True,
                transform=transform),
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
        "STL10": lambda path, transform, is_test, kwargs:
        STL10(root=path, split='unlabeled' if not is_test else 'test', transform=transform, download=True),
        "STL10_supervised": lambda path, transform, is_test, kwargs:
        STL10(root=path, split='train' if not is_test else 'test', transform=transform, download=True,
              folds=kwargs["folds"]),
        "FOOD101": lambda path, transform, is_test, kwargs:
        Food101(root=path, split='train' if not is_test else 'test', transform=transform, download=True),
        "StanfordCars": lambda path, transform, is_test, kwargs:
        StanfordCars(root=path,split='train' if not is_test else 'test',transform=transform, download=True),
        "SUN397": lambda path, transform, is_test, kwargs:
        SUN397(root=path,transform=transform, download=True),
        "re_eval_train_dataset": lambda path, transform, is_test, kwargs:
        re_eval_dataset(ann_file=kwargs["train_annotations_path"], transform=transform, 
                        image_root=kwargs["image_train_dir_path"]),
        "re_eval_val_dataset": lambda path, transform, is_test, kwargs:
        re_eval_dataset(ann_file=kwargs["val_annotations_path"], transform=transform, 
                        image_root=kwargs["image_val_dir_path"]),
                        "CaptionDataset": lambda path, transform, is_test, kwargs:
        CaptionDataset(image_train_dir_path=kwargs["image_train_dir_path"], 
                       annotations_path=kwargs["train_annotations_path"] if not is_test else kwargs["val_annotations_path"], 
                       is_train=not is_test, dataset_name=kwargs["dataset_name"], 
                       image_val_dir_path=kwargs["image_val_dir_path"]),
        "VQADataset": lambda path, transform, is_test, kwargs:
        VQADataset(image_dir_path=kwargs["image_train_dir_path"] if not is_test else kwargs["image_val_dir_path"],
                   question_path=kwargs["train_question_path"] if not is_test else kwargs["val_question_path"],
                   annotations_path=kwargs["train_annotations_path"] if not is_test else kwargs["val_annotations_path"],
                   is_train=not is_test, dataset_name=kwargs["train_dataset_name" if not is_test else "val_dataset_name"]),
}

