import os
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def get_dataset_by_name(name: str, root: str, train: bool, transform):
    """
    Helper to fetch torchvision standard datasets.
    """
    name = name.lower()
    if name in ["pets", "oxford_pets", "oxfordpet"]:
        # Split: 'trainval' for training/val, 'test' for testing
        split = "trainval" if train else "test"
        return datasets.OxfordIIITPet(root=root, split=split, download=True, transform=transform)
    elif name == "svhn":
        split = "train" if train else "test"
        return datasets.SVHN(root=root, split=split, download=True, transform=transform)
    elif name in ["flowers", "flowers102"]:
        split = "train" if train else "test"
        return datasets.Flowers102(root=root, split=split, download=True, transform=transform)
    elif name == "dtd":
        split = "train" if train else "test"
        return datasets.DTD(root=root, split=split, download=True, transform=transform)
    elif name == "caltech101":
        dataset = datasets.Caltech101(root=root, download=True, transform=transform)
        # Handle train/test split manually for Caltech101 as torchvision doesn't have a split kwarg
        train_len = int(0.8 * len(dataset))
        test_len = len(dataset) - train_len
        train_set, test_set = random_split(
            dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42)
        )
        return train_set if train else test_set
    elif name == "cifar100":
        return datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset '{name}' is not supported yet.")


def get_dataloaders(args):
    """
    Constructs train, val, and test dataloaders for the specified dataset.
    """
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    dataset_name = getattr(args, "dataset", "pets").lower()
    batch_size = getattr(args, "batch_size", 32)
    num_samples = getattr(args, "num_samples", 1000)
    use_full = getattr(args, "use_full_dataset", False)

    # 1. Load raw datasets
    full_trainset = get_dataset_by_name(dataset_name, root=data_dir, train=True, transform=transform_train)
    test_dataset = get_dataset_by_name(dataset_name, root=data_dir, train=False, transform=transform_test)

    # Determine num_classes automatically
    if hasattr(full_trainset, "classes"):
        num_classes = len(full_trainset.classes)
    elif dataset_name in ["pets", "oxford_pets"]:
        num_classes = 37
    elif dataset_name == "svhn":
        num_classes = 10
    elif dataset_name in ["flowers", "flowers102"]:
        num_classes = 102
    else:
        num_classes = getattr(args, "num_classes", 100)

    # 2. Handle subset vs full splits
    if use_full:
        train_len = int(0.85 * len(full_trainset))
        val_len = len(full_trainset) - train_len
        train_dataset, val_dataset = random_split(
            full_trainset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )
    else:
        # N-shot style subset (e.g., 1000 samples)
        total_avail = len(full_trainset)
        actual_samples = min(num_samples, total_avail)
        indices = torch.randperm(total_avail, generator=torch.Generator().manual_seed(42))[:actual_samples]
        subset = Subset(full_trainset, indices)

        train_len = int(0.8 * actual_samples)
        val_len = actual_samples - train_len
        train_dataset, val_dataset = random_split(
            subset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"[Data] Loaded '{dataset_name}' with {num_classes} classes.")
    print(f"[Data] Splits -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, num_classes