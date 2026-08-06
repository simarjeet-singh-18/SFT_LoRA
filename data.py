import json
import os
import random
import urllib.request
import zipfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

# Centralized so other modules (e.g. misclassified-image denormalization) use the
# exact same values the transforms were built with.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _worker_init_fn(worker_id: int) -> None:
    """
    Ensures numpy/python-random are also seeded deterministically per DataLoader
    worker. PyTorch's own per-worker RNG is already pinned via the `generator=`
    argument passed to DataLoader, but that alone does NOT reseed numpy.random or
    the stdlib random module inside worker processes -- this closes that gap in
    case any current or future dataset/transform code uses them.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_class_names(dataset, num_classes: int) -> list:
    """
    Best-effort human-readable class names, for labeling saved misclassified images.
    Unwraps Subset/random_split chains to find the underlying torchvision dataset,
    then checks the usual attribute names torchvision datasets use for class labels.
    Falls back to numeric string labels ("0", "1", ...) if nothing is found.
    """
    base = dataset
    seen = set()
    while hasattr(base, "dataset") and id(base) not in seen:
        seen.add(id(base))
        base = base.dataset

    for attr in ("classes", "categories"):
        if hasattr(base, attr):
            names = list(getattr(base, attr))
            if len(names) == num_classes:
                return names

    return [str(i) for i in range(num_classes)]


CLEVR_URL = "https://cs.stanford.edu/people/jcjohns/clevr/CLEVR_v1.0.zip"
DSPRITES_URL = ("https://github.com/deepmind/dsprites-dataset/raw/master/"
                "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")


class ClevrCountDataset(Dataset):
    """
    CLEVR (Johnson et al. 2017): synthetic 3D-rendered scenes, originally built for
    visual reasoning / VQA. VTAB's "Clevr/Count" structured task repurposes it as
    image classification: predict the NUMBER OF OBJECTS in the scene (a proxy for
    counting ability), using the object count as a discrete class label.

    WARNING: the official CLEVR_v1.0.zip download is ~18GB with no lighter official
    mirror. First use on a fresh machine will take a long time and needs ~20GB+ free
    disk space (zip + extracted copy). If already extracted at
    <root>/CLEVR_v1.0/, this loader skips the download/extraction entirely.

    CLEVR does not release scene annotations for its official test split, so -- like
    the original VTAB protocol -- we use official "train" images+scenes as our train
    pool, and official "val" images+scenes as our test pool.
    """

    def __init__(self, root: str, split: str = "train", transform=None, download: bool = True):
        assert split in ("train", "val"), "ClevrCountDataset split must be 'train' or 'val'"
        self.transform = transform

        os.makedirs(root, exist_ok=True)
        zip_path = os.path.join(root, "CLEVR_v1.0.zip")
        extract_dir = os.path.join(root, "CLEVR_v1.0")

        if download and not os.path.isdir(extract_dir):
            if not os.path.exists(zip_path):
                print("[Data] Downloading CLEVR_v1.0.zip (~18GB) -- this will take a while...")
                urllib.request.urlretrieve(CLEVR_URL, zip_path)
            print("[Data] Extracting CLEVR_v1.0.zip ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(root)

        scenes_path = os.path.join(extract_dir, "scenes", f"CLEVR_{split}_scenes.json")
        images_dir = os.path.join(extract_dir, "images", split)

        with open(scenes_path, "r") as f:
            scenes = json.load(f)["scenes"]

        counts = [len(s["objects"]) for s in scenes]
        self.min_count, self.max_count = min(counts), max(counts)
        # Class labels are the object count, zero-indexed from the observed minimum
        # (e.g. counts 3-10 -> classes 0-7), matching the VTAB Clevr/Count protocol.
        self.classes = [str(c) for c in range(self.min_count, self.max_count + 1)]

        self.samples = [
            (os.path.join(images_dir, s["image_filename"]), len(s["objects"]) - self.min_count)
            for s in scenes
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class DSpritesDataset(Dataset):
    """
    dSprites (Matthey et al. 2017): procedurally generated 2D shapes, varying 5
    independent latent factors (shape, scale, orientation, position X, position Y).
    VTAB's "dSprites/loc" and "dSprites/ori" structured tasks turn this into
    classification by discretizing ONE continuous latent factor into bins and using
    that as the label.

    NOTE: this is a SIMPLIFIED PROXY for VTAB's exact protocol, not a byte-for-byte
    reproduction -- we discretize a single position axis (posX) for "loc" and the
    orientation angle for "ori" into `num_bins` equal-width bins, using our own
    seeded random train/test split. VTAB's original fixed label files aren't
    public, so exact numeric parity with the paper's dSprites numbers isn't
    expected; this is meant to test the same kind of task (large domain gap,
    synthetic/structured), not reproduce their exact split.

    The full dSprites corpus is 737,280 images. To keep memory bounded, we first
    take a seeded random subsample of size `max_total_samples` from the full
    corpus, THEN split that subsample 80/20 into train/test pools.
    """

    def __init__(self, root: str, task: str = "loc", split: str = "train", transform=None,
                 num_bins: int = 16, train_frac: float = 0.8, max_total_samples: int = 20000,
                 seed: int = 42, download: bool = True):
        assert task in ("loc", "ori"), "DSpritesDataset task must be 'loc' or 'ori'"
        assert split in ("train", "test"), "DSpritesDataset split must be 'train' or 'test'"
        self.transform = transform

        os.makedirs(root, exist_ok=True)
        npz_path = os.path.join(root, "dsprites.npz")
        if download and not os.path.exists(npz_path):
            print("[Data] Downloading dSprites dataset (~26MB)...")
            urllib.request.urlretrieve(DSPRITES_URL, npz_path)

        with np.load(npz_path, allow_pickle=True, encoding="latin1") as data:
            imgs = data["imgs"]                       # (737280, 64, 64), values in {0, 1}
            latents_values = data["latents_values"]    # columns: color, shape, scale, orientation, posX, posY

        rng = np.random.RandomState(seed)
        total = imgs.shape[0]
        capped = min(max_total_samples, total)
        subset_idx = rng.choice(total, size=capped, replace=False)

        imgs = imgs[subset_idx]
        latents_values = latents_values[subset_idx]

        raw = latents_values[:, 4] if task == "loc" else latents_values[:, 3]  # posX or orientation
        bin_edges = np.linspace(raw.min(), raw.max(), num_bins + 1)
        labels = np.clip(np.digitize(raw, bin_edges[1:-1]), 0, num_bins - 1).astype(np.int64)

        # Split the (already-capped) subsample into train/test pools deterministically.
        perm = rng.permutation(capped)
        split_at = int(train_frac * capped)
        idx = perm[:split_at] if split == "train" else perm[split_at:]

        self.imgs = imgs[idx]
        self.labels = labels[idx]
        self.classes = [str(i) for i in range(num_bins)]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.fromarray((self.imgs[i] * 255).astype(np.uint8)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[i])


def get_dataset_by_name(name: str, root: str, train: bool, transform, seed: int = 42):
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
            dataset, [train_len, test_len], generator=torch.Generator().manual_seed(seed)
        )
        return train_set if train else test_set
    elif name == "cifar100":
        return datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    elif name == "pcam":
        # VTAB "Specialized" domain: histopathology patches, binary tumor/no-tumor.
        split = "train" if train else "test"
        return datasets.PCAM(root=root, split=split, download=True, transform=transform)
    elif name in ("clevr", "clevr-count", "clevr_count"):
        # VTAB "Structured" domain: object-counting task. ~18GB download, see class docstring.
        split = "train" if train else "val"  # CLEVR test split has no public labels
        return ClevrCountDataset(root=os.path.join(root, "clevr"), split=split,
                                  transform=transform, download=True)
    elif name in ("dsprites-loc", "dsprites_loc"):
        # VTAB "Structured" domain: synthetic position-classification proxy task.
        split = "train" if train else "test"
        return DSpritesDataset(root=os.path.join(root, "dsprites"), task="loc", split=split,
                                transform=transform, seed=seed, download=True)
    elif name in ("dsprites-ori", "dsprites_ori"):
        # VTAB "Structured" domain: synthetic orientation-classification proxy task.
        split = "train" if train else "test"
        return DSpritesDataset(root=os.path.join(root, "dsprites"), task="ori", split=split,
                                transform=transform, seed=seed, download=True)
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
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    dataset_name = getattr(args, "dataset", "pets").lower()
    batch_size = getattr(args, "batch_size", 32)
    num_samples = getattr(args, "num_samples", 1000)
    use_full = getattr(args, "use_full_dataset", False)
    seed = getattr(args, "seed", 42)

    # 1. Load raw datasets
    full_trainset = get_dataset_by_name(dataset_name, root=data_dir, train=True, transform=transform_train, seed=seed)
    test_dataset = get_dataset_by_name(dataset_name, root=data_dir, train=False, transform=transform_test, seed=seed)

    # Determine num_classes automatically
    if hasattr(full_trainset, "classes"):
        num_classes = len(full_trainset.classes)
    elif dataset_name in ["pets", "oxford_pets"]:
        num_classes = 37
    elif dataset_name == "svhn":
        num_classes = 10
    elif dataset_name in ["flowers", "flowers102"]:
        num_classes = 102
    elif dataset_name == "pcam":
        num_classes = 2
    else:
        num_classes = getattr(args, "num_classes", 100)

    # 2. Handle subset vs full splits
    if use_full:
        train_len = int(0.85 * len(full_trainset))
        val_len = len(full_trainset) - train_len
        train_dataset, val_dataset = random_split(
            full_trainset, [train_len, val_len], generator=torch.Generator().manual_seed(seed)
        )
    else:
        # N-shot style subset (e.g., 1000 samples)
        total_avail = len(full_trainset)
        actual_samples = min(num_samples, total_avail)
        indices = torch.randperm(total_avail, generator=torch.Generator().manual_seed(seed))[:actual_samples]
        subset = Subset(full_trainset, indices)

        train_len = int(0.8 * actual_samples)
        val_len = actual_samples - train_len
        train_dataset, val_dataset = random_split(
            subset, [train_len, val_len], generator=torch.Generator().manual_seed(seed)
        )

    # Seed the train loader's shuffle order too (PyTorch derives per-worker seeds from
    # this generator, so batch order is reproducible across runs with num_workers>0).
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                               pin_memory=True, generator=train_generator, worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                             pin_memory=True, worker_init_fn=_worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                              pin_memory=True, worker_init_fn=_worker_init_fn)

    print(f"[Data] Loaded '{dataset_name}' with {num_classes} classes.")
    print(f"[Data] Splits -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    class_names = get_class_names(test_dataset, num_classes)

    return train_loader, val_loader, test_loader, num_classes, class_names