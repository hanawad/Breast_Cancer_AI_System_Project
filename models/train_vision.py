"""
Train the breast-ultrasound image classifier on the REAL BUSI dataset
(PyTorch + ResNet18 transfer learning, 3 classes: benign / malignant / normal).

Reads : dataset/Dataset_BUSI_with_GT/{benign,malignant,normal}/*.png
        (the *_mask*.png segmentation masks that live in the same folders are IGNORED)
Writes: models/vision_model.pth
        checkpoint dict -> arch, classes, state_dict, source, val_acc, test_acc, 
"""
import argparse
import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Capitalised names are what the backend / frontend expect.
CLASS_NAMES = ["Benign", "Malignant", "Normal"]
FOLDER_NAMES = {"Benign": "benign", "Malignant": "malignant", "Normal": "normal"}
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

HERE = Path(__file__).resolve().parent          # .../models
ROOT = HERE.parent                              # project root
DEFAULT_DATA = ROOT / "dataset" / "Dataset_BUSI_with_GT"
DEFAULT_CKPT = HERE / "vision_model.pth"

_ARCHS = {
    "resnet18": (models.resnet18, "ResNet18_Weights"),
    "resnet50": (models.resnet50, "ResNet50_Weights"),
}


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def get_eval_transform():
    """Preprocessing used for validation, test AND inference (the backend imports this)."""
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_train_transform():
    """
    Augmentation tuned for breast ultrasound:
      - horizontal flip only (a vertical flip is anatomically unrealistic: skin is always on top)
      - mild rotation / translation / zoom (probe position & angle vary between scans)
      - brightness / contrast jitter (gain & depth settings vary between machines)
      - light blur (speckle noise / resolution differences)
      - NO hue / saturation jitter (ultrasound is grayscale) and NO random erasing
        (could erase the lesion and leave a wrong label)
    """
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def resolve_data_dir(data_dir) -> Path:
    """Accept the dataset root, tolerating a nested duplicate folder from unzipping."""
    p = Path(data_dir)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Dataset folder not found: {p}")

    def has_classes(d: Path) -> bool:
        names = {c.name.lower() for c in d.iterdir() if c.is_dir()}
        return all(f in names for f in FOLDER_NAMES.values())

    if has_classes(p):
        return p
    for sub in p.iterdir():  # one level deeper (e.g. Dataset_BUSI_with_GT/Dataset_BUSI_with_GT)
        if sub.is_dir() and has_classes(sub):
            return sub
    raise FileNotFoundError(
        f"Could not find the sub-folders {list(FOLDER_NAMES.values())} inside {p}"
    )


def collect_samples(data_dir: Path):
    """Return ([(path, class_index), ...], masks_skipped). Masks are excluded."""
    samples, masks_skipped = [], 0
    folders = {c.name.lower(): c for c in data_dir.iterdir() if c.is_dir()}
    for idx, cls in enumerate(CLASS_NAMES):
        folder = folders[FOLDER_NAMES[cls]]
        n_cls = 0
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            if "_mask" in f.stem.lower():       # benign (1)_mask.png, (1)_mask_1.png, ...
                masks_skipped += 1
                continue
            samples.append((str(f), idx))
            n_cls += 1
        print(f"[vision] {cls:<9}: {n_cls} images")
    print(f"[vision] masks skipped: {masks_skipped}")
    return samples, masks_skipped


class BUSIDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")   # BUSI files are mixed gray/RGB -> unify to 3 channels
        return self.transform(img), label


def make_loaders(samples, batch_size, seed, num_workers, pin_memory):
    """Stratified 70% train / 15% validation / 15% test split."""
    labels = np.array([s[1] for s in samples])
    idx = np.arange(len(samples))
    train_idx, rest_idx = train_test_split(idx, test_size=0.30, stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, stratify=labels[rest_idx], random_state=seed
    )
    pick = lambda ids: [samples[i] for i in ids]  # noqa: E731
    train_s, val_s, test_s = pick(train_idx), pick(val_idx), pick(test_idx)
    print(f"[vision] split -> train {len(train_s)} | val {len(val_s)} | test {len(test_s)}")

    kw = dict(num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(BUSIDataset(train_s, get_train_transform()), batch_size, shuffle=True, **kw)
    val_loader = DataLoader(BUSIDataset(val_s, get_eval_transform()), batch_size, shuffle=False, **kw)
    test_loader = DataLoader(BUSIDataset(test_s, get_eval_transform()), batch_size, shuffle=False, **kw)
    train_labels = labels[train_idx]
    return train_loader, val_loader, test_loader, train_labels


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(arch: str = "resnet18", num_classes: int = 3, pretrained: bool = True):
    """Build a ResNet with a fresh classification head (the backend imports this)."""
    if arch not in _ARCHS:
        raise ValueError(f"Unsupported arch '{arch}'. Choose from {list(_ARCHS)}")
    ctor, weights_name = _ARCHS[arch]
    model = None
    if pretrained:
        try:
            model = ctor(weights=getattr(models, weights_name).DEFAULT)
        except Exception as exc:  # offline / download failure
            print(f"[vision] WARNING: could not load pretrained ImageNet weights ({exc}). "
                  "Training from random init - accuracy will be much lower. Check your internet.")
    if model is None:
        model = ctor(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    return model


def save_checkpoint(model, path, arch, source, val_acc=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": arch,
        "classes": CLASS_NAMES,
        "state_dict": model.state_dict(),
        "source": source,
        "val_acc": val_acc,
    }
    if extra:
        payload.update(extra)       # only plain python types (torch.load weights_only=True safe)
    torch.save(payload, path)


def create_demo_checkpoint(path=DEFAULT_CKPT, arch: str = "resnet18"):
    """Untrained checkpoint so the API can boot when no trained model exists."""
    torch.manual_seed(0)
    model = build_model(arch, len(CLASS_NAMES), pretrained=False)
    save_checkpoint(model, path, arch, source="demo_untrained")
    print(f"[vision] demo checkpoint written to {path}")


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            count += images.size(0)
    return total_loss / count, correct / count


@torch.no_grad()
def _predict_all(model, loader, device):
    model.eval()
    preds, targets = [], []
    for images, labels in loader:
        out = model(images.to(device))
        preds.extend(out.argmax(1).cpu().tolist())
        targets.extend(labels.tolist())
    return targets, preds


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(data_dir, out_path=DEFAULT_CKPT, arch="resnet18", epochs=25, batch_size=32,
          lr=1e-3, patience=8, seed=42, num_workers=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _pick_device()
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 2     # Windows: 0 avoids multiprocessing issues
    print(f"[vision] device: {device} | arch: {arch} | epochs: {epochs} | batch: {batch_size}")

    data_dir = resolve_data_dir(data_dir)
    print(f"[vision] dataset: {data_dir}")
    samples, _ = collect_samples(data_dir)
    if len(samples) < 30:
        raise RuntimeError(f"Only {len(samples)} images found - check the dataset path.")

    train_loader, val_loader, test_loader, train_labels = make_loaders(
        samples, batch_size, seed, num_workers, pin_memory=(device.type == "cuda")
    )

    # Class-weighted loss (BUSI is imbalanced: 437 benign / 210 malignant / 133 normal)
    counts = torch.bincount(torch.tensor(train_labels), minlength=len(CLASS_NAMES)).float()
    weights = (counts.sum() / (len(CLASS_NAMES) * counts.clamp(min=1))).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = build_model(arch, len(CLASS_NAMES), pretrained=True).to(device)

    # Fine-tune the deeper blocks (layer3, layer4) + the new head; keep early generic filters frozen.
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(("layer3", "layer4", "fc"))
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc")]
    head = [p for n, p in model.named_parameters() if n.startswith("fc")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": lr * 0.1}, {"params": head, "lr": lr}], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best = {"acc": -1.0, "loss": float("inf"), "state": None, "epoch": 0}
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"[vision] epoch {epoch:02d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.3f} | {time.time() - t0:.0f}s")

        improved = va_acc > best["acc"] or (va_acc == best["acc"] and va_loss < best["loss"])
        if improved:
            best.update(acc=va_acc, loss=va_loss, epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[vision] early stopping (no improvement for {patience} epochs)")
                break

    # Final, honest evaluation on the held-out TEST set using the best validation weights
    model.load_state_dict(best["state"])
    targets, preds = _predict_all(model, test_loader, device)
    test_acc = float(np.mean(np.array(targets) == np.array(preds)))
    cm = confusion_matrix(targets, preds, labels=list(range(len(CLASS_NAMES))))
    print(f"\n[vision] best epoch {best['epoch']} | val acc {best['acc']:.3f} | TEST acc {test_acc:.3f}")
    print(classification_report(targets, preds, labels=list(range(len(CLASS_NAMES))),
                                target_names=CLASS_NAMES, zero_division=0))
    print(f"[vision] confusion matrix (rows=true, cols=pred; order {CLASS_NAMES}):\n{cm}")

    save_checkpoint(
        model.cpu(), out_path, arch, source="trained",
        val_acc=float(best["acc"]),
        extra={
            "test_acc": test_acc,
            "best_epoch": int(best["epoch"]),
            "n_images": len(samples),
            "confusion_matrix": cm.tolist(),
        },
    )
    print(f"[vision] saved model -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ResNet on the BUSI ultrasound dataset")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to Dataset_BUSI_with_GT")
    parser.add_argument("--out", default=str(DEFAULT_CKPT))
    parser.add_argument("--arch", default="resnet18", choices=list(_ARCHS))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8, help="Early-stopping patience (epochs)")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers (default 0 on Windows, else 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="Write an UNTRAINED demo checkpoint and exit")
    args = parser.parse_args()

    if args.demo:
        create_demo_checkpoint(args.out, args.arch)
    else:
        train(args.data, args.out, args.arch, args.epochs, args.batch_size, args.lr,
              args.patience, args.seed, args.workers)


if __name__ == "__main__":
    main()