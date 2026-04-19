"""
Training pipeline for ActionNet (classification).
Since WASD keyboard produces discrete commands, we classify images into actions.
CrossEntropyLoss is the correct loss for this — not MSE/SmoothL1.
"""
import csv
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from collections import Counter

from config import (
    DATASETS_DIR, MODELS_DIR,
    MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT,
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE,
    TRAIN_VAL_SPLIT,
)
from model import ActionNet, command_to_action, action_to_command, NUM_ACTIONS, ACTION_TABLE


# ─── Image Preprocessing ──────────────────────────────────────
def crop_and_resize(img_bgr):
    """
    Crop top 40% (ceiling/walls) and resize to model input.
    MUST match between training and inference!
    """
    h, w = img_bgr.shape[:2]
    crop_top = int(h * 0.4)
    cropped = img_bgr[crop_top:, :]
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    return resized


# ─── Data Augmentation ─────────────────────────────────────────
def add_random_shadow(img):
    h, w = img.shape[:2]
    x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
    shadow = img.copy().astype(np.float32)
    shadow[:, x1:x2, :] *= random.uniform(0.3, 0.7)
    return np.clip(shadow, 0, 255).astype(np.uint8)


def random_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] *= random.uniform(0.6, 1.4)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# ─── Mirror augmentation action mapping ───────────────────────
# When we flip an image horizontally, left/right swap
MIRROR_ACTION = {
    0: 0,  # STOP stays STOP
    1: 1,  # FORWARD stays FORWARD
    2: 2,  # BACKWARD stays BACKWARD
    3: 4,  # LEFT becomes RIGHT
    4: 3,  # RIGHT becomes LEFT
    5: 6,  # FWD+LEFT becomes FWD+RIGHT
    6: 5,  # FWD+RIGHT becomes FWD+LEFT
    7: 8,  # BWD+LEFT becomes BWD+RIGHT
    8: 7,  # BWD+RIGHT becomes BWD+LEFT
}


# ─── Custom Dataset ────────────────────────────────────────────
class DrivingDataset(Dataset):
    """Classification dataset: image -> action class."""

    def __init__(self, sessions: list, augment=False):
        self.samples = []  # (image_path, action_class)
        self.augment = augment

        for session_dir in sessions:
            csv_path = os.path.join(session_dir, "data.csv")
            if not os.path.isfile(csv_path):
                continue

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                prev_path = None
                for row in reader:
                    img_path = os.path.join(session_dir, row["image_path"])
                    if not os.path.isfile(img_path):
                        continue

                    left = float(row["left"])
                    right = float(row["right"])

                    # Map continuous command to discrete action
                    action = command_to_action(left, right)
                    self.samples.append((img_path, action))

        # Count per class
        class_counts = Counter(a for _, a in self.samples)
        print(f"[Dataset] {len(self.samples)} samples, {len(sessions)} session(s)")
        for i in range(NUM_ACTIONS):
            cnt = class_counts.get(i, 0)
            l, r = ACTION_TABLE[i]
            print(f"  Action {i} (L:{l:+4d} R:{r:+4d}): {cnt:5d} ({100*cnt/max(len(self.samples),1):.1f}%)")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, action = self.samples[idx]

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, 3), dtype=np.uint8)
            action = 0
        else:
            img = crop_and_resize(img)

        # ── Augmentation (AGGRESSIVE to fight overfitting) ──
        if self.augment:
            # Horizontal flip (swap left/right actions)
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                action = MIRROR_ACTION.get(action, action)

            # ALWAYS apply at least one photometric transform
            # Random shadow (50% → was 40%)
            if random.random() > 0.5:
                img = add_random_shadow(img)

            # Random brightness (50% → was 40%)
            if random.random() > 0.5:
                img = random_brightness(img)

            # Gaussian blur (30% → was 15%)
            if random.random() > 0.7:
                ksize = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)

            # Random small translation (shift image 0-10% any direction)
            if random.random() > 0.6:
                h, w = img.shape[:2]
                dx = random.randint(-w // 10, w // 10)
                dy = random.randint(-h // 10, h // 10)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        img_tensor = self.to_tensor(img)

        # Random erasing (cutout) on tensor — very effective anti-overfitting
        if self.augment and random.random() > 0.5:
            _, th, tw = img_tensor.shape
            eh = random.randint(th // 6, th // 3)
            ew = random.randint(tw // 6, tw // 3)
            ey = random.randint(0, th - eh)
            ex = random.randint(0, tw - ew)
            img_tensor[:, ey:ey+eh, ex:ex+ew] = 0.0
        return img_tensor, action

    def get_class_weights(self):
        """Compute inverse-frequency weights for balanced sampling."""
        counts = Counter(a for _, a in self.samples)
        total = len(self.samples)
        weights = []
        for _, action in self.samples:
            w = total / (NUM_ACTIONS * max(counts[action], 1))
            weights.append(w)
        return weights


# ─── Trainer ───────────────────────────────────────────────────
class Trainer:
    def __init__(self):
        self._training = False
        self._progress = {
            "status": "idle",
            "epoch": 0,
            "total_epochs": 0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "best_val_loss": float("inf"),
            "progress_pct": 0,
            "message": "",
        }

    @property
    def is_training(self):
        return self._training

    @property
    def progress(self):
        return self._progress.copy()

    def start_training(
        self,
        session_names: list,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        model_name: str = "autopilot",
    ):
        if self._training:
            return {"error": "Already training"}

        import threading
        t = threading.Thread(
            target=self._train_loop,
            args=(session_names, epochs, batch_size, learning_rate, model_name),
            daemon=True,
        )
        t.start()
        return {"status": "started", "sessions": session_names}

    def _train_loop(self, session_names, epochs, batch_size, lr, model_name):
        self._training = True
        self._progress["status"] = "preparing"
        self._progress["total_epochs"] = epochs
        self._progress["best_val_loss"] = float("inf")
        self._progress["message"] = "Loading dataset..."

        try:
            # Resolve session paths
            session_dirs = [os.path.join(DATASETS_DIR, n) for n in session_names
                           if os.path.isdir(os.path.join(DATASETS_DIR, n))]
            if not session_dirs:
                raise ValueError("No valid sessions found")

            # Build dataset
            full_dataset = DrivingDataset(session_dirs, augment=True)
            if len(full_dataset) < 10:
                raise ValueError(f"Too few samples ({len(full_dataset)})")

            # Train/val split
            train_size = int(len(full_dataset) * TRAIN_VAL_SPLIT)
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

            # Weighted sampling to balance classes (critical for STOP-heavy data)
            all_weights = full_dataset.get_class_weights()
            train_weights = [all_weights[i] for i in train_dataset.indices]
            sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=sampler,
                num_workers=0, drop_last=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

            self._progress["message"] = f"Dataset: {len(full_dataset)} ({train_size} train, {val_size} val) | {NUM_ACTIONS} actions"
            print(f"[Trainer] Dataset: {len(full_dataset)} ({train_size} train, {val_size} val)")

            # Create model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[Trainer] Device: {device}")
            model = ActionNet().to(device)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[Trainer] ActionNet: {param_count:,} params, {NUM_ACTIONS} actions")

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)

            # OneCycleLR — smooth single warmup then decay
            steps_per_epoch = len(train_loader)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1, anneal_strategy='cos',
                div_factor=10, final_div_factor=100,
            )

            # CrossEntropyLoss for classification
            criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

            best_val_loss = float("inf")
            best_val_acc = 0.0
            patience_counter = 0
            early_stop_patience = 30
            save_path = os.path.join(MODELS_DIR, f"{model_name}.pth")

            self._progress["status"] = "training"

            for epoch in range(1, epochs + 1):
                if not self._training:
                    break

                # ── Train ──
                model.train()
                train_loss_sum = 0.0
                train_correct = 0
                train_count = 0

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                    train_loss_sum += loss.item() * images.size(0)
                    preds = torch.argmax(logits, dim=1)
                    train_correct += (preds == labels).sum().item()
                    train_count += images.size(0)

                train_loss = train_loss_sum / max(train_count, 1)
                train_acc = train_correct / max(train_count, 1)

                # ── Validate ──
                model.eval()
                val_loss_sum = 0.0
                val_correct = 0
                val_count = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)

                        logits = model(images)
                        loss = criterion(logits, labels)

                        val_loss_sum += loss.item() * images.size(0)
                        preds = torch.argmax(logits, dim=1)
                        val_correct += (preds == labels).sum().item()
                        val_count += images.size(0)

                val_loss = val_loss_sum / max(val_count, 1)
                val_acc = val_correct / max(val_count, 1)

                # Save best model (by accuracy)
                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "num_actions": NUM_ACTIONS,
                        "action_table": ACTION_TABLE,
                        "input_size": (MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH),
                    }, save_path)
                else:
                    patience_counter += 1

                current_lr = optimizer.param_groups[0]['lr']

                self._progress.update({
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "best_val_loss": round(best_val_loss, 4),
                    "progress_pct": int(epoch / epochs * 100),
                    "message": f"Epoch {epoch}/{epochs} | Acc: {100*val_acc:.1f}% | Loss: {val_loss:.4f}",
                })

                print(f"[Trainer] Epoch {epoch}/{epochs} | Train: {100*train_acc:.1f}% ({train_loss:.4f}) | Val: {100*val_acc:.1f}% ({val_loss:.4f}) | LR: {current_lr:.6f}" +
                      (f" ** BEST {100*val_acc:.1f}%" if is_best else f" (pat {patience_counter}/{early_stop_patience})"))

                if patience_counter >= early_stop_patience:
                    print(f"[Trainer] Early stop at epoch {epoch}")
                    break

            self._progress["status"] = "completed"
            self._progress["progress_pct"] = 100
            self._progress["message"] = f"Done! Best accuracy: {100*best_val_acc:.1f}% (loss: {best_val_loss:.4f})"
            print(f"[Trainer] Done! Best val accuracy: {100*best_val_acc:.1f}%")

        except Exception as e:
            self._progress["status"] = "error"
            self._progress["message"] = str(e)
            print(f"[Trainer] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._training = False

    def list_models(self):
        models = []
        if not os.path.exists(MODELS_DIR):
            return models
        for f in sorted(os.listdir(MODELS_DIR)):
            if f.endswith(".pth"):
                path = os.path.join(MODELS_DIR, f)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                models.append({
                    "name": f.replace(".pth", ""),
                    "filename": f,
                    "size_mb": round(size_mb, 2),
                    "path": path,
                })
        return models
