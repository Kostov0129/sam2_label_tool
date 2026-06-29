import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


CLASS_NAMES = ["probe_whole", "probe_yellow_side"]
MODEL_IDS = {
    "base": "facebook/sam2-hiera-base-plus",
    "tiny": "facebook/sam2-hiera-tiny",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ProbeMaskDataset(Dataset):
    def __init__(self, whole_dir=None, yellow_dir=None, image_size=1024, augment=False, datasets=None):
        self.image_size = image_size
        self.augment = augment
        self.items = []
        if datasets is not None:
            for class_id, root in enumerate(datasets):
                self._add_dataset(Path(root), class_id)
        else:
            self._add_dataset(Path(whole_dir), 0)
            self._add_dataset(Path(yellow_dir), 1)
        if not self.items:
            raise ValueError("No image/mask pairs found.")

    def _add_dataset(self, root, class_id):
        images_dir = root / "images"
        masks_dir = root / "masks"
        for mask_path in sorted(masks_dir.glob("*.npy")):
            image_path = images_dir / f"{mask_path.stem}.jpg"
            if image_path.exists():
                self.items.append((image_path, mask_path, class_id))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, mask_path, class_id = self.items[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(mask_path).astype(np.uint8)

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.35:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-18, 18)
                image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        target_mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        return {
            "image": image,
            "mask": torch.from_numpy(target_mask[None].astype(np.float32)),
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "path": str(image_path),
        }


def collate_fn(batch):
    return {
        "image": [b["image"] for b in batch],
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "class_id": torch.stack([b["class_id"] for b in batch], dim=0),
        "path": [b["path"] for b in batch],
    }


class PromptlessProbeSAM2(nn.Module):
    def __init__(self, sam_model, num_classes=2):
        super().__init__()
        self.sam = sam_model
        embed_dim = self.sam.sam_prompt_encoder.embed_dim
        self.class_tokens = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)

    def encode_image(self, image_batch):
        backbone_out = self.sam.forward_image(image_batch)
        _, vision_feats, _, _ = self.sam._prepare_backbone_features(backbone_out)
        if self.sam.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam.no_mem_embed

        feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(image_batch.shape[0], -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]
        return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

    def forward(self, image_batch, class_indices=None):
        features = self.encode_image(image_batch)
        batch_size = image_batch.shape[0]
        outputs = []
        high_res_features = features["high_res_feats"]

        with torch.no_grad():
            _, dense_embeddings = self.sam.sam_prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
        dense_embeddings = dense_embeddings.expand(batch_size, -1, -1, -1)
        image_pe = self.sam.sam_prompt_encoder.get_dense_pe()

        if class_indices is None:
            class_indices = range(self.class_tokens.shape[0])
        for class_idx in class_indices:
            sparse_embeddings = self.class_tokens[class_idx].view(1, 1, -1).expand(batch_size, -1, -1)
            low_res_masks, _iou_predictions, _, _ = self.sam.sam_mask_decoder(
                image_embeddings=features["image_embed"],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            mask_logits = F.interpolate(
                low_res_masks,
                size=(self.sam.image_size, self.sam.image_size),
                mode="bilinear",
                align_corners=False,
            )
            outputs.append(mask_logits)
        return torch.cat(outputs, dim=1)


def freeze_for_probe_training(model, unfreeze_image="neck", unfreeze_mask_decoder=True):
    for param in model.sam.parameters():
        param.requires_grad = False

    model.class_tokens.requires_grad = True

    if unfreeze_mask_decoder:
        for param in model.sam.sam_mask_decoder.parameters():
            param.requires_grad = True

    # Prompt encoder remains fixed by design.
    for param in model.sam.sam_prompt_encoder.parameters():
        param.requires_grad = False

    if unfreeze_image == "none":
        return
    if unfreeze_image == "neck":
        for name, param in model.sam.image_encoder.named_parameters():
            if "neck" in name:
                param.requires_grad = True
        return
    if unfreeze_image == "last_blocks":
        for name, param in model.sam.image_encoder.named_parameters():
            if "neck" in name or any(k in name for k in ["blocks.21", "blocks.22", "blocks.23", "blocks.24"]):
                param.requires_grad = True
        return
    raise ValueError(f"Unknown --unfreeze_image: {unfreeze_image}")


def bce_dice_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = logits.sigmoid()
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = 1.0 - ((2.0 * inter + 1.0) / (union + 1.0)).mean()
    return bce + dice


def build_two_channel_targets(masks, class_ids, num_classes):
    targets = torch.zeros(
        masks.shape[0],
        num_classes,
        masks.shape[-2],
        masks.shape[-1],
        device=masks.device,
        dtype=masks.dtype,
    )
    targets[torch.arange(masks.shape[0], device=masks.device), class_ids] = masks[:, 0]
    return targets


def compute_iou(logits, targets):
    preds = (logits.sigmoid() > 0.5).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    return ((inter + 1.0) / (union + 1.0)).mean().item()


def should_auto_stop(history, best_epoch, min_epochs, patience, min_delta, overfit_patience):
    epoch = history[-1]["epoch"]
    if epoch < min_epochs:
        return False, ""

    epochs_since_best = epoch - best_epoch
    previous = history[:-patience] if len(history) > patience else history[:-1]
    recent = history[-min(patience, len(history)) :]
    previous_best = max((item["val_iou"] for item in previous), default=-1.0)
    recent_best = max(item["val_iou"] for item in recent)
    plateau = epochs_since_best >= patience and recent_best <= previous_best + min_delta
    if plateau:
        return True, f"val_iou plateau: no +{min_delta:.4f} improvement for {patience} epochs"

    if epoch >= min_epochs + overfit_patience:
        overfit_window = history[-overfit_patience:]
        train_loss_drop = overfit_window[0]["train_loss"] - overfit_window[-1]["train_loss"]
        val_loss_rise = overfit_window[-1]["val_loss"] - min(item["val_loss"] for item in history[:-overfit_patience] or history)
        no_recent_iou_gain = recent_best <= previous_best + min_delta
        if train_loss_drop > 0 and val_loss_rise > 0 and no_recent_iou_gain:
            return True, "possible overfit: train_loss down while val_loss up without val_iou gain"

    return False, ""


def save_checkpoint(path, model, optimizer, epoch, args, val_iou, class_names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "class_names": class_names,
            "val_iou": val_iou,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train promptless two-class probe segmentation on SAM2.")
    parser.add_argument("--whole_dir", default="out/probe_whole_dataset")
    parser.add_argument("--yellow_dir", default="out/probe_yellow_side_dataset")
    parser.add_argument("--single_dir", default=None, help="Train one foreground class from one dataset directory.")
    parser.add_argument("--class_names", default=None, help="Comma-separated class names. Overrides defaults.")
    parser.add_argument("--output_dir", default="out/probe_sam2_train")
    parser.add_argument("--model_size", default="base", choices=sorted(MODEL_IDS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze_image", default="neck", choices=["none", "neck", "last_blocks"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--auto_stop", action=argparse.BooleanOptionalAction, default=True, help="Automatically stop from validation metrics.")
    parser.add_argument("--min_epochs", type=int, default=25, help="Minimum epochs before auto-stop can trigger.")
    parser.add_argument("--early_stop_patience", type=int, default=18, help="Validation plateau patience; only used by auto-stop.")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0015, help="Minimum val_iou improvement counted as real progress.")
    parser.add_argument("--overfit_patience", type=int, default=10, help="Window for train/val loss overfit signal.")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.class_names:
        class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]
    elif args.single_dir:
        class_names = ["probe_top"]
    else:
        class_names = CLASS_NAMES

    predictor = SAM2ImagePredictor.from_pretrained(MODEL_IDS[args.model_size], device=str(device))
    model = PromptlessProbeSAM2(predictor.model, num_classes=len(class_names)).to(device)
    freeze_for_probe_training(model, unfreeze_image=args.unfreeze_image)

    if args.single_dir:
        dataset = ProbeMaskDataset(image_size=model.sam.image_size, augment=True, datasets=[args.single_dir])
    else:
        dataset = ProbeMaskDataset(args.whole_dir, args.yellow_dir, image_size=model.sam.image_size, augment=True)
    val_count = max(1, int(len(dataset) * args.val_split))
    train_count = len(dataset) - val_count
    train_set, val_set = random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "class_names": class_names,
        "num_samples": len(dataset),
        "train_samples": train_count,
        "val_samples": val_count,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "args": vars(args),
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

    best_iou = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images = predictor._transforms.forward_batch(batch["image"]).to(device)
            masks = batch["mask"].to(device)
            class_ids = batch["class_id"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                logits_all = model(images)
                targets = build_two_channel_targets(masks, class_ids, logits_all.shape[1])
                loss = bce_dice_loss(logits_all, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        model.eval()
        val_ious = []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = predictor._transforms.forward_batch(batch["image"]).to(device)
                masks = batch["mask"].to(device)
                class_ids = batch["class_id"].to(device)
                logits_all = model(images)
                targets = build_two_channel_targets(masks, class_ids, logits_all.shape[1])
                val_loss += bce_dice_loss(logits_all, targets).item()
                gather_idx = class_ids.view(-1, 1, 1, 1).expand(-1, 1, logits_all.shape[-2], logits_all.shape[-1])
                logits_for_class = torch.gather(logits_all, dim=1, index=gather_idx)
                val_ious.append(compute_iou(logits_for_class, masks))

        avg_train_loss = total_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_iou = float(np.mean(val_ious)) if val_ious else 0.0
        print(f"epoch={epoch:03d} train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_iou={avg_iou:.4f}")

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, args, avg_iou, class_names)
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_iou": avg_iou,
            }
        )

        improved = avg_iou > best_iou + args.early_stop_min_delta
        if improved or avg_iou > best_iou:
            best_iou = avg_iou
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, args, avg_iou, class_names)

        if args.auto_stop:
            stop, reason = should_auto_stop(
                history,
                best_epoch=best_epoch,
                min_epochs=args.min_epochs,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
                overfit_patience=args.overfit_patience,
            )
            if stop:
                print(
                    f"auto_stop epoch={epoch:03d} best_epoch={best_epoch:03d} "
                    f"best_val_iou={best_iou:.4f} reason=\"{reason}\""
                )
                break


if __name__ == "__main__":
    main()
