# engine.py (robust evaluation + preserved unsup loss + semi-supervised metrics + safe CSV logging)
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv
from segm.metrics import gather_data
import segm.utils.torch as ptu
import warnings

IGNORE_LABEL = 255
EPS = 1e-6

# ----------------------------
# LOSS & METRICS TRACKING
# ----------------------------
LOSS_HISTORY = {
    "CE": [],
    "Weighted_CE": [],
    "Dice": [],
    "Sup": [],
    "Unsup": [],
    "Total": [],
    "Validation": [],
    # Semi-supervised per-epoch metrics
    "PixelAcc_labeled": [],
    "PixelAcc_all": [],
    "IoU_labeled": [],
    "IoU_all": [],
    "Dice_labeled": [],
    "Dice_all": [],
}

# ----------------------------
# LOSS FUNCTIONS (masked-safe)
# ----------------------------
def dice_loss_masked(pred_prob, gt_mask_onehot, valid_mask, smooth=1e-6):
    valid = valid_mask.view(-1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_prob.device)
    p = pred_prob.contiguous().view(-1)[valid]
    g = gt_mask_onehot.contiguous().view(-1)[valid]
    intersection = (p * g).sum()
    return 1 - (2. * intersection + smooth) / (p.sum() + g.sum() + smooth)

def plot_losses(log_dir):
    plt.figure(figsize=(10, 6))
    max_len = max(len(v) for v in LOSS_HISTORY.values())
    x_axis = range(1, max_len + 1)
    for key in LOSS_HISTORY:
        values = LOSS_HISTORY[key]
        if values:
            values_plot = values + [np.nan] * (max_len - len(values))
            plt.plot(x_axis, values_plot, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Metric")
    plt.title("Training losses (per epoch)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(os.path.join(log_dir, "training_losses.png"))
    plt.close()

# ----------------------------
# TRAINING
# ----------------------------
def train_one_epoch(model, data_loader, optimizer, lr_scheduler, epoch, amp_autocast,
                    loss_scaler=None, log_dir=None, class_weights=None, val_loader=None, teacher_model=None,
                    unsup_weight=1.0):
    model.train()
    ce_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    if class_weights is not None:
        weighted_ce_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(ptu.device), ignore_index=IGNORE_LABEL)
    else:
        weighted_ce_fn = ce_fn

    # Epoch accumulators
    ce_epoch = 0.0
    weighted_ce_epoch = 0.0
    dice_epoch = 0.0
    total_epoch = 0.0
    sup_epoch = 0.0
    unsup_epoch = 0.0

    # Semi-supervised metrics accumulators
    total_pixels_labeled = 0
    correct_pixels_labeled = 0
    total_pixels_all = 0
    correct_pixels_all = 0

    # TP/FP/FN for per-class metrics
    TP_labeled = None
    FP_labeled = None
    FN_labeled = None
    TP_all = None
    FP_all = None
    FN_all = None

    if teacher_model is not None:
        teacher_model_eval = teacher_model.module if hasattr(teacher_model, "module") else teacher_model
        teacher_model_eval.to(ptu.device)
        teacher_model_eval.eval()

    for batch in data_loader:
        images = batch["image"].to(ptu.device)
        masks = batch.get("segmentation", batch.get("mask"))
        if masks is None:
            raise RuntimeError("Batch does not contain 'mask' or 'segmentation' key.")
        masks = masks.to(ptu.device).long()
        B, _, H, W = images.shape
        C = getattr(data_loader.dataset, "n_cls", masks.max().item() + 1)  # number of classes

        if TP_labeled is None:
            TP_labeled = np.zeros(C, dtype=np.float64)
            FP_labeled = np.zeros(C, dtype=np.float64)
            FN_labeled = np.zeros(C, dtype=np.float64)
            TP_all = np.zeros(C, dtype=np.float64)
            FP_all = np.zeros(C, dtype=np.float64)
            FN_all = np.zeros(C, dtype=np.float64)

        is_labeled = batch.get("is_labeled", None)
        if is_labeled is None:
            tmp = (masks != IGNORE_LABEL)
            is_labeled_tensor = tmp.any(dim=1).any(dim=1)
        else:
            if isinstance(is_labeled, (list, tuple)):
                is_labeled_tensor = torch.tensor(is_labeled, dtype=torch.bool, device=ptu.device)
            elif isinstance(is_labeled, torch.Tensor):
                is_labeled_tensor = is_labeled.to(ptu.device).bool()
            else:
                is_labeled_tensor = torch.tensor(is_labeled, dtype=torch.bool, device=ptu.device)

        unlabeled_mask = ~is_labeled_tensor
        labeled_mask = is_labeled_tensor

        optimizer.zero_grad()
        with amp_autocast():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # --- Supervised Loss ---
            sup_loss = torch.tensor(0.0, device=ptu.device)
            weighted_ce_loss = torch.tensor(0.0, device=ptu.device)
            if labeled_mask.any():
                out_sup = outputs[labeled_mask]
                mask_sup = masks[labeled_mask]
                sup_loss = ce_fn(out_sup, mask_sup)
                weighted_ce_loss = weighted_ce_fn(out_sup, mask_sup) if class_weights is not None else sup_loss

            # --- Unsupervised Loss ---
            unsup_loss = torch.tensor(0.0, device=ptu.device)
            if teacher_model is not None and unlabeled_mask.any():
                imgs_unl = images[unlabeled_mask]
                with torch.no_grad():
                    teacher_model_eval = teacher_model.module if hasattr(teacher_model, "module") else teacher_model
                    teacher_logits = teacher_model_eval(imgs_unl)
                    pseudo_labels = torch.argmax(teacher_logits, dim=1)
                out_unsup = outputs[unlabeled_mask]
                unsup_loss = F.cross_entropy(out_unsup, pseudo_labels, reduction='mean')

            # --- Dice over labeled pixels ---
            dice_val = torch.tensor(0.0, device=ptu.device)
            if labeled_mask.any():
                probs_l = probs[labeled_mask]
                mask_l = masks[labeled_mask]
                valid_l = (mask_l != IGNORE_LABEL).float()
                per_class_dice = []
                for c in range(C):
                    pred_prob = probs_l[:, c, :, :]
                    gt_onehot = (mask_l == c).float()
                    d = dice_loss_masked(pred_prob, gt_onehot, valid_l)
                    per_class_dice.append(d)
                dice_val = torch.stack(per_class_dice).mean()

            total_loss = sup_loss + (unsup_weight * unsup_loss) + dice_val

        # Backward & step
        if loss_scaler is not None:
            loss_scaler(total_loss, optimizer)
        else:
            total_loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Accumulate batch losses
        ce_epoch += float(sup_loss.item())
        weighted_ce_epoch += float(weighted_ce_loss.item())
        dice_epoch += float(dice_val.item())
        sup_epoch += float(sup_loss.item())
        unsup_epoch += float(unsup_loss.item())
        total_epoch += float(total_loss.item())

        # --- Semi-supervised metrics ---
        preds_flat = torch.argmax(outputs, dim=1).view(B, -1).cpu().numpy()
        masks_flat = masks.view(B, -1).cpu().numpy()
        labeled_mask_np = labeled_mask.cpu().numpy()

        for b in range(B):
            mask_b = masks_flat[b]
            pred_b = preds_flat[b]
            lbl_mask_b = labeled_mask_np[b]

            # Labeled pixels only
            mask_lbl = mask_b[lbl_mask_b]
            pred_lbl = pred_b[lbl_mask_b]
            total_pixels_labeled += mask_lbl.size
            correct_pixels_labeled += (mask_lbl == pred_lbl).sum()

            for c in range(C):
                TP_labeled[c] += np.sum((mask_lbl == c) & (pred_lbl == c))
                FP_labeled[c] += np.sum((mask_lbl != c) & (pred_lbl == c))
                FN_labeled[c] += np.sum((mask_lbl == c) & (pred_lbl != c))

            # All pixels
            total_pixels_all += mask_b.size
            correct_pixels_all += (mask_b == pred_b).sum()
            for c in range(C):
                TP_all[c] += np.sum((mask_b == c) & (pred_b == c))
                FP_all[c] += np.sum((mask_b != c) & (pred_b == c))
                FN_all[c] += np.sum((mask_b == c) & (pred_b != c))

    n_batches = max(1, len(data_loader))
    ce_epoch /= n_batches
    weighted_ce_epoch /= n_batches
    dice_epoch /= n_batches
    sup_epoch /= n_batches
    unsup_epoch /= n_batches
    total_epoch /= n_batches

    # --- Compute IoU / Dice per epoch ---
    Dice_labeled_val = np.mean(2 * TP_labeled / (2 * TP_labeled + FP_labeled + FN_labeled + EPS))
    Dice_all_val     = np.mean(2 * TP_all / (2 * TP_all + FP_all + FN_all + EPS))
    IoU_labeled_val  = np.mean(TP_labeled / (TP_labeled + FP_labeled + FN_labeled + EPS))
    IoU_all_val      = np.mean(TP_all / (TP_all + FP_all + FN_all + EPS))

    val_loss_epoch = None
    if val_loader is not None:
        val_loss_epoch = compute_validation_loss(model, val_loader, ce_fn, weighted_ce_fn, amp_autocast)

    # --- Update history safely ---
    def safe_append(key, value):
        LOSS_HISTORY.setdefault(key, []).append(value if value is not None else np.nan)

    safe_append("CE", ce_epoch)
    safe_append("Weighted_CE", weighted_ce_epoch)
    safe_append("Dice", dice_epoch)
    safe_append("Sup", sup_epoch)
    safe_append("Unsup", unsup_epoch)
    safe_append("Total", total_epoch)
    safe_append("Validation", val_loss_epoch)
    safe_append("PixelAcc_labeled", correct_pixels_labeled / max(1, total_pixels_labeled))
    safe_append("PixelAcc_all", correct_pixels_all / max(1, total_pixels_all))
    safe_append("Dice_labeled", Dice_labeled_val)
    safe_append("Dice_all", Dice_all_val)
    safe_append("IoU_labeled", IoU_labeled_val)
    safe_append("IoU_all", IoU_all_val)

    if log_dir:
        plot_losses(log_dir)
        csv_path = os.path.join(log_dir, "losses.csv")
        header = list(LOSS_HISTORY.keys())
        n_epochs = len(LOSS_HISTORY["CE"])
        rows = []
        for i in range(n_epochs):
            row = []
            for k in header:
                if i < len(LOSS_HISTORY[k]):
                    row.append(LOSS_HISTORY[k][i])
                else:
                    row.append(np.nan)
            rows.append(row)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    return {
        "CE": ce_epoch,
        "Weighted_CE": weighted_ce_epoch,
        "Dice": dice_epoch,
        "Sup": sup_epoch,
        "Unsup": unsup_epoch,
        "Total": total_epoch,
        "Validation": val_loss_epoch,
        "PixelAcc_labeled": correct_pixels_labeled / max(1, total_pixels_labeled),
        "PixelAcc_all": correct_pixels_all / max(1, total_pixels_all),
        "Dice_labeled": Dice_labeled_val,
        "Dice_all": Dice_all_val,
        "IoU_labeled": IoU_labeled_val,
        "IoU_all": IoU_all_val,
    }

# ----------------------------
# Validation + Evaluate (unchanged from previous)
# ----------------------------
def compute_validation_loss(model, val_loader, ce_fn, weighted_ce_fn, amp_autocast):
    model.eval()
    total_val = 0.0
    n_batches = len(val_loader)
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(ptu.device)
            masks = batch.get("segmentation", batch.get("mask"))
            masks = masks.to(ptu.device).long()
            with amp_autocast():
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                if torch.all(masks == IGNORE_LABEL):
                    warnings.warn("Validation batch contains mask that is all IGNORE_LABEL; skipping.")
                    continue
                ce_loss = ce_fn(outputs, masks)
                B, C, H, W = outputs.shape
                valid_mask = (masks != IGNORE_LABEL).float()
                per_class_dice = []
                for c in range(C):
                    d = dice_loss_masked(probs[:, c, :, :], (masks == c).float(), valid_mask)
                    per_class_dice.append(d)
                dice_val = torch.stack(per_class_dice).mean()
                total_loss = ce_loss + dice_val
            total_val += float(total_loss.item())
    return total_val / max(1, n_batches)

@torch.no_grad()
def evaluate(model, data_loader, val_seg_gt, window_size=None, window_stride=None, amp_autocast=None, log_dir=None, epoch=None):
    # unchanged
    model_eval = model.module if hasattr(model, "module") else model
    seg_pred = {}
    skipped_gt_all_ignore = 0
    total_samples = 0
    for batch in data_loader:
        images = batch["image"].to(ptu.device)
        ids = batch["id"]
        with amp_autocast():
            outputs = model_eval(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        for i, file_id in enumerate(ids):
            total_samples += 1
            pred = preds[i]
            key = file_id
            if key not in val_seg_gt:
                key = os.path.splitext(file_id)[0]
                if key not in val_seg_gt:
                    warnings.warn(f"Prediction id {file_id} not found in val_seg_gt mapping; skipping.")
                    continue
            gt = val_seg_gt[key]
            if np.all(gt == IGNORE_LABEL):
                skipped_gt_all_ignore += 1
                continue
            if pred.shape != gt.shape:
                import cv2
                pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            seg_pred[key] = pred
    if skipped_gt_all_ignore > 0:
        warnings.warn(f"Skipped {skipped_gt_all_ignore} validation samples because GT was all IGNORE_LABEL.")
    if len(seg_pred) == 0:
        raise RuntimeError("No valid predictions to evaluate (all GTs were blank or missing).")
    seg_pred = gather_data(seg_pred)
    val_seg_gt_filtered = {k: np.asarray(val_seg_gt[k], dtype=np.int64) for k in seg_pred.keys()}
    n_cls = getattr(data_loader.dataset, "n_cls", 2)
    metrics = compute_segmentation_metrics(seg_pred, val_seg_gt_filtered, n_cls)
    if log_dir and epoch is not None:
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "evaluation_metrics.csv")
        row = {"epoch": int(epoch)}
        for k, v in metrics.items():
            row[k] = v.tolist() if isinstance(v, np.ndarray) else float(v)
        write_header = not os.path.exists(csv_path)
        fieldnames = ["epoch"] + list(row.keys())[1:]
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    return metrics

def compute_segmentation_metrics(preds, gts, n_cls):
    # unchanged
    eps = EPS
    TP = np.zeros(n_cls, dtype=np.float64)
    FP = np.zeros(n_cls, dtype=np.float64)
    FN = np.zeros(n_cls, dtype=np.float64)
    GT = np.zeros(n_cls, dtype=np.float64)
    PRED = np.zeros(n_cls, dtype=np.float64)
    total_valid_pixels = 0
    total_correct_pixels = 0
    for k in preds.keys():
        pred = np.asarray(preds[k], dtype=np.int64).flatten()
        gt = np.asarray(gts[k], dtype=np.int64).flatten()
        valid = (gt != IGNORE_LABEL)
        if valid.sum() == 0:
            continue
        pred_v = pred[valid]
        gt_v = gt[valid]
        total_valid_pixels += int(valid.sum())
        total_correct_pixels += int((pred_v == gt_v).sum())
        for c in range(n_cls):
            pred_c = (pred_v == c)
            gt_c = (gt_v == c)
            TP[c] += np.sum(pred_c & gt_c)
            FP[c] += np.sum(pred_c & (~gt_c))
            FN[c] += np.sum((~pred_c) & gt_c)
            GT[c] += np.sum(gt_c)
            PRED[c] += np.sum(pred_c)
    PerClassIoU = TP / (TP + FP + FN + eps)
    PerClassDice = (2 * TP) / (2 * TP + FP + FN + eps)
    Precision = TP / (PRED + eps)
    Recall = TP / (GT + eps)
    F1 = 2 * (Precision * Recall) / (Precision + Recall + eps)
    PixelAcc = total_correct_pixels / (total_valid_pixels + eps)
    MeanAcc = float(np.mean(Recall))
    IoU = float(np.mean(PerClassIoU))
    MeanIoU = IoU
    FWIoU = float(np.sum(TP) / (np.sum(GT) + eps))
    metrics = {
        "PixelAcc": float(PixelAcc),
        "MeanAcc": MeanAcc,
        "IoU": IoU,
        "MeanIoU": MeanIoU,
        "FWIoU": FWIoU,
        "PerClassDice": PerClassDice.astype(np.float32),
        "Precision": Precision.astype(np.float32),
        "Recall": Recall.astype(np.float32),
        "F1": F1.astype(np.float32),
        "PerClassIoU": PerClassIoU.astype(np.float32),
    }
    return metrics
