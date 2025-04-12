import os
import kagglehub
import torch
from torch import nn
import glob
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import tqdm
import numpy as np

from dataset.loader import NucleiDataset, clean_and_check_data
from model.lora_adapter import lora_adapter
from model.metrics import dice_metric, aji_fast_metric, pq_fast_metric
from mobile_sam import sam_model_registry
from segment_anything import SamPredictor, sam_model_registry
import config
from utils.helpers import create_folder
from utils.config import save_dir

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    base_path = kagglehub.dataset_download("ipateam/nuinsseg")
    image_paths = glob.glob(os.path.join(base_path, "**", "tissue images", "*.png"))
    mask_paths = glob.glob(os.path.join(base_path, "**", "label masks modify", "*.tif"))

    image_paths, mask_paths = clean_and_check_data(image_paths, mask_paths)
    dataset = NucleiDataset(image_paths, mask_paths)

    model = sam_model_registry["vit_t"](checkpoint=config.MODEL_CHECKPOINT_PATH).to(config.DEVICE)
    lora_adapter(model.image_encoder, device=config.DEVICE)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], 
        lr=config.LEARNING_RATE
    )

    kf = KFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)
    folds = list(kf.split(dataset))
    avg_dice, avg_aji, avg_pq = [], [], []
    best_model_paths = []
    create_folder(config.SAVE_DIR)

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\nFold {fold+1}")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=config.BATCH_SIZE_TRAIN, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=config.BATCH_SIZE_VAL)

        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=3, shuffle=True)
    val_loader  = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=2)
    best_dice = 0.0
    best_model_path = None
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, masks = imgs.to(device), masks.to(device)

            with torch.no_grad():
                emb = model.image_encoder(imgs)
            sparse, dense = model.prompt_encoder(
                None,
                None,
                None)
            image_pe = model.prompt_encoder.get_dense_pe()

            masks_pred, _ = model.mask_decoder(emb, image_pe, sparse, dense, False)
            logits = masks_pred[:, 0]  # [B, H, W]
            logits = nn.functional.interpolate(logits.unsqueeze(1), size=(1024, 1024), mode='bilinear', align_corners=False)

            loss = nn.functional.binary_cross_entropy_with_logits(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Training Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    dice_scores, aji_scores, pq_scores = [], [], []
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Evaluating"):
            imgs, masks = imgs.to(device), masks.to(device)
            emb = model.image_encoder(imgs)
            sparse, dense = model.prompt_encoder(
                None,
                None,
                None)
            image_pe = model.prompt_encoder.get_dense_pe()
            masks_pred, _ = model.mask_decoder(emb, image_pe, sparse, dense, False)
            pred = torch.sigmoid(masks_pred[:, 0])  # [B, H, W]
            pred = nn.functional.interpolate(pred.unsqueeze(1), size=(1024, 1024), mode='bilinear', align_corners=False)

            pred_bin = (pred > 0.5).float().cpu().numpy().squeeze().astype(np.uint8)
            gt_bin = masks.cpu().numpy().squeeze().astype(np.uint8)
            dice_scores.append(dice_metric(torch.tensor(pred_bin), torch.tensor(gt_bin)).item())
            aji_scores.append(aji_fast_metric(gt_bin, pred_bin))
            pq_result, _ = pq_fast_metric(gt_bin, pred_bin)
            pq_scores.append(pq_result[2])

    avg_dice = np.mean(dice_scores)
    avg_aji = np.mean(aji_scores)
    avg_pq = np.mean(pq_scores)

    print(f" Validation Dice: {np.mean(dice_scores):.4f} | AJI: {np.mean(aji_scores):.4f} | PQ: {np.mean(pq_scores):.4f}")
    if avg_dice > best_dice:
            best_dice = avg_dice
            model_path = os.path.join(save_dir, f"sam_fold{fold+1}_best.pth")
            torch.save(model.state_dict(), model_path)
            best_model_path = model_path
            print(f"Best model found at {model_path}")
    avg_dice.append(np.mean(dice_scores))
    avg_aji.append(np.mean(aji_scores))
    avg_pq.append(np.mean(pq_scores))

    print("\n 5-Fold CV Average Results")
    print(f"Avg Dice: {np.mean(avg_dice):.4f}")
    print(f"Avg AJI : {np.mean(avg_aji):.4f}")
    print(f"Avg PQ  : {np.mean(avg_pq):.4f}")

    print("\nBest model paths:")
    for path in best_model_paths:
      print(path)

if __name__ == "__main__":
    main()
