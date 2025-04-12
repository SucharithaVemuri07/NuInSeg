import os
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import glob

from skimage.color import label2rgb
from skimage.measure import label
from skimage.segmentation import watershed, clear_border
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi
from skimage.filters import sobel

from dataset.loader import NucleiDataset, clean_and_check_data
from model.lora_adapter import lora_adapter
from mobile_sam import sam_model_registry
import config

# Load dataset
base_path = kagglehub.dataset_download("ipateam/nuinsseg")
image_paths = glob.glob(os.path.join(base_path, "**", "tissue images", "*.png"))
mask_paths = glob.glob(os.path.join(base_path, "**", "label masks modify", "*.tif"))
image_paths, mask_paths = clean_and_check_data(image_paths, mask_paths)

dataset = NucleiDataset(image_paths, mask_paths)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

model = sam_model_registry["vit_t"](checkpoint=config.MODEL_CHECKPOINT_PATH).to(config.DEVICE)
lora_adapter(model.image_encoder, device=config.DEVICE)

load_path = "/path/to/best_model"  
model.load_state_dict(torch.load(load_path, map_location=config.DEVICE))
model.eval()

def visualize_prediction(img_tensor, gt_mask, pred_bin, watershed_labels):
    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
    gt_mask_np = gt_mask.cpu().squeeze().numpy()
    pred_bin = pred_bin
    watershed_labels = watershed_labels

    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    axs[0].imshow(img_np)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(gt_mask_np)
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis("off")

    axs[2].imshow(pred_bin)
    axs[2].set_title("Predicted Mask")
    axs[2].axis("off")

    axs[3].imshow(label2rgb(watershed_labels, image=img_np, bg_label=0))
    axs[3].set_title("Watershed Instances")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()

sample_count = 0
with torch.no_grad():
    for imgs, masks in tqdm.tqdm(val_loader, desc="Predicting"):
        imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
        
        emb = model.image_encoder(imgs)
        sparse, dense = model.prompt_encoder(None, None, None)
        image_pe = model.prompt_encoder.get_dense_pe()
        masks_pred, _ = model.mask_decoder(emb, image_pe, sparse, dense, False)

        pred = torch.sigmoid(masks_pred[:, 0])
        pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(1024, 1024), mode='bilinear', align_corners=False)

        pred_bin = (pred > 0.5).float().cpu().numpy().squeeze().astype(np.uint8)
        gt_bin = masks.cpu().numpy().squeeze().astype(np.uint8)

        distance = ndi.distance_transform_edt(pred_bin)
        markers = label(pred_bin)
        labels = watershed(-distance, markers, mask=pred_bin)
        labels = remove_small_objects(labels, min_size=30)
        labels = clear_border(labels)

        visualize_prediction(imgs[0], masks[0], pred_bin, labels)

        sample_count += 1
        if sample_count >= 3:
            break
