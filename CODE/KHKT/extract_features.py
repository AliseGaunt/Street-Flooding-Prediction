
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import matplotlib.pyplot as plt
import pandas as pd
import os

# ==================== CẤU HÌNH ====================
MODEL_PATH = "/content/drive/MyDrive/CODE/KHKT/checkpoints/segformer_floodnet_epoch15"
TEST_DIR = "/content/drive/MyDrive/CODE/KHKT/FloodNet/test/img"
SAVE_DIR = "/content/drive/MyDrive/CODE/KHKT/FloodNet/test_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load processor
try:
    processor = SegformerImageProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    print("Loaded processor from checkpoint.")
except:
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        do_reduce_labels=False,
        size={"height": 512, "width": 512}
    )
    print("Loaded default base processor.")

# Label mapping (FloodNet 0–9)
id2label = {
    0: "background",
    1: "building_flooded",
    2: "building_non_flooded",
    3: "road_flooded",
    4: "road_non_flooded",
    5: "water",
    6: "tree",
    7: "vehicle",
    8: "pool",
    9: "grass",
}
label2id = {v: k for k, v in id2label.items()}

# Load model
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_PATH,
    num_labels=10,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    local_files_only=True
)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Color map
id2color = {
    0: (0, 0, 0),
    1: (0, 0, 128),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (128, 128, 128),
    5: (0, 255, 255),
    6: (0, 128, 0),
    7: (255, 0, 0),
    8: (128, 0, 0),
    9: (0, 255, 0),
}

# ==================== HÀM INFERENCE ====================
def predict_flood_level(image_path):
    # 1. Đọc ảnh
    image = Image.open(image_path).convert("RGB")

    # 2. Preprocess
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    # 3. Inference
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits  # (1, 10, h, w)

    # 4. Upsample về kích thước ảnh gốc
    upsampled_logits = F.interpolate(
        logits,
        size=image.size[::-1],  # (H, W)
        mode="bilinear",
        align_corners=False
    )
    seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()  # (H, W)

    # 5. Tính tỉ lệ nước
    flood_classes = [1, 3, 5]
    water_ratio = np.isin(seg, flood_classes).sum() / seg.size


    # 6. Phân mức ngập theo tỉ lệ water
    if water_ratio < 0.05:
        level, level_text = 0, "Không ngập"
    elif water_ratio < 0.1:
        level, level_text = 1, "Ngập nhẹ"
    elif water_ratio < 0.3:
        level, level_text = 2, "Ngập vừa"
    else:
        level, level_text = 3, "Ngập nặng"

    # 7. Tạo mask màu đẹp
    h, w = seg.shape
    mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in id2color.items():
        mask_colored[seg == label] = color

    return {
        "original": image,
        "segmentation": mask_colored,
        "seg_map": seg,
        "water_ratio": water_ratio,
        "flood_level": level,
        "flood_text": level_text
    }

# Các class ngập
FLOOD_CLASSES = [1, 3, 5]  # building_flooded, road_flooded, water

def extract_features(image_path):
    """
    Trích xuất feature để làm input cho mô hình dự báo ngập tương lai.
    """
    result = predict_flood_level(image_path)
    seg_map = result["seg_map"]

    # 1) Các tỷ lệ pixel
    total = seg_map.size
    ratios = {
        "water_ratio": np.mean(seg_map == 5),
        "building_flooded_ratio": np.mean(seg_map == 1),
        "road_flooded_ratio": np.mean(seg_map == 3),
        "overall_flood_ratio": np.mean(np.isin(seg_map, FLOOD_CLASSES)),
        "tree_ratio": np.mean(seg_map == 6),
        "vehicle_ratio": np.mean(seg_map == 7),
        "pool_ratio": np.mean(seg_map == 8),
        "grass_ratio": np.mean(seg_map == 9),
    }

    # 2) Trích xuất feature từ encoder của SegFormer
    img = Image.open(image_path).convert("RGB")

    inputs = processor(images=img, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_feat = outputs.hidden_states[-1]     # (1, C, H', W')
        pooled_feat = last_feat.mean(dim=[2, 3])  # => (1, C)
        pooled_feat = pooled_feat.cpu().numpy().flatten()

    # Trả về dictionary + vector feature
    return ratios, pooled_feat
