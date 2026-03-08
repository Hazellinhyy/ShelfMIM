import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
from segment_anything import sam_model_registry, SamPredictor


def xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def mask_to_rle(mask: np.ndarray):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def pick_best_mask(masks, scores, box_xyxy):
    # 1) 先按 score 最大
    i = int(np.argmax(scores))
    m = masks[i].astype(np.uint8)

    # 2) 可选：裁剪到 bbox 内，减少“粘连到邻居”
    x1, y1, x2, y2 = box_xyxy.astype(int)
    x1, y1 = max(x1, 0), max(y1, 0)
    m2 = np.zeros_like(m)
    m2[y1:y2, x1:x2] = m[y1:y2, x1:x2]
    return m2


def convert_one_split(images_dir, ann_in, ann_out, sam_ckpt, model_type="vit_h", device="cuda", min_area=50):
    with open(ann_in, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_by_id = {im["id"]: im for im in coco["images"]}

    anns_by_img = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    new_anns = []
    kept = dropped = 0

    for image_id, anns in tqdm(anns_by_img.items(), desc=os.path.basename(ann_in)):
        im = img_by_id.get(image_id)
        if im is None:
            dropped += len(anns)
            continue

        img_path = os.path.join(images_dir, im["file_name"])
        if not os.path.exists(img_path):
            # 有些数据 file_name 可能不带子目录，你可以在这里做额外 fallback
            dropped += len(anns)
            continue

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            dropped += len(anns)
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)

        for ann in anns:
            box = xywh_to_xyxy(ann["bbox"])
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)

            mask = pick_best_mask(masks, scores, box)
            area = int(mask.sum())
            if area < min_area:
                dropped += 1
                continue

            # bbox 用 mask 重新算，更一致
            ys, xs = np.where(mask > 0)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            bbox_from_mask = [x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)]

            new_ann = dict(ann)
            new_ann["segmentation"] = mask_to_rle(mask)
            new_ann["area"] = area
            new_ann["bbox"] = bbox_from_mask
            new_ann["iscrowd"] = 0

            new_anns.append(new_ann)
            kept += 1

    coco_out = dict(coco)
    coco_out["annotations"] = new_anns

    os.makedirs(os.path.dirname(ann_out), exist_ok=True)
    with open(ann_out, "w", encoding="utf-8") as f:
        json.dump(coco_out, f, ensure_ascii=False)

    print(f"[OK] {ann_in} -> {ann_out} | kept={kept}, dropped={dropped}")


if __name__ == "__main__":
    # ====== 你需要改这里 ======
    RPC_ROOT = "/root/projects/ShelfMIM/dataset/archive-Retail Product Checkout Dataset/retail_product_checkout"

    # 图片目录（注意：你配置里 train2019/val2019/test2019 是相对 root 的）
    TRAIN_IMG = os.path.join(RPC_ROOT, "train2019")
    VAL_IMG   = os.path.join(RPC_ROOT, "val2019")
    TEST_IMG  = os.path.join(RPC_ROOT, "test2019")

    # bbox 标注输入
    TRAIN_IN = os.path.join(RPC_ROOT, "subset_json/instances_train2019.json")
    VAL_IN   = os.path.join(RPC_ROOT, "subset_json/instances_val2019.json")
    TEST_IN  = os.path.join(RPC_ROOT, "subset_json/instances_test2019.json")

    # seg 标注输出（写回同目录也可以，建议新文件名）
    TRAIN_OUT = os.path.join(RPC_ROOT, "subset_json/instances_train2019_samseg.json")
    VAL_OUT   = os.path.join(RPC_ROOT, "subset_json/instances_val2019_samseg.json")
    TEST_OUT  = os.path.join(RPC_ROOT, "subset_json/instances_test2019_samseg.json")

    SAM_CKPT = "/root/projects/ShelfMIM/checkpoints/sam_vit_h_4b8939.pth"
    MODEL_TYPE = "vit_h"
    DEVICE = "cuda"

    convert_one_split(TRAIN_IMG, TRAIN_IN, TRAIN_OUT, SAM_CKPT, MODEL_TYPE, DEVICE)
    convert_one_split(VAL_IMG,   VAL_IN,   VAL_OUT,   SAM_CKPT, MODEL_TYPE, DEVICE)
    convert_one_split(TEST_IMG,  TEST_IN,  TEST_OUT,  SAM_CKPT, MODEL_TYPE, DEVICE)
