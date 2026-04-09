import argparse
import time
import os
from pathlib import Path
from typing import List, Optional

import torch
import cv2
from tqdm import tqdm
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, TracedModel
from dataset_perso import DatasetPerso


def auto_generate_weights(detections_list, grid_size=14, num_classes=6, img_shape=(640, 640)):
    # 7 canaux : [0:leaf, 1:root, 2:stem, 3:flower, 4:fruit, 5:seed, 6:FOND]
    masks = torch.zeros((num_classes + 1, grid_size, grid_size))

    # On initialise tout en Fond (canal 6)
    masks[6] = 1.0

    if not detections_list:
        return masks.unsqueeze(0)

    img_h, img_w = img_shape

    for d in detections_list:
        c = d['class_id']
        x1 = int((d['x_min_pixel'] / img_w) * grid_size)
        x2 = int((d['x_max_pixel'] / img_w) * grid_size)
        y1 = int((d['y_min_pixel'] / img_h) * grid_size)
        y2 = int((d['y_max_pixel'] / img_h) * grid_size)

        x1, x2 = max(0, min(grid_size - 1, x1)), max(0, min(grid_size, x2))
        y1, y2 = max(0, min(grid_size - 1, y1)), max(0, min(grid_size, y2))

        if y2 > y1 and x2 > x1:
            # 1. On active la classe détectée (ex: 0 pour feuille, 2 pour tige)
            masks[c, y1:y2, x1:x2] = 1.0

            # 2. ON DÉSACTIVE LE FOND à cet endroit précis
            # C'est la ligne qui te manquait pour séparer les deux !
            masks[6, y1:y2, x1:x2] = 0.0

    return masks.unsqueeze(0)


def run_detect_perso(
        weights=['best.pt'], source='', img_size=640, conf_thres=0.40, iou_thres=0.60,
        device='', grid_size=14, limit=None, clean_pt=False, skip_existing=True
):
    set_logging()
    device = select_device(device)

    if clean_pt:
        for p in Path(source).rglob("*.pt"):
            if p.name != Path(weights[0]).name:
                p.unlink()

    model = attempt_load(weights, map_location=device)
    names = model.module.names if hasattr(model, 'module') else model.names
    imgsz = check_img_size(img_size, s=int(model.stride.max()))
    model.half() if device.type != 'cpu' else model.float()

    dataset = DatasetPerso(source, img_size=imgsz)

    count = 0
    t0 = time.time()

    for path, img, im0 in tqdm(dataset, desc="Segmentation YOLO"):
        if limit is not None and count >= limit:
            break

        p = Path(path)
        save_path_pt = p.parent / f"{p.stem}_weights.pt"

        if skip_existing and save_path_pt.exists():
            count += 1
            continue

        img = torch.from_numpy(img).to(device)
        img = img.half() if device.type != 'cpu' else img.float()
        img /= 255.0
        if img.ndimension() == 3: img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)

        h_orig, w_orig = im0.shape[:2]
        dets_vit = []
        patches_data = []

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)

                    dets_vit.append({
                        'x_min_pixel': x1, 'y_min_pixel': y1,
                        'x_max_pixel': x2, 'y_max_pixel': y2,
                        'class_id': int(cls.item())
                    })

                    patch = im0[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
                    if patch.size > 0:
                        patch_resized = cv2.resize(patch, (224, 224))
                        patch_tensor = torch.from_numpy(patch_resized).permute(2, 0, 1).float() / 255.0
                        patches_data.append({
                            'tensor': patch_tensor,
                            'class_id': int(cls.item())
                        })

        poids_multi = auto_generate_weights(dets_vit, grid_size=grid_size, num_classes=len(names),
                                            img_shape=(h_orig, w_orig))

        data_to_save = {
            'global_heatmap': poids_multi.cpu(),
            'patches': patches_data
        }
        torch.save(data_to_save, save_path_pt)

        count += 1

    print(f"\nTerminé en {time.time() - t0:.2f}s. Fichiers générés/scannés : {count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt')
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--no-skip', action='store_false', dest='skip')
    parser.set_defaults(skip=True)
    opt = parser.parse_args()

    run_detect_perso(weights=[opt.weights], source=opt.source, limit=opt.limit,
                     clean_pt=opt.clean, skip_existing=opt.skip)