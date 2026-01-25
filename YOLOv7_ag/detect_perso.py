import argparse
import csv
import time
from pathlib import Path
from typing import List, Optional

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from tqdm import tqdm

# Tes imports locaux/externes
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages  # J'ai laissé LoadStreams/LoadImages, mais tu utilises DatasetPerso
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging, increment_path, strip_optimizer
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel, time_synchronized
from dataset_perso import DatasetPerso

random.seed(2)


# Renommage en 'run_detection' pour une meilleure clarté de son rôle
def run_detect_perso(
        weights: List[str] = ['yolov7.pt'],
        source: str = 'inference/images',
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str = '',
        view_img: bool = False,
        save_csv: bool = False,
        save_img: bool = False,
        classes: Optional[List[int]] = None,
        agnostic_nms: bool = False,
        augment: bool = False,
        project: str = 'runs/detect',
        name: str = 'exp',
        exist_ok: bool = False,
        no_trace: bool = False,
        update: bool = False,
):
    """
    Exécute le processus d'inférence (détection d'objets) et retourne le chemin
    du dossier de sauvegarde et les détections.
    """
    # 1. INITIALISATION DES ARGUMENTS
    # La fonction reçoit maintenant ses propres arguments, plus besoin de l'objet 'opt'
    trace = not no_trace
    save_img = save_img and not source.endswith('.txt')
    all_detections = []

    # 2. CONFIGURATION DES DOSSIERS
    set_logging()
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))
    (save_dir / 'labels' if save_csv else save_dir).mkdir(parents=True, exist_ok=True)

    # 3. CONFIGURATION DE PYTORCH
    device = select_device(device)
    half = device.type != 'cpu'

    # 4. CHARGEMENT DU MODÈLE
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride)

    if trace:
        model = TracedModel(model, device, img_size)

    if half:
        model.half()

    # 5. CHARGEMENT DES DONNÉES
    dataset = DatasetPerso(source, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        # run once
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()

    # 6. BOUCLE D'INFÉRENCE PRINCIPALE (MODIFIÉE : 'im0s' remplacé par 'im0' et retiré 'vid_cap')
    for path, img, im0 in tqdm(dataset, desc="Processing images"):  # Grrr, ajout de la description

        # Prétraitement
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 7. INFÉRENCE ET NMS
        pred = model(img, augment=augment)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # 8. TRAITEMENT DES DÉTECTIONS
        for i, det in enumerate(pred):

            p, s, im_display = path, '', im0.copy()  # Copie pour l'affichage si view_img/save_img est activé
            p = Path(p)
            save_path_base = str(save_dir / p.stem)  # Utilisation de p.stem pour le nom de base
            save_path_img = save_path_base + p.suffix  # Chemin complet pour l'image

            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im_display.shape)[[1, 0, 1, 0]]

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im_display.shape).round()

                # Print results (console)
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results (CSV & Image)
                for *xyxy, conf, cls in reversed(det):

                    #TODO : Changer ça pê rajouter une variable en paramètre
                    if True:
                        xyxy_pixels = [int(x.item()) for x in xyxy]
                        detection_data = {
                            'id': str(p.stem),
                            'class_id': int(cls.item()),
                            'class_name': names[int(cls.item())],
                            'confidence': round(conf.item(), 4),
                            'x_min_pixel': xyxy_pixels[0],
                            'y_min_pixel': xyxy_pixels[1],
                            'x_max_pixel': xyxy_pixels[2],
                            'y_max_pixel': xyxy_pixels[3],
                        }
                        all_detections.append(detection_data)

                    if save_img or view_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im_display, label=label, color=colors[int(cls)], line_thickness=2)

            # 9. SAUVEGARDE DES IMAGES

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path_img, im_display)
                    # print(f" The image with the result is saved in: {save_path_img}")

    # 10. FINALISATION ET CSV
    if save_csv or save_img:
        print(f"\nThe output with the result is saved in :{save_dir}")


        if all_detections:
            csv_output_path = save_dir / 'all_detections_results.csv'
            fieldnames = list(all_detections[0].keys())
            print(f"Sauvegarde de {len(all_detections)} détections dans {csv_output_path}")

            with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_detections)

            print("Sauvegarde CSV terminée !")

    print(f'Done. ({time.time() - t0:.3f}s)')
    return save_dir, all_detections  # Retourne les résultats


if __name__ == '__main__':
    # Le bloc '__main__' gère l'exécution en ligne de commande et appelle la fonction 'run_detection'

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-csv', action='store_true', help='save results to *.csv')
    parser.add_argument('--save_img', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    print(opt)

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov7.pt']:
                run_detect_perso(**vars(opt))  # Appel de la fonction avec les arguments de la ligne de commande
                strip_optimizer(opt.weights)
        else:
            run_detect_perso(**vars(opt))  # Appel de la fonction avec les arguments de la ligne de commande