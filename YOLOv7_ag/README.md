# YOLOv7-ag

**YOLOv7-ag** is an extension of YOLOv7 designed to enhance the detection of plant organs in herbarium images by integrating an **attention-gate** mechanism. This model outperforms YOLOv7, YOLOv8, and Faster R-CNN in terms of precision, especially for detecting small objects such as seeds, flowers, or stems.

## Objective

Automate the detection of plant organs (leaves, flowers, fruits, seeds, stems, roots) from digitized herbarium images, while addressing challenges such as overlapping, deformation, non-vegetal annotations, and morphological variability.

## Key Contributions

- Added **two attention-gate blocks** in the YOLOv7 model head.
- Improved detection of **small organs** through better exploitation of spatial contextual information.
- Deployment-ready with Triton Inference Server (`deploy/triton-inference-server`).

## Repository Structure

```
YOLOv7-ag/
├── cfg/                          # Configuration files
├── data/                         # Data and annotations
├── deploy/triton-inference-server/  # Triton server for inference
├── figure/                       # Figures used in publications
├── inference/images/            # Images for inference
├── models/                      # YOLOv7-ag model definition
├── runs/                        # Experiment results
├── scripts/                     # Utility scripts
├── tools/, utils/               # Training/inference utilities
├── detect.py, train.py          # Main training and detection scripts
├── run_train_yolo.py            # Simplified training launcher
├── run_detect_yolo.py           # Simplified detection launcher
├── yolov7.pt                    # Pretrained weights
└── README.md
```

## Training

```bash
python train.py --img 640 --batch 16 --epochs 800 --data data/herbarium.yaml --cfg cfg/yolov7-ag.yaml --weights '' --name yolov7-ag --device 0
```

## Detection

```bash
python detect.py --weights runs/train/yolov7-ag/weights/best.pt --conf 0.25 --img 640 --source inference/images
```

##  Results (Validation)

| Method       | Precision | Recall | mAP@0.5 |
|--------------|-----------|--------|---------|
| YOLOv7       | 97.3%     | 96.2%  | 97.0%   |
| **YOLOv7-ag**| **99.2%** | **98.0%**| **99.1%** |
| YOLOv8       | 94.5%     | 92.8%  | 96.1%   |

## Dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reference

If you use this repository, please cite our paper:

> Ariouat, H., Sklab, Y., et al. (2024). *Enhancing YOLOv7 for plant organs detection using attention-gate mechanism.* In Proceedings of PAKDD 2024.

## Links

- Paper: [7.	Ariouat H. Sklab Y., Pignal M, Jabbour F, Vignes Lebbe R, Prifti E, Zucker J-D, and Chenin E. 2024. Enhancing YOLOv7 for plant organs detection using attention-gate mechanism. In the Proceedings of Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD). 2024. https://doi.org/10.1007/978-981-97-2253-2_18](https://doi.org/10.1007/978-981-97-2253-2_18)


---
