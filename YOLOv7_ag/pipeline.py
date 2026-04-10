from detect_perso import run_detect_perso
import pandas as pd
directory, detections = run_detect_perso(source='inference/images', weights=['runs/train/yolov7-ag/weights/best.pt'],
                                         conf_thres=0.25, img_size=640, save_csv=False)
print(f"detections{detections}")
print(f"directory{directory}")


df = pd.DataFrame(detections)
print(df.head())