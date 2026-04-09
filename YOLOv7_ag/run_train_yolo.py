#%%
import os
import os.path
from os import path
print(f'Train YOLOv7 .... ')
cmd = "python train_vit.py --workers 0 --device 0,1 --batch-size 16 --epoch 400 --img 640 640 --data data/custom_data.yaml --hyp hyp.scratch.p5.yaml --cfg cfg/training/yolov7-GateAttention.yaml --name yolov7-ag --weights last.pt"
print(cmd)
os.system(cmd)

