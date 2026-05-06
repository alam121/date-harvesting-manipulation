from ultralytics import YOLO

mode = YOLO("yolo_26_large_5_5_26.pt")
mode.export(format='engine',device=0,half=True)

