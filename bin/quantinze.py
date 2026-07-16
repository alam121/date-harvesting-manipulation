from ultralytics import YOLO

mode = YOLO("best.pt")
mode.export(format='engine',device=0,half=True)

