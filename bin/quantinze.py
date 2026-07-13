from ultralytics import YOLO

mode = YOLO("kaust_farm.pt")
mode.export(format='engine',device=0,half=True)

