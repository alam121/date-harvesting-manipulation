from ultralytics import YOLO

mode = YOLO("yolo26_26April.pt")
mode.export(format='engine',device=0,half=True)

