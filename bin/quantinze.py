from ultralytics import YOLO

mode = YOLO("yolo26_zedone4k_3classes.pt")
mode.export(format='engine',device=0,half=True)

