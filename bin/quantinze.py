from ultralytics import YOLO

mode = YOLO("yolo_26_improved_exposure_corrected.pt")
mode.export(format='engine',device=0,half=True)

