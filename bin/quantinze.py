from ultralytics import YOLO

mode = YOLO("yolo26_improved_exposure_data.pt")
mode.export(format='engine',device=0,half=True)

