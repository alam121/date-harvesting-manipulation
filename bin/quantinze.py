from ultralytics import YOLO

mode = YOLO("yolo_model_kasut_farm_19August_v3.pt")
mode.export(format='engine',device=0,half=True)

