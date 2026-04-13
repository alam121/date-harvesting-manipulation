from ultralytics import YOLO

mode = YOLO("yolov26_small_zed_one4k.pt")
mode.export(format='engine',device=0,half=True)

