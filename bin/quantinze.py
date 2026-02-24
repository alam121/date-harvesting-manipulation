from ultralytics import YOLO

mode = YOLO("yolo_26_dates_trunk_seg_v2.pt")
mode.export(format='engine',device=0,half=True)

