from ultralytics import YOLO

mode = YOLO("yolo_26_dates_trunk_bunch_seg.pt")
mode.export(format='engine',device=0,half=True)

