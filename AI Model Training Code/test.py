from ultralytics import YOLO

#model = YOLO("best.pt")

#results = model("test.jpg")

#results[0].show()

model = YOLO("yolov11_version/thief_detection_yolov11/weights/best.pt")
model.predict("knife_69.png", save=True)
