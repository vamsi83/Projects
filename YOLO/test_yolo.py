import ultralytics
import numpy
ultralytics.checks()

from ultralytics import YOLO
model=YOLO("yolov8n.pt","v8")

det=model.predict(source=r"https://static.vecteezy.com/system/resources/thumbnails/015/227/028/small/guy-in-glasses-in-front-of-his-employees-young-business-people-in-formal-clothes-working-in-the-office-photo.jpg",conf=0.25, save=True)

print(det)

print(det[0].numpy())