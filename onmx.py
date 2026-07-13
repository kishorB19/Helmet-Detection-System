from ultralytics import YOLO
model = YOLO("helmet.pt")
model.export(format="onnx")