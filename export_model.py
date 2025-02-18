from ultralytics import YOLO

model = YOLO('./results_trains/entrainement_facture1_yolov8_new_field2/weights/best.pt')
model.export(format='onnx', opset=11)