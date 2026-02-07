from ultralytics import YOLO
model = YOLO("yolov8n.pt")
image_path = "https://ultralytics.com/images/zidane.jpg"
results = model.predict(image_path, conf=0.25)
import cv2
out_path = "inference.jpg"
cv2.imwrite(out_path, results[0].plot())
print(f"Saved {out_path}")
