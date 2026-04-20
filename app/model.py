import torch
from app.config import MODEL_NAME

class YOLOModel:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', MODEL_NAME, pretrained=True)

    def predict(self, image):
        results = self.model(image)
        return results
