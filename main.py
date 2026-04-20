import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image

app = FastAPI()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

@app.get("/")
def home():
    return {"message": "Object Detection API Running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image = Image.open(file.file)

    results = model(image)
    detections = results.pandas().xyxy[0]

    return detections.to_dict(orient="records")
