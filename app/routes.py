from fastapi import APIRouter, UploadFile, File
from app.model import YOLOModel
from app.utils import load_image, format_results
from app.schemas import DetectionResponse

router = APIRouter()
model = YOLOModel()

@router.get("/")
def health_check():
    return {"status": "API is running"}

@router.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    image = load_image(file.file)
    results = model.predict(image)
    detections = format_results(results)
    return {"detections": detections}
