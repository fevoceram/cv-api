from PIL import Image

def load_image(file):
    return Image.open(file).convert("RGB")

def format_results(results):
    df = results.pandas().xyxy[0]
    return df.to_dict(orient="records")
