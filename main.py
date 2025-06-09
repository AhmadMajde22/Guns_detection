import io
from fastapi import FastAPI,File,UploadFile
from fastapi.responses import StreamingResponse
from utils.helpers import predict_and_draw
from PIL import Image




app = FastAPI()

@app.get("/")
def read_root():
    return {
        "messgae": "Welcome to Guns Object Detection API"
    }

@app.post("/predict/")
async def predict(file:UploadFile=File(...)):

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    output_image = predict_and_draw(image)

    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr,format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr,media_type="image/png")
