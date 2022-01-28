import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile

from BiWAKO import ResNet


app = FastAPI(title="BiWAKO Image Classification API Demo")
model = ResNet()

# hello world
@app.get("/")
async def root():
    return {"message": "Hello Japan"}


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_COLOR)

    return {"prediction": model.predict(img)}


if __name__ == "__main__":
    uvicorn.run("fast_api_demo:app", port=8000, reload=True)
