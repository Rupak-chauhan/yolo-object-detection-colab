from fastapi import FastAPI, UploadFile, File, Form
from ultralytics import YOLO
# from IPython.display import display, Image
from augmentation.data_augmentation import AugmentationData
from PIL import Image
from io import BytesIO
import numpy as np

import torch

print("Available ====>", torch.cuda.is_available())

app = FastAPI()

@app.get("/train")
def train_model():
    # augmentation_client = AugmentationData()
    # augmentation_client.init_augmentation()
    # return {"response": "Trained model successfully."}
    model = YOLO('yolov8n.pt')
    # model.to('cuda')
    # model.train(data="data.yaml", epochs=200, imgsz=608, batch=8, save=True, save_crop=True, device=0)
    model.train(data="data.yaml", epochs=50,  patience=300, imgsz=608, batch=8, save=True, save_crop=True, device=[0, 1])
    print("Executed");
    return {"response": "Trained model successfully."}
    # mo

@app.get("/validate")
def validate_model():
    # model = YOLO('C:\Users\RupakSinghChauhan\Desktop\projects\python projects\yolo object detection\yolov8_cusom_model\runs\detect\train\weights\best.pt')
    model = YOLO('best.pt')

    # Customize validation settings
    metrics = model.val(data="data.yaml", epochs=50, imgsz=608, batch=8, conf=0.25, save=True)
    # metrics = model.val(data="data.yaml", epochs=200, imgsz=400, batch=4, conf=0.50, save=True)
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print("Map 50-95 :", metrics.box.map) # map50-95
    print("Map 50 :", metrics.box.map50) # map50
    print("Map 75 :", metrics.box.map75) # map75
    print("Map 50-95 :", metrics.box.maps) # a list contains map50-95 of each category
    return {"Hello": "Validate model successsfully."}

@app.get("/predict")
def predict_model():
    model = YOLO("best.pt")

    # Customize validation settings
    results = model.predict("test_images", epochs=50, save=True, show= True, imgsz=608, batch=8, conf=0.25)
    for r in results:
        print("Box ==> ", r.boxes)
    # model.predict("bus.jpg", save=True, imgsz=320, conf=0.5)
    return {"Hello": "Predicted model successfully."}

@app.post("/predict")
async def predict_model(img: UploadFile = File(...)):
    model = YOLO("best.pt")
    fd = await img.read()
    np_img = np.array(Image.open(BytesIO(fd)))
    # Customize validation settings
    results = model.predict(np_img, epochs=200, save=True, show= True, imgsz=608, batch=8, conf=0.25)
    for r in results:
        print("Box ==> ", r.boxes)
    # model.predict("bus.jpg", save=True, imgsz=320, conf=0.5)
    return {"Hello": "Predicted model successfully."}