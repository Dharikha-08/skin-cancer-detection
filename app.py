import os
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
import traceback

# Import our modular Tabular Model
from tabular_model import TabularModel

# Configuration
MODEL_PATH = "best_image_model_opt.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Fusion Parameters
def get_fusion_params():
    alpha, threshold = 0.6, 0.5
    if os.path.exists("fusion_params.txt"):
        with open("fusion_params.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "alpha:" in line: alpha = float(line.split(":")[1].strip())
                if "threshold:" in line: threshold = float(line.split(":")[1].strip())
    return alpha, threshold

# Global Loaders
tab_model = TabularModel()
tab_model.load_model()
alpha, threshold = get_fusion_params()

def load_image_model():
    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=2)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

img_model = load_image_model()

# App Initialization
app = FastAPI(title="Multimodal DermVision API")

# FIX: Explicitly allow the React origin to solve CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for development to prevent CORS errors across ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
async def root():
    return {"status": "ok", "fusion_params": {"alpha": alpha, "threshold": threshold}}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    sex: str = Form(None),
    location: str = Form(None),
    elevation: str = Form(None),
    diff: str = Form(None),
    score: int = Form(0),
    pig_net: str = Form("absent"),
    streaks: str = Form("absent"),
    pigment: str = Form("absent"),
    reg_struc: str = Form("absent"),
    dots: str = Form("absent"),
    blue_veil: str = Form("absent"),
    vasc: str = Form("absent")
):
    try:
        # Default sanity check for None values
        sex = sex or "male"
        location = location or "back"
        elevation = elevation or "palpable"
        diff = diff or "medium"
        
        # 1. Image Model Logic
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            img_prob = torch.softmax(img_model(img_tensor), dim=1)[:, 1].item()

        # 2. Tabular Model Logic
        # Mapping frontend names to TabularModel expected column names
        input_data = {
            'pigment_network': pig_net,
            'streaks': streaks,
            'pigmentation': pigment,
            'regression_structures': reg_struc,
            'dots_and_globules': dots,
            'blue_whitish_veil': blue_veil,
            'vascular_structures': vasc,
            'level_of_diagnostic_difficulty': diff,
            'elevation': elevation,
            'location': location,
            'sex': sex,
            'seven_point_score': score
        }
        
        df_input = pd.DataFrame([input_data])
        tab_prob = tab_model.predict_proba(df_input)[0]

        # 3. Fusion
        final_prob = alpha * img_prob + (1 - alpha) * tab_prob
        label = "MALIGNANT" if final_prob > threshold else "BENIGN"

        return {
            "label": label,
            "probability": round(float(final_prob), 4),
            "breakdown": {
                "image_component": round(float(img_prob), 4),
                "tabular_component": round(float(tab_prob), 4)
            },
            "fusion_weight": round(float(alpha), 2),
            "threshold": round(float(threshold), 2),
            "status": "success"
        }

    except Exception as e:
        # Crucial: Log the full error to track down 'NoneType' or logic errors
        print("INTERNAL ERROR:", traceback.format_exc())
        return {"status": "error", "message": f"Server Logic Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)