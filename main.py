from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request 
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import warnings; warnings.filterwarnings('ignore')
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import joblib as jb
import numpy as np
import os

load_dotenv()
app = FastAPI()
templates = Jinja2Templates(directory="templates") 
app.mount("/static", StaticFiles(directory="static"), name="static")

def loadNumpyFiles():
    try:
        Threshold_PATH = os.getenv("threshold")
        iForestThreshold_PATH = os.getenv("iForestThreshold")
        LOFThreshold_PATH = os.getenv("LOFThreshold")
        if not Threshold_PATH or not iForestThreshold_PATH or not LOFThreshold_PATH:
            raise ValueError("Threshold paths missing in .env")
        threshold = np.load(Threshold_PATH)
        iForestThreshold = np.load(iForestThreshold_PATH)
        LOFThreshold = np.load(LOFThreshold_PATH)
        return threshold, iForestThreshold, LOFThreshold
    except FileNotFoundError as e:
        raise RuntimeError(f"Threshold file not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading threshold files: {e}")

def loadJBFiles():
    try:
        iForest_PATH = os.getenv("iForest")
        LOF_PATH = os.getenv("LOF")
        RobuScaler_PATH = os.getenv("RobuScaler")
        if not iForest_PATH or not LOF_PATH or not RobuScaler_PATH:
            raise ValueError("Model paths missing in .env")
        iForest = jb.load(iForest_PATH)
        LOF = jb.load(LOF_PATH)
        RobuScaler = jb.load(RobuScaler_PATH)
        return iForest, LOF, RobuScaler
    except FileNotFoundError as e:
        raise RuntimeError(f"Model file not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Model loading error: {e}")

def ScaleData(Data):
    return RobuScaler.transform(Data)

def Decision_Function(Data, Scaled_Data):
    iForest_Score = -iForest.decision_function(Data)[0]
    LOF_Score = -LOF.decision_function(Scaled_Data)[0]
    iForest_Prediction = iForest.predict(Data)[0]
    LOF_Prediction = LOF.predict(Scaled_Data)[0]
    return iForest_Score, LOF_Score, iForest_Prediction, LOF_Prediction

def Normalized(iForestThreshold, LOFThreshold, iForest_Score, LOF_Score):
    i_min, i_max = np.min(iForestThreshold), np.max(iForestThreshold)
    l_min, l_max = np.min(LOFThreshold), np.max(LOFThreshold)
    iForest_Normalized = (iForest_Score - i_min) / (i_max - i_min)
    LOF_Normalized = (LOF_Score - l_min) / (l_max - l_min)
    return iForest_Normalized, LOF_Normalized

def EnsembleScore(iForest_Score, LOF_Score):
    return (0.6 * iForest_Score + 0.4 * LOF_Score)

def label_and_risk(Score, threshold):
    Label = -1 if Score >= threshold else 1
    if Score >= 0.80:
        Risk = "High"
    elif Score >= 0.60:
        Risk = "Medium"
    else:
        Risk = "Low"
    return Label, Risk

def ModelConfidenceAgreement(iForest_Prediction, LOF_Prediction,
                             iForest_Score, LOF_Score):
    iForest_conf = (iForestThreshold < iForest_Score).mean()
    LOF_conf = (LOFThreshold < LOF_Score).mean()
    confidence = (iForest_conf + LOF_conf) / 2
    agreement = 1 if iForest_Prediction == LOF_Prediction else 0
    if agreement == 0:
        confidence *= 0.7
    return confidence, agreement

def Action(Score):
    if Score < 0.2:
        return "A00"
    elif Score < 0.4:
        return "M10"
    elif Score < 0.6:
        return "V20"
    elif Score < 0.8:
        return "O30"
    else:
        return "B40"

def Result(Score, Label, Risk, Confidence, Agreement):
    return {
        "Score": float(round(Score, 3)),
        "Label": int(Label),
        "Risk_Level": Risk,
        "Model_Confidence": float(round(Confidence, 3)),
        "Model_Agreement": Agreement,
        "Action": Action(Score)
    }

def Predict(Data, threshold):
    Scaled_Data = ScaleData(Data)
    iForest_Score, LOF_Score, iForest_Prediction, LOF_Prediction = Decision_Function(
        Data, Scaled_Data
    )
    iForest_Normalized, LOF_Normalized = Normalized(
        iForestThreshold,
        LOFThreshold,
        iForest_Score,
        LOF_Score
    )
    Score = EnsembleScore(iForest_Normalized, LOF_Normalized)
    Label, Risk = label_and_risk(Score, threshold)
    Confidence, Agreement = ModelConfidenceAgreement(
        iForest_Prediction,
        LOF_Prediction,
        iForest_Score,
        LOF_Score
    )
    return Result(Score, Label, Risk, Confidence, Agreement)

class UserInput(BaseModel):
    devices_count: int
    avg_session_duration: float
    api_rate: float
    geo_jump_km: float
    activations_24h: int
    failed_login_ratio: float
    api_std_7d: float
    session_trend: float

@app.on_event("startup")
def load_models():
    global threshold, iForestThreshold, LOFThreshold
    global iForest, LOF, RobuScaler
    try:
        threshold, iForestThreshold, LOFThreshold = loadNumpyFiles()
        iForest, LOF, RobuScaler = loadJBFiles()
        print("Models loaded successfully")
    except Exception as e:
        print("Startup Error:", e)
        raise RuntimeError("Application startup failed")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Unexpected server error",
            "detail": str(exc)
        }
    )

@app.get("/health")
def health():
    print(f"[{datetime.now().replace(microsecond=0)}] API Request Received")
    return {"status": "API running"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    print(f"[{datetime.now().replace(microsecond=0)}] API Request Received")
    return templates.TemplateResponse(
        "api-status.html",
        {"request": request}
    ) 

@app.get("/model-metadata", response_class=HTMLResponse)
def metadata(request: Request):
    print(f"[{datetime.now().replace(microsecond=0)}] API Request Received")
    return templates.TemplateResponse(
        "model-metadata.html",
        {"request": request}
    ) 

#@app.get("/documentation", response_class=HTMLResponse)
#def documentation(request: Request):
#    print(f"[{datetime.now().replace(microsecond=0)}] API Request Received")
#    return templates.TemplateResponse(
#        "api-documentation.html",
#        {"request": request}
#    ) 

@app.post("/predict")
def predict(data: UserInput):
    try:
        UserData = pd.DataFrame([data.model_dump()])
        if UserData.empty:
            raise ValueError("Input data is empty")
        print(f"[{datetime.now().replace(microsecond=0)}] API Request Received")
        result = Predict(UserData, threshold)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        print("Prediction Error:", e)
        raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=str(e)
        )