import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
API_Key_Header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)

GOOGLE_SHEET_API_URL = os.getenv("GOOGLE_SHEET_API_URL")
GOOGLE_SHEET_SECRET = os.getenv("GOOGLE_SHEET_SECRET")

def verifyAPI_Key(
    candidateAPI_Key: str = Security(API_Key_Header)
):
    if not candidateAPI_Key:
        raise HTTPException(
            status_code=401,
            detail="API Key missing"
        )
    if not GOOGLE_SHEET_API_URL or not GOOGLE_SHEET_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Authentication configuration unavailable"
        )
    try:
        response = requests.post(
            GOOGLE_SHEET_API_URL,
            json={
                "api_key": candidateAPI_Key,
                "secret": GOOGLE_SHEET_SECRET
            },
            timeout=20
        )
        response.raise_for_status()
        result = response.json()
        if result.get("valid") is not True:
            raise HTTPException(
                status_code=401,
                detail=result.get(
                    "message",
                    "Invalid API Key"
                )
            )
        print("API Key verified successfully")
        return candidateAPI_Key
    except HTTPException:
        raise
    except requests.RequestException as e:
        print("Google Sheets Connection Error:", e)
        raise HTTPException(
            status_code=503,
            detail="API Key verification service unavailable"
        )
    except ValueError:
        raise HTTPException(
            status_code=503,
            detail="Invalid authentication server response"
        )