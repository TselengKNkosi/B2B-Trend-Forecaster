"""
app.py:

FILE FOCUS:
Show the trend forecasting engine as a functioning application

Run server locally with: 
  uvicorn app:app --reload --port 8000
"""
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# local .py file imports
from bedrockclient import BedrockWriter
from decider import get_decider
from forecast import forecast
from signals import TrendSignal

load_dotenv() 

# Intialising FastAPI application instance 
app = FastAPI(
  title = "B2B Trend Forecaster API",
  description = "Automated beauty and fashion trend prediction powered by Strands Decider 2B and Amazon Bedrock.",
  version = "1.1.0",
)

# Instantiating decision engine
decider = get_decider()

# Conditionally intialising Bedrock Writer if model ID is configured in environ variables
writer = BedrockWriter() if os.getenv("BEDROCK_MODEL_ID") else None

# Loading authorized tenant API keys from environ 
  # Note: Production Amendment -> This should be replaced with a secure database lookup
API_KEYS = set(filter(None, (k.strip() for k in os.getenv("API_KEYS", "").split(","))))

class ForecastRequest(BaseModel):
  # A pydantic request schema validating incoming business trend payloads
  name: str = Field(..., description = "Name of the trend (e.g., 'glass skin', 'latina makeup').")
  category: str = Field(..., pattern = "^(beauty|fashion)$", description = "Vertical Sector: 'beauty' or 'fashion'.")
  search_index: List[float] = Field(..., min_length = 8, description = "Weekly search index series (min 8weeks).")
  social_mentions: List[float] = Field(..., min_length = 8, description = "Weekly social mention volume series (min 8 weeks).")
  sell_through: Optional[List[float]] = Field(default = None, description = "Weekly retailer sell-through metrics (optional)")

@app.get("/", include_in_schema = False)
def dashboard() -> FileResponse:
  #Static page = the data call it makesto forecast.py is API-key protected
  return FileResponse(Path(__file__).parent / "dashboard.html")

@app.post("/forecast", summary = "Run a B2B Trend Forecast")
def run_forecast(
  req: ForecastRequest,
  x_api_key: str = Header(..., description = "Enterprise API key.")
) -> dict:
  # Validating API key against authorised tenant store
  if not API_KEYS or x_api_key not in API_KEYS:
    raise HTTPException(
      status_code = 401,
      detail = "Unauthorised: Invalid or missing enterprise API key"
    )

  # converting request model into internal TrendSignal dataclass representation
  trend_signal = TrendSignal(**req.model_dump())

  # Running orchestration pipeline
  forecast_result = forecast(trend_signal, decider, writer)

  return forecast_result