from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import pandas as pd
from pathlib import Path
import asyncio
from datetime import datetime

app = FastAPI(title="NN Architecture Explorer API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ExperimentConfig(BaseModel):
    name: str
    framework: str  # 'pytorch' or 'tensorflow'
    hidden_dims: List[int]
    activation: str = 'relu'
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    dropout_rate: float = 0.0
    l2_reg: float = 0.0

class ExperimentStatus(BaseModel):
    status: str
    message: str

# Global state for tracking experiments
experiment_status = {}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Neural Network Architecture Explorer API",
        "version": "1.0.0",
        "endpoints": {
            "experiments": "/experiments",
            "results": "/results",
            "summary": "/summary",
            "compare": "/compare"
        }
    }

