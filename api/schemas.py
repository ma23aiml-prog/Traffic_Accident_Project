"""
api/schemas.py
--------------
Pydantic request / response schemas for the FastAPI prediction service.

Feature names MUST match those used during model training (train_model.py).
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Prediction request ────────────────────────────────────────────────────

class AccidentFeatures(BaseModel):
    """
    Input features for a single accident severity prediction.
    All fields are Optional so the API can handle missing values gracefully
    (the model pipeline imputes missing values internally).
    """
    # Time features
    hour_of_day:  Optional[float] = Field(None, ge=0, le=23,
        description="Hour of accident (0–23)")
    day_of_week:  Optional[int]   = Field(None, ge=0, le=6,
        description="Day of week (0=Monday, 6=Sunday)")
    month:        Optional[int]   = Field(None, ge=1, le=12,
        description="Month of accident (1=January, 12=December)")
    is_weekend:   Optional[int]   = Field(None, ge=0, le=1,
        description="1 if weekend (Sat/Sun), 0 otherwise")

    # Accident details
    number_of_vehicles:   Optional[int]   = Field(None, ge=1,
        description="Number of vehicles involved")
    number_of_casualties: Optional[int]   = Field(None, ge=1,
        description="Number of casualties")
    speed_limit:          Optional[int]   = Field(None, ge=0,
        description="Posted speed limit in mph (e.g. 30, 60, 70)")

    # Weather (from Open-Meteo grid)
    wx_temp_mean_c:   Optional[float] = Field(None,
        description="Daily mean temperature at 2m (°C)")
    wx_precip_mm:     Optional[float] = Field(None, ge=0,
        description="Daily precipitation sum (mm)")
    wx_wind_max_kmh:  Optional[float] = Field(None, ge=0,
        description="Daily maximum wind speed at 10m (km/h)")

    # Road context (decoded labels from DfT STATS19 lookup)
    road_type_label:               Optional[str] = Field(None,
        description="Road type label (e.g. 'Single carriageway', 'Motorway')")
    light_conditions_label:        Optional[str] = Field(None,
        description="Light conditions (e.g. 'Daylight', 'Darkness – lights lit')")
    weather_conditions_label:      Optional[str] = Field(None,
        description="Weather conditions (e.g. 'Fine no high winds', 'Raining no high winds')")
    road_surface_conditions_label: Optional[str] = Field(None,
        description="Road surface (e.g. 'Dry', 'Wet or damp')")
    junction_detail_label:         Optional[str] = Field(None,
        description="Junction type (e.g. 'Not at junction', 'Roundabout')")
    urban_or_rural_area_label:     Optional[str] = Field(None,
        description="Area type ('Urban', 'Rural')")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "hour_of_day": 8,
                    "day_of_week": 1,
                    "month": 3,
                    "is_weekend": 0,
                    "number_of_vehicles": 2,
                    "number_of_casualties": 1,
                    "speed_limit": 30,
                    "wx_temp_mean_c": 8.5,
                    "wx_precip_mm": 2.3,
                    "wx_wind_max_kmh": 25.0,
                    "road_type_label": "Single carriageway",
                    "light_conditions_label": "Daylight",
                    "weather_conditions_label": "Raining no high winds",
                    "road_surface_conditions_label": "Wet or damp",
                    "junction_detail_label": "T or staggered junction",
                    "urban_or_rural_area_label": "Urban",
                }
            ]
        }
    }


class BatchAccidentFeatures(BaseModel):
    """Request body for batch predictions."""
    records: list[AccidentFeatures] = Field(
        ..., min_length=1, max_length=500,
        description="List of accident records (max 500 per request)"
    )


# ── Prediction response ────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """Single prediction result."""
    predicted_class:     int   = Field(...,
        description="Predicted severity class (1=Fatal, 2=Serious, 3=Slight)")
    predicted_label:     str   = Field(...,
        description="Human-readable severity label")
    probability_fatal:   float = Field(..., ge=0, le=1,
        description="Probability of Fatal (class 1)")
    probability_serious: float = Field(..., ge=0, le=1,
        description="Probability of Serious (class 2)")
    probability_slight:  float = Field(..., ge=0, le=1,
        description="Probability of Slight (class 3)")


class PredictResponse(BaseModel):
    """Response for a single /predict call."""
    prediction: PredictionResult
    model_version: str


class BatchPredictResponse(BaseModel):
    """Response for a /predict-batch call."""
    predictions:   list[PredictionResult]
    count:         int
    model_version: str


# ── Health / info responses ────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str


class ModelInfoResponse(BaseModel):
    model_type:          str
    target_column:       str
    severity_classes:    dict
    numeric_features:    list[str]
    categorical_features: list[str]
    n_features:          int
    dataset_size:        int
    accuracy:            Optional[float]
    macro_f1:            Optional[float]
    trained_at:          str
    mlflow_run_id:       Optional[str]
