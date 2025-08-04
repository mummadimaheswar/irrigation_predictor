import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    # Model Configuration
    MODEL_PATH: str = "advanced_irrigation_model.h5"
    SCALER_PATH: str = "scaler.pkl"
    SEQUENCE_LENGTH: int = 48
    FEATURE_COLUMNS: list = None
    
    # App Configuration
    APP_TITLE: str = "Smart Irrigation Prediction System"
    APP_ICON: str = "🌾"
    LAYOUT: str = "wide"
    
    # Thresholds
    IRRIGATION_THRESHOLD: float = 0.5
    SOIL_MOISTURE_CRITICAL: float = 25.0
    TEMPERATURE_HIGH: float = 35.0
    HUMIDITY_LOW: float = 40.0
    
    # API Configuration (for future use)
    API_URL: str = os.getenv("API_URL", "")
    API_KEY: str = os.getenv("API_KEY", "")
    
    def __post_init__(self):
        if self.FEATURE_COLUMNS is None:
            self.FEATURE_COLUMNS = ['soil_moisture', 'temperature', 'humidity']

# Global config instance
config = AppConfig()
