#!/bin/bash

# Streamlit deployment script for Smart Irrigation System

echo "🌾 Deploying Smart Irrigation Prediction System..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p models
mkdir -p data
mkdir -p logs

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

echo "✅ Deployment setup complete!"
echo "🚀 Starting Streamlit app..."

# Run the app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
