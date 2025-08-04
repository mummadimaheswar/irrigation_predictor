import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import streamlit as st

def create_gauge_chart(value, title="Gauge", min_val=0, max_val=100):
    """Create a gauge chart for displaying metrics"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [None, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_val*0.3], 'color': "lightgray"},
                {'range': [max_val*0.3, max_val*0.7], 'color': "yellow"},
                {'range': [max_val*0.7, max_val], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val*0.9
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_time_series_chart(df, columns, title="Time Series"):
    """Create a multi-line time series chart"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, col in enumerate(columns):
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[col],
            mode='lines',
            name=col.replace('_', ' ').title(),
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        height=400
    )
    
    return fig

def format_data_for_display(df):
    """Format dataframe for nice display"""
    display_df = df.copy()
    
    # Format numeric columns
    numeric_columns = display_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if 'moisture' in col.lower() or 'humidity' in col.lower():
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")
        elif 'temperature' in col.lower():
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}°C")
        else:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    
    # Format timestamp
    if 'timestamp' in display_df.columns:
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    return display_df

def calculate_irrigation_score(soil_moisture, temperature, humidity):
    """Calculate irrigation score based on sensor values"""
    score = 0
    
    # Soil moisture component (40% weight)
    if soil_moisture < 20:
        score += 40
    elif soil_moisture < 30:
        score += 30
    elif soil_moisture < 40:
        score += 15
    
    # Temperature component (35% weight)
    if temperature > 35:
        score += 35
    elif temperature > 30:
        score += 25
    elif temperature > 25:
        score += 10
    
    # Humidity component (25% weight)
    if humidity < 30:
        score += 25
    elif humidity < 50:
        score += 15
    elif humidity < 70:
        score += 5
    
    return min(score, 100)

@st.cache_data
def load_demo_data():
    """Load demo data for testing"""
    np.random.seed(42)
    current_time = datetime.now()
    
    data = []
    soil_moisture = 45.0
    
    for i in range(168):  # 1 week of hourly data
        timestamp = current_time + timedelta(hours=i)
        
        # Realistic patterns
        hour_of_day = timestamp.hour
        day_temp_variation = 8 * np.sin(2 * np.pi * hour_of_day / 24)
        temperature = 25 + day_temp_variation + np.random.normal(0, 2)
        
        humidity = max(30, min(95, 75 - 0.5 * (temperature - 25) + np.random.normal(0, 5)))
        
        # Soil moisture decreases over time, with irrigation events
        evaporation_rate = max(0, (temperature - 15) * (100 - humidity) / 1000)
        soil_moisture -= evaporation_rate + np.random.normal(0, 0.3)
        
        # Irrigation events (random)
        if soil_moisture < 25 and np.random.random() < 0.3:
            soil_moisture += np.random.uniform(15, 25)
        
        soil_moisture = max(15, min(80, soil_moisture))
        
        data.append({
            'timestamp': timestamp,
            'soil_moisture': soil_moisture,
            'temperature': temperature,
            'humidity': humidity
        })
    
    return pd.DataFrame(data)
