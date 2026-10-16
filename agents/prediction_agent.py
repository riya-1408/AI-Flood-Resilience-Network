from agents.base_agent import BaseAgent
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import os
import datetime

class FloodMLP(nn.Module):
    """Improved Multi-Layer Perceptron for flood prediction"""
    def __init__(self, input_dim):
        super(FloodMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class PredictionAgent(BaseAgent):
    def __init__(self, agent_id: str, model_path: str):
        capabilities = ["flood_prediction", "risk_assessment", "ml_inference", "confidence_analysis"]
        super().__init__(agent_id, "PredictionAgent", capabilities)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.feature_columns = checkpoint.get('feature_columns', 
                                             ['elevation_m', 'rainfall_mm', 'river_proximity', 'slope_deg'])
        input_dim = len(self.feature_columns)
        model_type = checkpoint.get('model_type', 'LSTM')
        
        # Load appropriate model architecture
        if model_type == 'MLP':
            self.model = FloodMLP(input_dim=input_dim).to(self.device)
        else:
            # Fallback to LSTM for older models
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from models.train_model import FloodLSTM
            self.model = FloodLSTM(input_dim=input_dim).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.scaler = checkpoint['scaler']
        self.model_type = model_type
        
        print(f"   ✓ Model loaded: {model_type} with {input_dim} features, F1={checkpoint.get('f1_score', 0):.3f}")
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        import datetime
        self.status = "predicting"
        self.last_action = datetime.datetime.now()
        self.action_count += 1
        
        print(f"\n🤖 PREDICTION AGENT")
        
        try:
            # Extract or estimate all 20 features
            elevation_m = data.get('elevation_m', 50)
            rainfall_mm = data.get('rainfall_mm', 50)
            slope_deg = data.get('slope_deg', 5)
            river_proximity_binary = data.get('river_proximity', 1)
            
            # Convert binary river proximity to distance
            distance_to_water_m = 100 if river_proximity_binary == 1 else 2000
            
            # Temporal
            now = datetime.datetime.now()
            month = now.month
            season = 3 if month in [6, 7, 8, 9] else 1 if month in [12, 1, 2] else 2 if month in [3, 4, 5] else 4
            period_in_year = (month - 1) * 2 + (1 if now.day <= 15 else 2)

            # Hydrological & flood risk estimation based on inputs
            rainfall_intensity = min(1.0, max(0.0, rainfall_mm / 500.0))
            elevation_factor = max(0.05, (500.0 - min(480.0, elevation_m)) / 500.0)
            river_factor = 1.6 if river_proximity_binary == 1 else 0.6
            slope_factor = max(0.2, (45.0 - min(40.0, slope_deg)) / 45.0)

            composite_risk = (rainfall_intensity * 0.45 + elevation_factor * 0.25 + slope_factor * 0.15) * river_factor

            region_flood_freq = min(0.48, max(0.18, 0.22 + composite_risk * 0.25))
            region_avg_flood_area = max(0.5, composite_risk * 45.0)
            region_max_flood_area = max(1.0, region_avg_flood_area * 5.0)
            region_avg_pct_flooded = min(15.0, max(0.1, composite_risk * 5.0))
            region_std_pct_flooded = max(0.05, region_avg_pct_flooded * 0.5)
            region_avg_pop_exposed = min(24000.0, max(50.0, 500.0 + region_avg_flood_area * 180.0))
            prev1_area_flooded = max(0.0, region_avg_flood_area * (0.8 if rainfall_mm > 100 else 0.1))
            prev1_pct_flooded = max(0.0, region_avg_pct_flooded * (0.8 if rainfall_mm > 100 else 0.1))
            prev2_area_flooded = prev1_area_flooded * 0.7
            prev2_pct_flooded = prev1_pct_flooded * 0.7
            cumulative_3p_area = region_avg_flood_area + prev1_area_flooded + prev2_area_flooded
            flood_area_trend = (rainfall_mm - 120.0) / 40.0
            country_flood_freq = 0.30
            country_avg_area = 17.64
            country_seasonal_avg = 17.70 * (1.4 if season == 3 else 0.8)
            pop_density_proxy = min(4600.0, max(100.0, 3000.0 if elevation_m < 30 else 800.0))
            cropland_ratio = 0.08

            # Comprehensive features_dict supporting all model feature schemas
            features_dict = {
                # Schema 1: Satellite / Event dataset features
                'month': month,
                'season': season,
                'period_in_year': period_in_year,
                'region_flood_freq': region_flood_freq,
                'region_avg_flood_area': region_avg_flood_area,
                'region_max_flood_area': region_max_flood_area,
                'region_avg_pct_flooded': region_avg_pct_flooded,
                'region_std_pct_flooded': region_std_pct_flooded,
                'region_avg_pop_exposed': region_avg_pop_exposed,
                'prev1_area_flooded': prev1_area_flooded,
                'prev1_pct_flooded': prev1_pct_flooded,
                'prev2_area_flooded': prev2_area_flooded,
                'prev2_pct_flooded': prev2_pct_flooded,
                'cumulative_3p_area': cumulative_3p_area,
                'flood_area_trend': flood_area_trend,
                'country_flood_freq': country_flood_freq,
                'country_avg_area': country_avg_area,
                'country_seasonal_avg': country_seasonal_avg,
                'pop_density_proxy': pop_density_proxy,
                'cropland_ratio': cropland_ratio,

                # Schema 2: Topographic / physical features
                'elevation_m': elevation_m,
                'slope_deg': slope_deg,
                'aspect_deg': 180,
                'curvature': -0.5 if elevation_m < 100 else 0.5,
                'flow_accumulation': max(1000, 10000 / (elevation_m + 1)),
                'distance_to_water_m': distance_to_water_m,
                'watershed_area_km2': 50,
                'drainage_density': 2.0,
                'soil_permeability_mm_hr': 15,
                'topographic_wetness_index': 12 if elevation_m < 100 else 8,
                'rainfall_24h_mm': rainfall_mm,
                'antecedent_rainfall_7d_mm': rainfall_mm * 0.6,
                'soil_moisture_pct': min(80, rainfall_mm * 0.15),
                'temperature_c': 25,
                'urban_density_pct': 40,
                'vegetation_cover_pct': 50,
                'impervious_surface_pct': 30,
                'drainage_capacity': 0.6
            }
            
            # Create feature array in correct order
            if len(self.feature_columns) > 0:
                features = np.array([[features_dict.get(col, 0.0) for col in self.feature_columns]])
            else:
                features = np.array([[elevation_m, rainfall_mm, river_proximity_binary, slope_deg]])
            
            print(f"   Features ({len(self.feature_columns)}): elevation={elevation_m}m, rainfall={rainfall_mm}mm, slope={slope_deg}°")
            
            # Normalize and predict
            features_scaled = self.scaler.transform(features)
            
            if self.model_type == 'MLP':
                input_tensor = torch.FloatTensor(features_scaled).to(self.device)
            else:
                sequence = np.repeat(features_scaled, 10, axis=0)
                input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(input_tensor)
                flood_probability = prediction.item()
            
            risk_level = self._categorize_risk(flood_probability)
            confidence = self._calculate_confidence(flood_probability, features_dict)
            risk_factors = self._analyze_risk_factors(features_dict, flood_probability)
            
            print(f"   ✓ Prediction: {flood_probability*100:.1f}% | Risk: {risk_level} | Confidence: {confidence['level']}")
            
            self.status = "idle"
            
            return {
                "location": data.get('location', {}),
                "flood_probability": flood_probability,
                "risk_level": risk_level,
                "confidence": confidence,
                "risk_factors": risk_factors,
                "weather": data.get('weather', {}),
                "timestamp": data.get('timestamp'),
                "features_used": {
                    "elevation_m": elevation_m,
                    "rainfall_24h_mm": rainfall_mm,
                    "slope_deg": slope_deg,
                    "distance_to_water_m": distance_to_water_m
                }
            }
            
        except Exception as e:
            self.status = "error"
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _categorize_risk(self, probability: float) -> str:
        if probability >= 0.65:
            return "HIGH"
        elif probability >= 0.35:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_confidence(self, probability: float, features: Dict) -> Dict[str, Any]:
        """Calculate prediction confidence with bounds"""
        # Confidence is higher when the model is very certain (close to 0 or 1)
        certainty = abs(probability - 0.5) * 2  # 0 at 0.5, 1 at extremes
        
        # Margin of error decreases with certainty
        margin = max(0.05, 0.15 * (1 - certainty))
        
        conf_score = 0.7 + certainty * 0.25  # 70-95%
        
        return {
            "score": round(conf_score, 3),
            "level": "High" if conf_score > 0.85 else "Medium" if conf_score > 0.75 else "Low",
            "lower_bound": round(max(0, probability - margin), 3),
            "upper_bound": round(min(1, probability + margin), 3),
            "margin_of_error": round(margin, 3)
        }
    
    def _analyze_risk_factors(self, features: Dict, probability: float) -> list:
        """Analyze which features contribute most to the risk"""
        factors = []
        
        # Rainfall impact
        rainfall = features.get('rainfall_24h_mm', 0)
        if rainfall > 400:
            factors.append({"factor": "Extreme Rainfall", "impact": "critical", "value": f"{rainfall}mm/24h", "icon": "🌧️", "contribution": 0.35})
        elif rainfall > 200:
            factors.append({"factor": "Heavy Rainfall", "impact": "high", "value": f"{rainfall}mm/24h", "icon": "🌧️", "contribution": 0.25})
        elif rainfall > 100:
            factors.append({"factor": "Moderate Rainfall", "impact": "medium", "value": f"{rainfall}mm/24h", "icon": "🌦️", "contribution": 0.15})
        else:
            factors.append({"factor": "Light Rainfall", "impact": "low", "value": f"{rainfall}mm/24h", "icon": "🌤️", "contribution": 0.05})
        
        # Elevation impact
        elevation = features.get('elevation_m', 0)
        if elevation < 20:
            factors.append({"factor": "Very Low Elevation", "impact": "critical", "value": f"{elevation}m", "icon": "⬇️", "contribution": 0.30})
        elif elevation < 50:
            factors.append({"factor": "Low Elevation", "impact": "high", "value": f"{elevation}m", "icon": "⬇️", "contribution": 0.20})
        elif elevation < 200:
            factors.append({"factor": "Moderate Elevation", "impact": "medium", "value": f"{elevation}m", "icon": "➡️", "contribution": 0.10})
        else:
            factors.append({"factor": "High Elevation", "impact": "low", "value": f"{elevation}m", "icon": "⬆️", "contribution": 0.03})
        
        # Water proximity
        dist = features.get('distance_to_water_m', 1000)
        if dist < 200:
            factors.append({"factor": "Near Water Body", "impact": "high", "value": f"{dist}m", "icon": "🌊", "contribution": 0.20})
        else:
            factors.append({"factor": "Far from Water", "impact": "low", "value": f"{dist}m", "icon": "🏔️", "contribution": 0.05})
        
        # Slope
        slope = features.get('slope_deg', 5)
        if slope < 1:
            factors.append({"factor": "Flat Terrain", "impact": "high", "value": f"{slope}°", "icon": "🏞️", "contribution": 0.15})
        elif slope < 5:
            factors.append({"factor": "Gentle Slope", "impact": "medium", "value": f"{slope}°", "icon": "📐", "contribution": 0.08})
        else:
            factors.append({"factor": "Steep Slope", "impact": "low", "value": f"{slope}°", "icon": "⛰️", "contribution": 0.03})
        
        # Sort by contribution
        factors.sort(key=lambda x: x['contribution'], reverse=True)
        return factors
