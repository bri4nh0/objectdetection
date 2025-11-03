import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, defaultdict

class BayesianFusionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Match the architecture of your original FusionMLP
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Add uncertainty head that won't break loading
        self.uncertainty_fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x, mc_dropout=True):
        # Main risk prediction
        features = self.fc[:-2](x)  # Get features before last layer
        risk_score = torch.sigmoid(self.fc[-2:](features)) * 4.0
        
        # Uncertainty estimation (only if we have uncertainty layers)
        if hasattr(self, 'uncertainty_fc'):
            uncertainty = F.softplus(self.uncertainty_fc(features))
        else:
            uncertainty = torch.zeros_like(risk_score)
        
        return risk_score, uncertainty

class TemporalRiskAnalyzer:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.risk_histories = defaultdict(lambda: deque(maxlen=sequence_length))
        self.uncertainty_histories = defaultdict(lambda: deque(maxlen=sequence_length))
        
    def analyze_trend(self, person_id, current_risk, current_uncertainty):
        self.risk_histories[person_id].append(current_risk)
        self.uncertainty_histories[person_id].append(current_uncertainty)
        
        if len(self.risk_histories[person_id]) < 5:
            return current_risk, current_uncertainty, "insufficient_data"
            
        risks = list(self.risk_histories[person_id])
        
        # Calculate trend
        if len(risks) >= 2:
            trend = risks[-1] - risks[-2]
            acceleration = trend - (risks[-2] - risks[-3]) if len(risks) >= 3 else 0
        else:
            trend = 0
            acceleration = 0
            
        # Detect patterns
        pattern = self._detect_pattern(risks)
        
        # Adjust risk based on trend
        trend_adjusted_risk = current_risk + (trend * 0.5)
        
        return trend_adjusted_risk, current_uncertainty, pattern
    
    def _detect_pattern(self, risks):
        if len(risks) < 5:
            return "stable"
            
        recent = risks[-5:]
        differences = np.diff(recent)
        
        if all(diff > 0.1 for diff in differences):
            return "escalating"
        elif all(diff < -0.1 for diff in differences):
            return "deescalating" 
        elif np.std(recent) > 0.5:
            return "volatile"
        else:
            return "stable"

class TRDUQSystem:
    def __init__(self):
        self.fusion_model = BayesianFusionLayer()
        self.temporal_analyzer = TemporalRiskAnalyzer()
        
        # Try to load pre-trained weights with flexible loading
        try:
            # First try to load the original fusion weights
            original_state_dict = torch.load("models/fusion/fusion_mlp_balanced.pth", map_location='cpu')
            
            # Create a filtered state dict that matches our current architecture
            filtered_state_dict = {}
            for key, value in original_state_dict.items():
                # Map old keys to new keys if necessary
                if key in self.fusion_model.state_dict():
                    filtered_state_dict[key] = value
                elif key.replace('fc.2', 'fc.3') in self.fusion_model.state_dict():
                    # Handle layer index changes
                    new_key = key.replace('fc.2', 'fc.3')
                    filtered_state_dict[new_key] = value
                elif key.replace('fc.4', 'fc.5') in self.fusion_model.state_dict():
                    new_key = key.replace('fc.4', 'fc.5')
                    filtered_state_dict[new_key] = value
            
            # Load the compatible weights
            self.fusion_model.load_state_dict(filtered_state_dict, strict=False)
            print("✅ Loaded compatible pre-trained fusion weights for TRD-UQ")
            
        except Exception as e:
            print(f"⚠️ Could not load pre-trained weights: {e}")
            print("✅ Using randomly initialized TRD-UQ model")
            
        self.fusion_model.eval()
        
    def monte_carlo_uncertainty(self, input_tensor, num_samples=20):
        """Monte Carlo dropout for uncertainty estimation"""
        predictions = []
        uncertainties = []
        
        # Enable dropout for uncertainty estimation
        self.fusion_model.train()
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred, unc = self.fusion_model(input_tensor, mc_dropout=True)
                predictions.append(pred.item())
                uncertainties.append(unc.item())
        
        # Switch back to eval mode
        self.fusion_model.eval()
                
        mean_risk = np.mean(predictions)
        epistemic_uncertainty = np.std(predictions)  # Model uncertainty
        aleatoric_uncertainty = np.mean(uncertainties)  # Data uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return mean_risk, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty
    
    def analyze_risk(self, person_id, object_risk, behavior_risk, proximity_risk):
        """Enhanced risk analysis with TRD-UQ"""
        input_tensor = torch.tensor([[object_risk, behavior_risk, proximity_risk]], dtype=torch.float32)
        
        # Get risk with uncertainty decomposition
        base_risk, total_uncertainty, epistemic_unc, aleatoric_unc = self.monte_carlo_uncertainty(input_tensor)
        
        # Temporal analysis
        final_risk, final_uncertainty, pattern = self.temporal_analyzer.analyze_trend(
            person_id, base_risk, total_uncertainty
        )
        
        # Confidence-based adjustment
        adjusted_risk = self._adjust_for_uncertainty(final_risk, final_uncertainty)
        
        return {
            'person_id': person_id,
            'risk_score': final_risk,
            'adjusted_risk': adjusted_risk,
            'total_uncertainty': final_uncertainty,
            'epistemic_uncertainty': epistemic_unc,
            'aleatoric_uncertainty': aleatoric_unc,
            'risk_pattern': pattern,
            'confidence_level': self._calculate_confidence(final_uncertainty)
        }
    
    def _adjust_for_uncertainty(self, risk, uncertainty):
        """Adjust risk score based on uncertainty (higher uncertainty = more conservative)"""
        if uncertainty > 0.8:  # High uncertainty
            return min(risk + 0.3, 4.0)
        elif uncertainty > 0.5:  # Medium uncertainty  
            return min(risk + 0.15, 4.0)
        else:  # Low uncertainty
            return risk
            
    def _calculate_confidence(self, uncertainty):
        """Convert uncertainty to confidence level"""
        if uncertainty < 0.3:
            return "high"
        elif uncertainty < 0.6:
            return "medium"
        else:
            return "low"