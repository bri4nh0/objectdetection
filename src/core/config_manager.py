import yaml
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = "configs/system_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        return {
            'models': {
                'yolo': {
                    'path': 'models/yolo/yolov8sbest.pt',
                    'confidence': 0.3
                },
                'pose': {
                    'path': 'models/yolo/yolov8s-pose.pt',
                    'confidence': 0.3
                },
                'behavior': {
                    'path': 'models/fusion/attention_1dcnn_behavior.pth'
                },
                'fusion': {
                    'path': 'models/fusion/fusion_mlp_balanced.pth'
                }
            },
            'experiments': {
                'E1_PERFORMANCE': {
                    'name': 'System Performance Analysis',
                    'modules': ['object', 'pose', 'behavior', 'fusion'],
                    'logging_level': 'performance',
                    'enable_uq': False
                },
                'E2_FUSION_COMPARISON': {
                    'name': 'Fusion Method Comparison',
                    'modules': ['object', 'pose', 'behavior', 'fusion'],
                    'logging_level': 'fusion',
                    'enable_uq': True
                },
                'E3_SCENARIO_ANALYSIS': {
                    'name': 'Risk Scenario Analysis',
                    'modules': ['object', 'pose', 'behavior', 'fusion'],
                    'logging_level': 'scenario',
                    'enable_uq': True
                },
                'E4_UNCERTAINTY_QUANTIFICATION': {
                    'name': 'Uncertainty Quantification',
                    'modules': ['object', 'pose', 'behavior', 'fusion'],
                    'logging_level': 'uncertainty',
                    'enable_uq': True
                },
                'E5_ABLATION_STUDY': {
                    'name': 'Component Ablation Study',
                    'modules': ['object', 'pose', 'behavior', 'fusion'],
                    'logging_level': 'ablation',
                    'enable_uq': True
                }
            },
            'trd_uq': {
                'mc_samples': 20,
                'uncertainty_threshold': 0.8,
                'sequence_length': 30
            },
            'streaming': {
                'rtmp_url': "http://172.22.48.1/live/livestream.flv",
                'frame_size': [1280, 720],
                'target_fps': 30
            },
            'visualization': {
                'enable_dashboard': True,
                'show_uncertainty': True,
                'show_patterns': True
            }
        }
    
    def get_experiment_config(self, experiment_id: str) -> Dict[str, Any]:
        return self.config['experiments'].get(experiment_id, {})
    
    def get_model_path(self, model_type: str) -> str:
        return self.config['models'][model_type]['path']
    
    def get_trd_uq_config(self) -> Dict[str, Any]:
        return self.config.get('trd_uq', {})
    
    def is_uq_enabled(self, experiment_id: str) -> bool:
        exp_config = self.get_experiment_config(experiment_id)
        return exp_config.get('enable_uq', False)

# Global config instance
config_manager = ConfigManager()