"""`src.core` package initializer."""

__all__ = ["ensembles"]
# Core package for TRD-UQ system
from .trd_uq_system import TRDUQSystem, BayesianFusionLayer, TemporalRiskAnalyzer
from .config_manager import ConfigManager, config_manager
from .multimodal import MultimodalDangerousEventRecognizer

__all__ = [
    'TRDUQSystem', 
    'BayesianFusionLayer', 
    'TemporalRiskAnalyzer',
    'ConfigManager',
    'config_manager',
    'MultimodalDangerousEventRecognizer'
]