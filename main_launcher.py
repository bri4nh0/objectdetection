#!/usr/bin/env python3
"""
TRD-UQ Multimodal Risk Analysis System - Main Launcher
Temporal Risk Dynamics with Uncertainty Quantification
"""

import argparse
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.multimodal import MultimodalDangerousEventRecognizer
from src.core.config_manager import config_manager
from src.utils.performance_monitor import performance_monitor

def setup_experiment_environment(experiment_id: str):
    """Setup environment for specific experiment"""
    exp_config = config_manager.get_experiment_config(experiment_id)
    
    print(f"\n🎯 Setting up Experiment: {exp_config.get('name', experiment_id)}")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs('experiments', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('recordings', exist_ok=True)
    
    experiment_dir = f"experiments/{experiment_id}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"✅ Experiment directory: {experiment_dir}")
    print(f"✅ Logging level: {exp_config.get('logging_level', 'standard')}")
    print(f"✅ Uncertainty Quantification: {exp_config.get('enable_uq', False)}")
    
    return experiment_dir

def main():
    parser = argparse.ArgumentParser(
        description='TRD-UQ Multimodal Risk Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment Types:
  E1_PERFORMANCE      - System performance and FPS analysis
  E2_FUSION_COMPARISON - Compare TRD-UQ vs simple fusion methods
  E3_SCENARIO_ANALYSIS - Analyze different risk scenarios
  E4_UNCERTAINTY_QUANTIFICATION - Validate uncertainty estimates
  E5_ABLATION_STUDY   - Component ablation study
        
Examples:
  python main_launcher.py --experiment E1_PERFORMANCE
  python main_launcher.py --experiment E3_SCENARIO_ANALYSIS --enable-uq
  python main_launcher.py --experiment E5_ABLATION_STUDY --record
        """
    )
    
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['E1_PERFORMANCE', 'E2_FUSION_COMPARISON', 'E3_SCENARIO_ANALYSIS', 
                               'E4_UNCERTAINTY_QUANTIFICATION', 'E5_ABLATION_STUDY'],
                       help='Experiment configuration to run')
    
    parser.add_argument('--enable-uq', action='store_true', default=None,
                       help='Enable Uncertainty Quantification (overrides experiment config)')
    
    parser.add_argument('--record', action='store_true', default=True,
                       help='Record processed video')
    
    parser.add_argument('--no-dashboard', action='store_true', default=False,
                       help='Disable real-time visualization dashboard')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 TRD-UQ MULTIMODAL RISK ANALYSIS SYSTEM")
    print("Temporal Risk Dynamics with Uncertainty Quantification")
    print("="*70)
    
    try:
        # Setup experiment environment
        experiment_dir = setup_experiment_environment(args.experiment)
        
        # Initialize performance monitoring
        performance_monitor.start_time = performance_monitor.start_time  # Reset timer
        
        # Initialize multimodal recognizer with experiment configuration
        detector = MultimodalDangerousEventRecognizer()
        
        # Configure experiment settings
        detector.experiment_mode = args.experiment
        detector.enable_uq = args.enable_uq if args.enable_uq is not None else config_manager.is_uq_enabled(args.experiment)
        detector.enable_dashboard = not args.no_dashboard
        detector.record_video = args.record
        
        print(f"\n✅ System initialized successfully!")
        print(f"   Experiment: {args.experiment}")
        print(f"   TRD-UQ Enabled: {detector.enable_uq}")
        print(f"   Dashboard: {detector.enable_dashboard}")
        print(f"   Recording: {detector.record_video}")
        print(f"   Hardware: RTX 3050Ti, 16GB RAM, 8 cores")
        
        input("\nPress Enter to start processing... (Ctrl+C to stop)\n")
        
        # Start main processing loop
        detector.run_experiment_loop()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown signal received...")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup and generate report
        print("\n" + "="*50)
        print("SYSTEM SHUTDOWN")
        print("="*50)
        
        # Print performance summary
        performance_monitor.print_summary()
        
        # Save final performance report
        report_path = f"experiments/{args.experiment}/performance_report.txt"
        with open(report_path, 'w') as f:
            stats = performance_monitor.get_stats()
            f.write("TRD-UQ System Performance Report\n")
            f.write("="*40 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✅ Performance report saved: {report_path}")
        print("🎉 TRD-UQ system shutdown complete!")

if __name__ == "__main__":
    main()