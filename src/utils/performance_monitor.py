import time
import psutil
from collections import deque
import numpy as np

class PerformanceMonitor:
    def __init__(self):
        self.inference_times = deque(maxlen=100)
        self.frame_times = deque(maxlen=100)
        self.module_times = {}
        self.start_time = time.time()
        self.frame_count = 0
        
    def log_inference(self, model_name: str, inference_time: float):
        if model_name not in self.module_times:
            self.module_times[model_name] = deque(maxlen=50)
        self.module_times[model_name].append(inference_time)
        self.inference_times.append(inference_time)
        
    def log_frame_processing(self, processing_time: float):
        self.frame_times.append(processing_time)
        self.frame_count += 1
        
    def get_stats(self) -> dict:
        if not self.frame_times:
            return {}
            
        stats = {
            'total_frames': self.frame_count,
            'avg_fps': 1.0 / np.mean(list(self.frame_times)) if self.frame_times else 0,
            'avg_inference_ms': np.mean(list(self.inference_times)) * 1000 if self.inference_times else 0,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'uptime_seconds': time.time() - self.start_time
        }
        
        # Add per-module stats
        for module, times in self.module_times.items():
            if times:
                stats[f'{module}_ms'] = np.mean(list(times)) * 1000
                
        return stats
    
    def print_summary(self):
        stats = self.get_stats()
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Frames processed: {stats.get('total_frames', 0)}")
        print(f"Average FPS: {stats.get('avg_fps', 0):.2f}")
        print(f"CPU Usage: {stats.get('cpu_percent', 0):.1f}%")
        print(f"Memory Usage: {stats.get('memory_percent', 0):.1f}%")
        print(f"Uptime: {stats.get('uptime_seconds', 0):.1f}s")
        
        # Module-specific times
        for key, value in stats.items():
            if key.endswith('_ms'):
                print(f"{key.replace('_ms', '').title()} Inference: {value:.2f}ms")

# Global monitor instance
performance_monitor = PerformanceMonitor()