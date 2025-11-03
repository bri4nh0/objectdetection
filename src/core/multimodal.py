import os
import sys
import time
import threading
import numpy as np
import cv2
import torch
import mysql.connector
from torch import nn
from collections import deque
from ultralytics import YOLO
from datetime import datetime


# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Simple Performance Monitor implementation
class SimplePerformanceMonitor:
    def __init__(self):
        self.inference_times = []
        self.frame_times = []
        self.start_time = time.time()
        
    def log_inference(self, model_name, inference_time):
        self.inference_times.append(inference_time)
        
    def log_frame_processing(self, processing_time):
        self.frame_times.append(processing_time)
        
    def get_stats(self):
        if not self.frame_times:
            return {}
        return {
            'avg_fps': 1.0 / np.mean(self.frame_times) if self.frame_times else 0,
            'avg_inference_ms': np.mean(self.inference_times) * 1000 if self.inference_times else 0,
            'total_frames': len(self.frame_times)
        }

# Create performance monitor instance
performance_monitor = SimplePerformanceMonitor()

# Import TRD-UQ system with fallback
try:
    from trd_uq_system import TRDUQSystem
    print("✅ TRD-UQ system imported successfully")
except ImportError as e:
    print(f"❌ TRD-UQ system import failed: {e}")
    # Fallback implementation
    class TRDUQSystem:
        def __init__(self):
            print("⚠️ Using fallback TRD-UQ system")
        def analyze_risk(self, person_id, object_risk, behavior_risk, proximity_risk):
            # Simple risk calculation as fallback
            base_risk = (object_risk + behavior_risk + proximity_risk) / 3 * 4
            return {
                'risk_score': base_risk,
                'adjusted_risk': base_risk,
                'total_uncertainty': 0.1,
                'epistemic_uncertainty': 0.05,
                'aleatoric_uncertainty': 0.05,
                'risk_pattern': 'stable',
                'confidence_level': 'medium'
            }

# Import config manager with fallback
try:
    from config_manager import config_manager
    print("✅ Config manager imported successfully")
except ImportError as e:
    print(f"❌ Config manager import failed: {e}")
    # Fallback implementation
    class ConfigManager:
        def get_experiment_config(self, experiment_id):
            return {
                'enable_uq': False, 
                'name': 'Standard',
                'logging_level': 'basic'
            }
        def get_model_path(self, model_type):
            # Default model paths
            paths = {
                'yolo': 'models/yolo/yolov8sbest.pt',
                'pose': 'models/yolo/yolov8s-pose.pt',
                'behavior': 'models/fusion/attention_1dcnn_behavior.pth',
                'fusion': 'models/fusion/fusion_mlp_balanced.pth'
            }
            return paths.get(model_type, '')
    
    config_manager = ConfigManager()

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

class EnhancedUnifiedLogger:
    def __init__(self, config):
        try:
            self.conn = mysql.connector.connect(**config)
            self.cursor = self.conn.cursor()
            self.connected = True
            print("✅ Database connected successfully")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            self.connected = False

    def log(self, experiment_id, frame_id=None, system_variant=None, avg_fps=None, frame_interval=None,
            object_risk=None, behavior_risk=None, proximity_risk=None, fusion_score=None,
            rule_score=None, predicted_level=None, scenario_id=None, object_type=None,
            behavior_type=None, module=None, latency_ms=None,
            uncertainty_total=None, uncertainty_epistemic=None, uncertainty_aleatoric=None,
            risk_pattern=None, confidence_level=None, trd_uq_score=None):
        
        if not self.connected:
            return
            
        sql = """
        INSERT INTO experiment_log (
            timestamp, frame_id, experiment_id, system_variant,
            avg_fps, frame_interval,
            object_risk, behavior_risk, proximity_risk, fusion_score,
            rule_score, predicted_level,
            scenario_id, object_type, behavior_type,
            module, latency_ms,
            uncertainty_total, uncertainty_epistemic, uncertainty_aleatoric,
            risk_pattern, confidence_level, trd_uq_score
        )
        VALUES (
            NOW(), %s, %s, %s,
            %s, %s,
            %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s, %s
        )
        """
        vals = (frame_id, experiment_id, system_variant, avg_fps, frame_interval,
                object_risk, behavior_risk, proximity_risk, fusion_score,
                rule_score, predicted_level, scenario_id, object_type,
                behavior_type, module, latency_ms,
                uncertainty_total, uncertainty_epistemic, uncertainty_aleatoric,
                risk_pattern, confidence_level, trd_uq_score)

        try:
            self.cursor.execute(sql, vals)
            self.conn.commit()
        except Exception as e:
            print(f"❌ Database logging error: {e}")

    def close(self):
        if self.connected:
            self.cursor.close()
            self.conn.close()

class Attention1DCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.attention = nn.Linear(128, 1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        attn_scores = torch.softmax(self.attention(x), dim=1)
        x = (x * attn_scores).sum(dim=1)
        return self.fc(x)

class FusionMLP(nn.Module):
    def __init__(self, input_size=3):
        super(FusionMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single scalar output
        )

    def forward(self, x):
        return self.fc(x)

class FrameReader:
    def __init__(self, stream_link):
        self.capture = cv2.VideoCapture(stream_link)
        self.running = True
        self.frame = None
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.frame = frame
            else:
                time.sleep(0.1)

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.capture.release()

class MultimodalDangerousEventRecognizer:
    def __init__(self, experiment_id="E1_PERFORMANCE"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.experiment_id = experiment_id
        self.experiment_config = config_manager.get_experiment_config(experiment_id)
        
        print(f"🚀 Initializing TRD-UQ System for {experiment_id}")
        print(f"   Device: {self.device}")
        
        # Initialize models with error handling
        try:
            self.model = YOLO('models/yolo/yolov8sbest.pt')
            self.pose_model = YOLO('models/yolo/yolov8s-pose.pt')
            self.model.conf = 0.3
            self.pose_model.conf = 0.3
            print("✅ YOLO models loaded")
        except Exception as e:
            print(f"❌ YOLO model loading failed: {e}")
            # Create dummy models for fallback
            self.model = type('DummyModel', (), {'names': ['person'], 'conf': 0.3})()
            self.pose_model = self.model

        self.pose_buffers = {}
        self.escalation_ema = {}
        self.SEQ_LEN = 30
        self.EMA_DECAY = 0.7

        # Behavior model with error handling
        try:
            self.behavior_model = Attention1DCNN(input_dim=34, num_classes=2).to(self.device)
            self.behavior_model.load_state_dict(torch.load('models/fusion/attention_1dcnn_behavior.pth', map_location=self.device))
            self.behavior_model.eval()
            print("✅ Behavior model loaded")
        except Exception as e:
            print(f"❌ Behavior model loading failed: {e}")
            self.behavior_model = None

        # Fusion model with error handling
        try:
            self.fusion_model = FusionMLP(input_size=3)
            self.fusion_model.load_state_dict(torch.load("models/fusion/fusion_mlp_balanced.pth", map_location=self.device))
            self.fusion_model.to(self.device).eval()
            print("✅ Fusion model loaded")
        except Exception as e:
            print(f"❌ Fusion model loading failed: {e}")
            self.fusion_model = None

        # TRD-UQ System
        self.trd_uq_system = TRDUQSystem()
        self.enable_uq = self.experiment_config.get('enable_uq', False)
        
        # Enhanced logger
        self.logger = EnhancedUnifiedLogger({
            'host': 'localhost',
            'user': 'root',
            'password': 'Gth531$@',
            'database': 'model_evaluation'
        })

        # Experiment settings
        self.enable_dashboard = True
        self.system_variant = self._get_system_variant()

        print(f"✅ TRD-UQ System Initialized for {experiment_id}")
        print(f"   Uncertainty Quantification: {self.enable_uq}")
        print(f"   System Variant: {self.system_variant}")

    def _get_system_variant(self):
        """Determine system variant based on experiment and UQ setting"""
        base_variant = self.experiment_config.get('name', 'Standard')
        if self.enable_uq:
            return f"TRD-UQ_{base_variant}"
        return base_variant

    def process_frame(self, frame):
        start_time = time.time()
        
        try:
            object_results = self.model(frame)
            pose_results = self.pose_model(frame)

            obj_boxes = []
            obj_classes = []
            obj_centers = []
            obj_confidences = []

            if hasattr(object_results[0], 'boxes') and object_results[0].boxes is not None:
                for box, cls, conf in zip(object_results[0].boxes.xyxy.cpu(), object_results[0].boxes.cls.cpu(), object_results[0].boxes.conf.cpu()):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    obj_boxes.append([x1, y1, x2, y2])
                    obj_classes.append(int(cls))
                    obj_confidences.append(float(conf))
                    obj_centers.append((cx, cy))

            poses = []
            centers = []
            if hasattr(pose_results[0], 'keypoints') and pose_results[0].keypoints is not None:
                for keypoints in pose_results[0].keypoints.xy:
                    kpts = keypoints.cpu().numpy()
                    center = np.mean(kpts, axis=0)
                    centers.append(center)
                    poses.append(kpts.flatten())
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            obj_boxes, obj_classes, obj_confidences, obj_centers, poses, centers = [], [], [], [], [], []

        processing_time = time.time() - start_time
        performance_monitor.log_frame_processing(processing_time)
        performance_monitor.log_inference('object_detection', processing_time)

        return obj_boxes, obj_classes, obj_confidences, obj_centers, poses, centers

    def run_experiment_loop(self):
        """Main experiment processing loop"""
        RTMP_STREAM = "http://172.22.48.1/live/livestream.flv"
        
        # Label mappings
        label_aliases = {
            'gun': 'Weapon',
            'knife': 'Weapon',
            'fire': 'Fire',
            'pothole': 'Pothole'
        }

        category_info = {
            'Weapon': ((0, 0, 255), 3),
            'Fire': ((0, 165, 255), 2),
            'Pothole': ((255, 0, 0), 1)
        }

        frame_reader = FrameReader(RTMP_STREAM)
        time.sleep(1)

        pose_trackers = {}
        next_id = 0
        
        print("🚀 Starting TRD-UQ Experiment Loop...")
        print("Press 'q' to quit, 'd' to toggle dashboard")

        while True:
            frame = frame_reader.read()
            if frame is None:
                continue

            frame = cv2.resize(frame, (1280, 720))
            start_obj = time.time()
            obj_boxes, obj_classes, obj_confidences, obj_centers, poses, centers = self.process_frame(frame)
            fps = 1 / (time.time() - start_obj) if (time.time() - start_obj) > 0 else 0
            end_obj = time.time()

            # Log performance for latency experiments
            if "LATENCY" in self.experiment_id or "PERFORMANCE" in self.experiment_id:
                self.logger.log(
                    experiment_id=self.experiment_id,
                    frame_id=0,
                    module="object_detector",
                    latency_ms=(end_obj - start_obj) * 1000,
                    avg_fps=round(fps, 2),
                    frame_interval=1 / fps if fps > 0 else 0
                )

            # Draw object detections
            for (x1, y1, x2, y2), cls, conf in zip(obj_boxes, obj_classes, obj_confidences):
                try:
                    cls = int(cls)
                    raw_label = self.model.names[cls].lower()
                    category = label_aliases.get(raw_label, "Unknown")
                    color, _ = category_info.get(category, ((255, 255, 255), 0))
                    
                    label_text = f"{raw_label.capitalize()} ({conf:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                    cv2.putText(frame, f"{category} Risk", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                except Exception as e:
                    print(f"❌ Object drawing error: {e}")

            # Pose tracking and analysis
            updated_ids = set()

            for pose, center in zip(poses, centers):
                try:
                    matched_id = None
                    min_dist = float('inf')
                    for pid, pdata in pose_trackers.items():
                        dist = euclidean(center, pdata['center'])
                        if dist < 50 and dist < min_dist:
                            min_dist = dist
                            matched_id = pid
                    if matched_id is None:
                        matched_id = next_id
                        next_id += 1
                        pose_trackers[matched_id] = {'buffer': deque(maxlen=30)}
                    pose_trackers[matched_id]['center'] = center
                    pose_trackers[matched_id]['buffer'].append(pose)
                    updated_ids.add(matched_id)

                    # Draw pose keypoints
                    for (x, y) in pose.reshape(-1, 2):
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                except Exception as e:
                    print(f"❌ Pose tracking error: {e}")

            # Remove lost trackers
            pose_trackers = {pid: data for pid, data in pose_trackers.items() if pid in updated_ids}

            # Analyze each tracked person
            for pid, data in pose_trackers.items():
                if len(data['buffer']) == 30:
                    # Behavior analysis
                    behavior_pred = 0
                    behavior_risk = 0
                    if self.behavior_model is not None:
                        try:
                            start_behav = time.time()
                            pose_seq = torch.tensor(list(data['buffer']), dtype=torch.float32).unsqueeze(0).to(self.device)
                            behavior_out = self.behavior_model(pose_seq)
                            behavior_pred = behavior_out.argmax(dim=1).item()
                            behavior_risk = 1 if behavior_pred == 1 else 0
                            end_behav = time.time()
                            performance_monitor.log_inference('behavior_analysis', end_behav - start_behav)
                        except Exception as e:
                            print(f"❌ Behavior analysis error: {e}")

                    # Object risk calculation
                    raw_label = "unknown"
                    min_distance = float('inf')
                    object_risk = 0
                    for (obj_box, cls, obj_center) in zip(obj_boxes, obj_classes, obj_centers):
                        try:
                            dist = euclidean(data['center'], obj_center)
                            if dist < min_distance:
                                min_distance = dist
                                raw_label = self.model.names[int(cls)]
                                category = label_aliases.get(raw_label, "Unknown")
                                _, object_risk = category_info.get(category, ((255,255,255), 0))
                        except Exception as e:
                            print(f"❌ Object risk calculation error: {e}")

                    proximity_risk = 1 / (min_distance + 1) if min_distance != float('inf') else 0

                    # Risk fusion with TRD-UQ
                    fusion_score = 0
                    uncertainty = 0.0
                    risk_pattern = "standard"
                    confidence = "high"
                    
                    if self.enable_uq:
                        # TRD-UQ Enhanced Fusion
                        try:
                            start_fusion = time.time()
                            trd_results = self.trd_uq_system.analyze_risk(pid, object_risk, behavior_risk, proximity_risk)
                            fusion_score = trd_results['adjusted_risk']
                            uncertainty = trd_results['total_uncertainty']
                            risk_pattern = trd_results['risk_pattern']
                            confidence = trd_results['confidence_level']
                            end_fusion = time.time()
                            performance_monitor.log_inference('trd_uq_fusion', end_fusion - start_fusion)
                        except Exception as e:
                            print(f"❌ TRD-UQ fusion error: {e}")
                            fusion_score = (object_risk + behavior_risk + proximity_risk) / 3 * 4
                    elif self.fusion_model is not None:
                        # Standard fusion
                        try:
                            start_fusion = time.time()
                            fusion_input = torch.tensor([proximity_risk, behavior_risk, object_risk], dtype=torch.float32).unsqueeze(0).to(self.device)
                            fusion_score = self.fusion_model(fusion_input).item()
                            end_fusion = time.time()
                            performance_monitor.log_inference('standard_fusion', end_fusion - start_fusion)
                        except Exception as e:
                            print(f"❌ Standard fusion error: {e}")
                            fusion_score = (object_risk + behavior_risk + proximity_risk) / 3 * 4
                    else:
                        # Fallback fusion
                        fusion_score = (object_risk + behavior_risk + proximity_risk) / 3 * 4

                    # EMA smoothing
                    if pid not in self.escalation_ema:
                        self.escalation_ema[pid] = fusion_score
                    else:
                        self.escalation_ema[pid] = self.EMA_DECAY * self.escalation_ema[pid] + (1 - self.EMA_DECAY) * fusion_score

                    final_risk = max(0, min(round(self.escalation_ema[pid]), 3))
                    escalation_labels = ["NORMAL", "HAZARD", "DANGEROUS", "CRITICAL"]
                    esc_label = escalation_labels[min(final_risk, len(escalation_labels)-1)]

                    # Enhanced visualization with TRD-UQ info
                    try:
                        cx, cy = map(int, data['center'])
                        if self.enable_uq:
                            display_text = f"B:{behavior_pred} {esc_label} UQ:{uncertainty:.2f}"
                            color = (0, 255, 255) if uncertainty < 0.5 else (0, 165, 255) if uncertainty < 0.8 else (0, 0, 255)
                        else:
                            display_text = f"B:{behavior_pred} {esc_label}"
                            color = (0, 255, 255)
                        
                        cv2.putText(frame, display_text, (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    except Exception as e:
                        print(f"❌ Visualization error: {e}")

                    # Enhanced logging based on experiment type
                    try:
                        self._log_experiment_data(
                            pid, fps, object_risk, behavior_risk, proximity_risk, 
                            fusion_score, final_risk, raw_label, behavior_pred,
                            uncertainty, risk_pattern, confidence, trd_results if self.enable_uq else None
                        )
                    except Exception as e:
                        print(f"❌ Logging error: {e}")

            # Display FPS and experiment info
            status_text = f"FPS: {fps:.2f} | Exp: {self.experiment_id} | UQ: {self.enable_uq}"
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow("TRD-UQ Multimodal Detection", frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.enable_dashboard = not self.enable_dashboard
                print(f"Dashboard {'enabled' if self.enable_dashboard else 'disabled'}")

        # Cleanup
        frame_reader.stop()
        self.logger.close()
        
        # Print performance summary
        stats = performance_monitor.get_stats()
        print("\n📊 Performance Summary:")
        print(f"   Total Frames: {stats.get('total_frames', 0)}")
        print(f"   Average FPS: {stats.get('avg_fps', 0):.2f}")
        print(f"   Average Inference: {stats.get('avg_inference_ms', 0):.2f}ms")
        
        cv2.destroyAllWindows()

    def _log_experiment_data(self, pid, fps, object_risk, behavior_risk, proximity_risk, 
                           fusion_score, final_risk, raw_label, behavior_pred,
                           uncertainty, risk_pattern, confidence, trd_results=None):
        """Enhanced logging for different experiment types"""
        
        base_log_data = {
            'experiment_id': self.experiment_id,
            'frame_id': pid,
            'system_variant': self.system_variant,
            'avg_fps': round(fps, 2),
            'frame_interval': 1 / fps if fps > 0 else 0,
            'object_risk': object_risk,
            'behavior_risk': behavior_risk,
            'proximity_risk': proximity_risk,
            'fusion_score': float(fusion_score),
            'predicted_level': final_risk,
            'object_type': raw_label,
            'behavior_type': "Punching" if behavior_pred == 1 else "Neutral",
            'uncertainty_total': float(uncertainty) if uncertainty else None,
            'risk_pattern': risk_pattern,
            'confidence_level': confidence
        }

        # Add TRD-UQ specific data if available
        if trd_results and self.enable_uq:
            base_log_data.update({
                'uncertainty_epistemic': float(trd_results.get('epistemic_uncertainty', 0)),
                'uncertainty_aleatoric': float(trd_results.get('aleatoric_uncertainty', 0)),
                'trd_uq_score': float(trd_results.get('risk_score', 0))
            })

        # Experiment-specific logging
        if "SCENARIO" in self.experiment_id:
            scenario_id = self._determine_scenario(raw_label, behavior_pred, proximity_risk)
            base_log_data['scenario_id'] = scenario_id

        elif "FUSION" in self.experiment_id:
            rule_score = 3 if object_risk == 3 and behavior_risk == 1 else 2 if object_risk == 3 else 1
            base_log_data['rule_score'] = rule_score

        self.logger.log(**base_log_data)

    def _determine_scenario(self, raw_label, behavior_pred, proximity_risk):
        """Determine scenario based on detections"""
        scenario_id = "Neutral"
        if raw_label in ["fire", "weapon", "pothole"]:
            scenario_id = raw_label.capitalize()
        if behavior_pred == 1:
            scenario_id = "Punching"
        if raw_label in ["weapon"] and behavior_pred == 1:
            scenario_id = "Knife+Punching"
        if raw_label == "fire" and behavior_pred == 0 and proximity_risk > 0.5:
            scenario_id = "Fire+Proximity"
        return scenario_id

# For backward compatibility
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='E1_PERFORMANCE')
    parser.add_argument('--enable-uq', action='store_true')
    args = parser.parse_args()
    
    detector = MultimodalDangerousEventRecognizer(experiment_id=args.experiment)
    detector.enable_uq = args.enable_uq
    detector.run_experiment_loop()