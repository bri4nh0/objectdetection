import os
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

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

class UnifiedLogger:
    def __init__(self, config):
        self.conn = mysql.connector.connect(**config)
        self.cursor = self.conn.cursor()

    def log(self, experiment_id, frame_id=None, system_variant=None, avg_fps=None, frame_interval=None,
            object_risk=None, behavior_risk=None, proximity_risk=None, fusion_score=None,
            rule_score=None, predicted_level=None, scenario_id=None, object_type=None,
            behavior_type=None, module=None, latency_ms=None):
        
        sql = """
        INSERT INTO experiment_log (
            timestamp, frame_id, experiment_id, system_variant,
            avg_fps, frame_interval,
            object_risk, behavior_risk, proximity_risk, fusion_score,
            rule_score, predicted_level,
            scenario_id, object_type, behavior_type,
            module, latency_ms
        )
        VALUES (
            NOW(), %s, %s, %s,
            %s, %s,
            %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s
        )
        """
        vals = (frame_id, experiment_id, system_variant, avg_fps, frame_interval,
                object_risk, behavior_risk, proximity_risk, fusion_score,
                rule_score, predicted_level, scenario_id, object_type,
                behavior_type, module, latency_ms)

        self.cursor.execute(sql, vals)
        self.conn.commit()

    def close(self):
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
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('models/yolo/yolov8sbest.pt')
        self.pose_model = YOLO('models/yolo/yolov8s-pose.pt')
        self.model.conf = 0.3
        self.pose_model.conf = 0.3

        self.pose_buffers = {}
        self.escalation_ema = {}
        self.SEQ_LEN = 30
        self.EMA_DECAY = 0.7

        self.behavior_model = Attention1DCNN(input_dim=34, num_classes=2).to(self.device)
        self.behavior_model.load_state_dict(torch.load('models/fusion/attention_1dcnn_behavior.pth', map_location=self.device))
        self.behavior_model.eval()

        self.fusion_model = FusionMLP(input_size=3)
        self.fusion_model.load_state_dict(torch.load("models/fusion/fusion_mlp_balanced.pth", map_location=self.device))
        self.fusion_model.to(self.device).eval()

        self.logger = UnifiedLogger({
            'host': 'localhost',
            'user': 'root',
            'password': 'Gth531$@',
            'database': 'model_evaluation'
        })
        
        # Initialize video recording
        self.recording = False
        self.video_writer = None

    def start_recording(self, output_path="recordings", frame_size=(1280, 720), fps=20):
        """Start recording the processed frames to a video file"""
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.avi"
        output_file = os.path.join(output_path, filename)
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
        self.recording = True
        print(f"Started recording to: {output_file}")
        
        return output_file

    def stop_recording(self):
        """Stop recording and release the video writer"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("Recording stopped")

    def record_frame(self, frame):
        """Record a frame if recording is enabled"""
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)

    def process_frame(self, frame):
        object_results = self.model(frame)
        pose_results = self.pose_model(frame)

        obj_boxes = []
        obj_classes = []
        obj_centers = []
        obj_confidences = []

        if object_results[0].boxes is not None:
            for box, cls, conf in zip(object_results[0].boxes.xyxy.cpu(), object_results[0].boxes.cls.cpu(), object_results[0].boxes.conf.cpu()):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                obj_boxes.append([x1, y1, x2, y2])
                obj_classes.append(int(cls))
                obj_confidences.append(float(conf))  # NEW
                obj_centers.append((cx, cy))

        poses = []
        centers = []
        if pose_results[0].keypoints is not None:
            for keypoints in pose_results[0].keypoints.xy:
                kpts = keypoints.cpu().numpy()
                center = np.mean(kpts, axis=0)
                centers.append(center)
                poses.append(kpts.flatten())

        return obj_boxes, obj_classes, obj_confidences, obj_centers, poses, centers

# === RTMP Link ===
RTMP_STREAM = "http://172.22.48.1/live/livestream.flv"

# === Label Alias + Risk/Color Maps ===
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

EXPERIMENT_MODE = "E1_FULL"

# === Module toggles ===
ENABLE_POSE = True
ENABLE_BEHAVIOR = True
ENABLE_FUSION = True

# === Toggle logic for specific experiment configurations ===
if EXPERIMENT_MODE == "E1_YOLO":
    ENABLE_POSE = False
    ENABLE_BEHAVIOR = False
    ENABLE_FUSION = False
elif EXPERIMENT_MODE == "E1_YOLO_POSE":
    ENABLE_POSE = True
    ENABLE_BEHAVIOR = False
    ENABLE_FUSION = False
elif EXPERIMENT_MODE == "E1_FULL":
    ENABLE_POSE = True
    ENABLE_BEHAVIOR = True
    ENABLE_FUSION = True
elif EXPERIMENT_MODE == "E5_NO_FUSION":
    ENABLE_POSE = True
    ENABLE_BEHAVIOR = True
    ENABLE_FUSION = False
elif EXPERIMENT_MODE == "E5_NO_BEHAVIOR":
    ENABLE_POSE = True
    ENABLE_BEHAVIOR = False
    ENABLE_FUSION = True
elif EXPERIMENT_MODE == "E5_FULL":
    ENABLE_POSE = True
    ENABLE_BEHAVIOR = True
    ENABLE_FUSION = True
  # Change to "E2", "E3", "E4", or "E5" depending on the experiment

if __name__ == "__main__":
    detector = MultimodalDangerousEventRecognizer()
    frame_reader = FrameReader(RTMP_STREAM)
    time.sleep(1)

    pose_trackers = {}
    next_id = 0
    
    # Start recording automatically
    recording_path = detector.start_recording()

    while True:
        frame = frame_reader.read()
        if frame is None:
            continue

        frame = cv2.resize(frame, (1280, 720))
        start_obj = time.time()
        obj_boxes, obj_classes, obj_confidences, obj_centers, poses, centers = detector.process_frame(frame)
        fps = 1 / (time.time() - start_obj)
        end_obj = time.time()

        if EXPERIMENT_MODE == "E4_LATENCY":
            detector.logger.log(
                experiment_id="E4_Latency",
                frame_id=0,
                module="object_detector",
                latency_ms=(end_obj - start_obj) * 1000
    )

        # === Draw Object Labels ===
        for (x1, y1, x2, y2), cls, conf in zip(obj_boxes, obj_classes, obj_confidences):
            cls = int(cls)
            raw_label = detector.model.names[cls].lower()
            category = label_aliases.get(raw_label, "Unknown")
            color, _ = category_info.get(category, ((255, 255, 255), 0))
            # === Enhanced Bounding Box and Risk Label Rendering ===
            label_text = f"{raw_label.capitalize()} ({conf:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Thicker box for better visibility
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # Optional: show category label below the box (e.g., "Weapon Risk")
            category_label = f"{category} Risk"
            cv2.putText(frame, category_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        updated_ids = set()

        for pose, center in zip(poses, centers):
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

            for (x, y) in pose.reshape(-1, 2):
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        pose_trackers = {pid: data for pid, data in pose_trackers.items() if pid in updated_ids}

        for pid, data in pose_trackers.items():
            if len(data['buffer']) == 30:
                if ENABLE_BEHAVIOR:
                    pose_seq = torch.tensor(list(data['buffer']), dtype=torch.float32).unsqueeze(0).to(detector.device)
                    start_behav = time.time()
                    behavior_out = detector.behavior_model(pose_seq)
                    end_behav = time.time()
                    behavior_pred = behavior_out.argmax(dim=1).item()
                    behavior_risk = 1 if behavior_pred == 1 else 0
                else:
                    behavior_pred = 0
                    behavior_risk = 0

                raw_label = "unknown"  # initialize in case no object is nearby
                min_distance = float('inf')
                object_risk = 0
                for (obj_box, cls, obj_center) in zip(obj_boxes, obj_classes, obj_centers):
                    dist = euclidean(data['center'], obj_center)
                    if dist < min_distance:
                        min_distance = dist
                        raw_label = detector.model.names[int(cls)]  # ✅ fix here
                        category = label_aliases.get(raw_label, "Unknown")
                        _, object_risk = category_info.get(category, ((255,255,255), 0))

                proximity_risk = 1 / (min_distance + 1)
                fusion_input = torch.tensor([proximity_risk, behavior_risk, object_risk], dtype=torch.float32).unsqueeze(0).to(detector.device)

                if ENABLE_FUSION:
                    start_fusion = time.time()

                    # Get scalar output from regression model (no softmax!)
                    fusion_score = detector.fusion_model(fusion_input).item()
                    escalation_raw = fusion_score  # for EMA smoothing and predicted_level

                    end_fusion = time.time()

                    # Apply EMA smoothing per person
                    if pid not in detector.escalation_ema:
                        detector.escalation_ema[pid] = escalation_raw
                    else:
                        detector.escalation_ema[pid] = 0.7 * detector.escalation_ema[pid] + 0.3 * escalation_raw

                    ema_score = detector.escalation_ema[pid]
                    final_esc = round(ema_score)

                else:
                    escalation_raw = 3 if object_risk == 3 and behavior_risk == 1 else 2 if object_risk == 3 else 1

                if pid not in detector.escalation_ema:
                    detector.escalation_ema[pid] = escalation_raw
                else:
                    old = detector.escalation_ema[pid]
                    detector.escalation_ema[pid] = detector.EMA_DECAY * old + (1 - detector.EMA_DECAY) * escalation_raw

                final_esc = max(0, min(round(detector.escalation_ema[pid]), 3))
                escalation_labels = ["NORMAL", "HAZARD", "DANGEROUS", "CRITICAL"]
                esc_label = escalation_labels[min(final_esc, len(escalation_labels)-1)]

                cx, cy = map(int, data['center'])
                cv2.putText(frame, f"B:{behavior_pred} {esc_label}", (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                # === Unified Logging Per Experiment ===
                if EXPERIMENT_MODE == "E1_YOLO":
                    detector.logger.log(
                        experiment_id="E1_FPS",
                        system_variant="YOLO-only",
                        avg_fps=round(fps, 2),
                        frame_interval=1 / fps
                    )

                elif EXPERIMENT_MODE == "E1_YOLO_POSE":
                    detector.logger.log(
                        experiment_id="E1_FPS",
                        system_variant="YOLO+Pose",
                        avg_fps=round(fps, 2),
                        frame_interval=1 / fps
                    )

                elif EXPERIMENT_MODE == "E1_FULL":
                    detector.logger.log(
                        experiment_id="E1_FPS",
                        frame_id=pid,
                        system_variant="YOLO+Pose+Fusion+Behavior",
                        avg_fps=round(fps, 2),
                        frame_interval=1 / fps,
                        object_risk=object_risk,
                        behavior_risk=behavior_risk,
                        proximity_risk=proximity_risk,
                        fusion_score=float(fusion_score),
                        rule_score=3 if object_risk == 3 and behavior_risk == 1 else 2 if object_risk == 3 else 1,
                        predicted_level=final_esc,
                        object_type=raw_label,
                        behavior_type="Punching" if behavior_pred == 1 else "Neutral"
                    )

                elif EXPERIMENT_MODE == "E2":
                    detector.logger.log(
                        experiment_id="E2_Fusion",
                        frame_id=pid,
                        object_risk=object_risk,
                        behavior_risk=behavior_risk,
                        proximity_risk=proximity_risk,
                        fusion_score=escalation_raw,  # ✅ the real fusion score
                        rule_score=3 if object_risk == 3 and behavior_risk == 1 else 2 if object_risk == 3 else 1,
                        predicted_level=final_esc
                    )

                elif EXPERIMENT_MODE == "E3":
                    # Dynamically determine scenario_id based on detections
                    scenario_id = "Neutral"
                    if raw_label in ["fire", "weapon", "pothole"]:
                        scenario_id = raw_label.capitalize()
                    if behavior_pred == 1:
                        scenario_id = "Punching"
                    if raw_label in ["weapon"] and behavior_pred == 1:
                        scenario_id = "Knife+Punching"
                    if raw_label == "fire" and behavior_pred == 0 and proximity_risk > 0.5:
                        scenario_id = "Fire+Proximity"

                    # Log all relevant risk and output fields
                    detector.logger.log(
                        experiment_id="E3_Scenario",
                        frame_id=pid,
                        scenario_id=scenario_id,
                        object_type=raw_label,
                        behavior_type="Punching" if behavior_pred == 1 else "Neutral",
                        object_risk=object_risk,
                        behavior_risk=behavior_risk,
                        proximity_risk=proximity_risk,
                        fusion_score=float(fusion_score),
                        predicted_level=final_esc
                    )
                
                elif EXPERIMENT_MODE == "E4_LATENCY" and ENABLE_BEHAVIOR:
                    detector.logger.log(
                        experiment_id="E4_Latency",
                        frame_id=pid,
                        module="behavior_model",
                        latency_ms=(end_behav - start_behav) * 1000
                    )

                elif EXPERIMENT_MODE == "E4_LATENCY" and ENABLE_FUSION:
                    detector.logger.log(
                        experiment_id="E4_Latency",
                        frame_id=pid,
                        module="fusion_mlp",
                        latency_ms=(end_fusion - start_fusion) * 1000
                    )

                elif EXPERIMENT_MODE == "E5_NO_FUSION":
                    detector.logger.log(
                        experiment_id="E5_Ablation",
                        frame_id=pid,
                        system_variant="No-FusionMLP",
                        object_risk=object_risk,
                        behavior_risk=behavior_risk,
                        predicted_level=final_esc
                    )

                elif EXPERIMENT_MODE == "E5_NO_BEHAVIOR":
                    detector.logger.log(
                        experiment_id="E5_Ablation",
                        frame_id=pid,
                        system_variant="No-BehaviorModel",
                        object_risk=object_risk,
                        behavior_risk=behavior_risk,
                        predicted_level=final_esc
                    )

                elif EXPERIMENT_MODE == "E5_FULL":
                    detector.logger.log(
                        experiment_id="E5_Ablation",
                        frame_id=pid,
                        system_variant="Full-System",
                        object_risk=object_risk,
                        behavior_risk=behavior_risk,
                        predicted_level=final_esc
                    )

        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        
        # Record the frame with all detections
        detector.record_frame(frame)
        
        cv2.imshow("Multimodal Detection", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # Toggle recording with 'r' key
            if detector.recording:
                detector.stop_recording()
            else:
                detector.start_recording()

    # Cleanup
    frame_reader.stop()
    detector.stop_recording()
    detector.logger.close()
    cv2.destroyAllWindows()