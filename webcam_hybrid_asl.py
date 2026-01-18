#!/usr/bin/env python3
"""
REAL-TIME ASL RECOGNITION WITH HYBRID MODEL
Uses webcam for live sign language translation. 
"""

import cv2
import torch
import numpy as np
import json
from collections import deque
import time
from transformers import VideoMAEImageProcessor

from hybrid_asl_model import HybridASLModel, MediaPipeLandmarkExtractor


class HybridASLWebcam:
    """Real-time ASL recognition using hybrid model."""
    
    def __init__(self,
                 model_path='best_hybrid_asl_model. pth',
                 label_mapping_path='label_mapping.json',
                 num_frames=16,
                 device='cuda'):
        
        print("Loading Hybrid ASL System...")
        
        self.device = device
        self.num_frames = num_frames
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            label_to_idx = json.load(f)
        self.idx_to_label = {v:  k for k, v in label_to_idx.items()}
        num_classes = len(label_to_idx)
        
        # Load model
        self.model = HybridASLModel(
            num_classes=num_classes,
            videomae_model='MCG-NJU/videomae-base',
            hidden_dim=256,
            fusion_type='concat'
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        print("✓ Model loaded")
        
        # Initialize extractors
        self.landmark_extractor = MediaPipeLandmarkExtractor()
        self.videomae_processor = VideoMAEImageProcessor. from_pretrained('MCG-NJU/videomae-base')
        print("✓ Feature extractors ready")
        
        # Frame buffers
        self.frame_buffer = deque(maxlen=num_frames)
        self.landmark_buffer = deque(maxlen=num_frames)
        
        # Prediction state
        self.current_prediction = None
        self.confidence = 0.0
        self.prediction_history = deque(maxlen=5)  # For smoothing
        
        print("✓ System ready!")
        print()
    
    def process_frame(self, frame_rgb, timestamp_ms):
        """Process a single frame and add to buffers."""
        
        # Store frame for VideoMAE
        self.frame_buffer.append(frame_rgb. copy())
        
        # Extract landmarks
        landmarks = self.landmark_extractor.extract_frame_landmarks(
            frame_rgb, timestamp_ms
        )
        self.landmark_buffer.append(landmarks)
    
    @torch.no_grad()
    def predict(self):
        """Make prediction from current buffers."""
        
        if len(self.frame_buffer) < self.num_frames:
            return None, 0.0
        
        # Prepare VideoMAE input
        frames = list(self.frame_buffer)
        pixel_values = self.videomae_processor(
            frames, return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Prepare landmark input
        landmarks = np.array(list(self.landmark_buffer))
        landmarks = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict
        predictions, confidence, probs = self.model. predict(pixel_values, landmarks)
        
        pred_idx = predictions[0].item()
        conf = confidence[0].item()
        
        word = self.idx_to_label. get(pred_idx, "Unknown")
        
        return word, conf
    
    def draw_ui(self, frame, fps):
        """Draw UI overlay on frame."""
        h, w = frame.shape[:2]
        
        # Background panel
        cv2.rectangle(frame, (10, 10), (w-10, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w-10, 160), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "HYBRID ASL RECOGNITION", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 255), 2)
        
        # FPS and buffer status
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        buffer_pct = len(self.frame_buffer) / self.num_frames * 100
        cv2.putText(frame, f"Buffer:  {len(self.frame_buffer)}/{self.num_frames} ({buffer_pct:.0f}%)", 
                   (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        # Prediction result
        if self.current_prediction:
            # Confidence bar
            bar_width = int((w - 40) * self.confidence)
            bar_color = (0, 255, 0) if self.confidence > 0.7 else (0, 165, 255) if self.confidence > 0.4 else (0, 0, 255)
            cv2.rectangle(frame, (20, 90), (20 + bar_width, 110), bar_color, -1)
            cv2.rectangle(frame, (20, 90), (w-20, 110), (255, 255, 255), 2)
            
            # Prediction text
            cv2.putText(frame, f"SIGN: {self.current_prediction}", 
                       (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, (0, 255, 0), 2)
            
            cv2.putText(frame, f"{self.confidence*100:.1f}%", 
                       (w-100, 145), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Collecting frames...  Make a sign!", 
                       (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (200, 200, 200), 1)
        
        # Controls
        cv2.putText(frame, "Q: Quit | R: Reset | SPACE: Force Predict", 
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (150, 150, 150), 1)
        
        return frame
    
    def run(self, camera_id=0):
        """Run the webcam recognition loop."""
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ Could not open webcam!")
            return
        
        print("="*60)
        print("WEBCAM STARTED")
        print("="*60)
        print("Controls:")
        print("  Q / ESC  - Quit")
        print("  R        - Reset buffers")
        print("  SPACE    - Force prediction")
        print("="*60)
        print()
        
        fps = 0
        frame_count = 0
        start_time = time.time()
        last_predict_time = 0
        predict_cooldown = 1.0  # seconds
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mirror
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Calculate timestamp
                timestamp_ms = int(frame_count * 1000 / 30)
                
                # Process frame
                self.process_frame(frame_rgb, timestamp_ms)
                
                # Auto-predict when buffer is full
                current_time = time.time()
                if (len(self.frame_buffer) >= self.num_frames and 
                    current_time - last_predict_time > predict_cooldown):
                    
                    word, conf = self.predict()
                    if word and conf > 0.3: 
                        self.current_prediction = word
                        self.confidence = conf
                        last_predict_time = current_time
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = 30 / elapsed
                    start_time = time.time()
                
                # Draw UI
                display_frame = self.draw_ui(frame, fps)
                
                # Show
                cv2.imshow('Hybrid ASL Recognition', display_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    break
                elif key == ord('r'):
                    self.frame_buffer.clear()
                    self.landmark_buffer. clear()
                    self.current_prediction = None
                    self.confidence = 0.0
                    print("✓ Buffers reset")
                elif key == ord(' '):
                    if len(self.frame_buffer) >= self.num_frames:
                        word, conf = self.predict()
                        if word: 
                            self.current_prediction = word
                            self.confidence = conf
                            print(f"Prediction: {word} ({conf*100:.1f}%)")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.landmark_extractor.close()
            print("\n✓ Webcam closed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid ASL Webcam Recognition')
    parser.add_argument('--model', default='best_hybrid_asl_model.pth')
    parser.add_argument('--labels', default='label_mapping.json')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    recognizer = HybridASLWebcam(
        model_path=args.model,
        label_mapping_path=args.labels,
        num_frames=args.num_frames,
        device=args.device
    )
    
    recognizer.run(camera_id=args.camera)


if __name__ == "__main__":
    main()