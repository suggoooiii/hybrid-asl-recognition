def process_video(self, video_path):
    """Process entire video → (30, 162) array"""
    
    cap = cv2.VideoCapture(str(video_path))
    frames_features = []
    
    while cap.isOpened() and len(frames_features) < 30:  # Max 30 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR → RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run MediaPipe Holistic
        results = self.holistic.process(image)
        
        # Extract 162 features
        features = self.extract_landmarks(results)
        frames_features.append(features)
    
    cap.release()
    
    # ═══════════════════════════════════════════════════════════
    # PADDING:  If video < 30 frames, repeat last frame
    # ═══════════════════════════════════════════════════════════
    if len(frames_features) < 30:
        last_frame = frames_features[-1]
        padding = np.tile(last_frame, (30 - len(frames_features), 1))
        frames_features = np.vstack([frames_features, padding])
    
    return np.array(frames_features)  # Shape: (30, 162)