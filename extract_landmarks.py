def extract_landmarks(self, results):
    """Extract landmarks from MediaPipe results"""
    features = []
    
    # ═══════════════════════════════════════════════════════════
    # LEFT HAND (21 landmarks × 3 = 63 features)
    # ═══════════════════════════════════════════════════════════
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks. landmark:
            features.extend([lm.x, lm.y, lm.z])  # Normalized 0-1
    else:
        features.extend([0.0] * 63)  # Padding if hand not detected
    
    # ═══════════════════════════════════════════════════════════
    # RIGHT HAND (21 landmarks × 3 = 63 features)
    # ═══════════════════════════════════════════════════════════
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks. landmark:
            features.extend([lm.x, lm. y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # ═══════════════════════════════════════════════════════════
    # POSE - Upper body only (landmarks 11-22 = 12 × 3 = 36 features)
    # ════════════════════════��══════════════════════════════════
    if results.pose_landmarks:
        for i in range(11, 23):  # Shoulders → Hips
            lm = results.pose_landmarks. landmark[i]
            features. extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 36)
    
    return np.array(features, dtype=np.float32)  # Shape: (162,)