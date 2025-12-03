"""
Stage 0: Multi-Person CV Preprocessing for Experiment 3
Tracks two athletes with relative interaction metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dependencies'))

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from collections import deque

try:
    from ultralytics import YOLO
    from scenedetect import detect, ContentDetector
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Warning: YOLOv8 or dependencies not available. Install with:")
    print("pip3 install ultralytics 'scenedetect[opencv]'")


class AthleteTracker:
    """Tracks two athletes across frames with persistent IDs."""
    
    def __init__(self):
        self.athlete_1_history = deque(maxlen=30)
        self.athlete_2_history = deque(maxlen=30)
        self.initialized = False
        self.frames_since_detection = {'athlete_1': 0, 'athlete_2': 0}
    
    def interpolate_athlete(self, athlete_id: str) -> Optional[Dict]:
        """Interpolate missing athlete using last known position + velocity."""
        history = self.athlete_1_history if athlete_id == 'athlete_1' else self.athlete_2_history
        
        if len(history) < 2:
            return None
        
        # Get last two positions to estimate velocity
        last = history[-1]
        prev = history[-2]
        
        # Calculate velocity
        last_centroid = np.array(last['centroid'])
        prev_centroid = np.array(prev['centroid'])
        velocity = last_centroid - prev_centroid
        
        # Extrapolate position
        new_centroid = tuple(last_centroid + velocity)
        
        # Create interpolated detection with reduced confidence
        interpolated = {
            'keypoints': last['keypoints'],  # Use last known pose
            'bbox': last['bbox'],
            'centroid': new_centroid,
            'confidence': last['confidence'] * 0.6,  # Reduce confidence
            'interpolated': True
        }
        
        return interpolated
    
    def assign_athletes(self, detections: List[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Assign detections to athlete_1 and athlete_2 based on spatial consistency."""
        if len(detections) == 0:
            # Try to interpolate both if missing for < 3 frames
            athlete_1 = None
            athlete_2 = None
            
            if self.frames_since_detection['athlete_1'] < 3:
                athlete_1 = self.interpolate_athlete('athlete_1')
                if athlete_1:
                    self.frames_since_detection['athlete_1'] += 1
            
            if self.frames_since_detection['athlete_2'] < 3:
                athlete_2 = self.interpolate_athlete('athlete_2')
                if athlete_2:
                    self.frames_since_detection['athlete_2'] += 1
            
            return athlete_1, athlete_2
        
        if len(detections) == 1:
            if not self.initialized:
                self.athlete_1_history.append(detections[0])
                self.initialized = True
                self.frames_since_detection['athlete_1'] = 0
                return detections[0], None
            
            dist_to_1 = self._distance_to_history(detections[0], self.athlete_1_history)
            dist_to_2 = self._distance_to_history(detections[0], self.athlete_2_history)
            
            if dist_to_1 < dist_to_2:
                self.athlete_1_history.append(detections[0])
                self.frames_since_detection['athlete_1'] = 0
                
                # Try to interpolate athlete_2
                athlete_2 = None
                if self.frames_since_detection['athlete_2'] < 3:
                    athlete_2 = self.interpolate_athlete('athlete_2')
                    if athlete_2:
                        self.frames_since_detection['athlete_2'] += 1
                
                return detections[0], athlete_2
            else:
                self.athlete_2_history.append(detections[0])
                self.frames_since_detection['athlete_2'] = 0
                
                # Try to interpolate athlete_1
                athlete_1 = None
                if self.frames_since_detection['athlete_1'] < 3:
                    athlete_1 = self.interpolate_athlete('athlete_1')
                    if athlete_1:
                        self.frames_since_detection['athlete_1'] += 1
                
                return athlete_1, detections[0]
        
        if not self.initialized:
            sorted_detections = sorted(detections[:2], key=lambda d: d['centroid'][0])
            self.athlete_1_history.append(sorted_detections[0])
            self.athlete_2_history.append(sorted_detections[1])
            self.initialized = True
            self.frames_since_detection = {'athlete_1': 0, 'athlete_2': 0}
            return sorted_detections[0], sorted_detections[1]
        
        best_assignment = self._find_best_assignment(detections[:2])
        athlete_1 = best_assignment.get('athlete_1')
        athlete_2 = best_assignment.get('athlete_2')
        
        if athlete_1:
            self.athlete_1_history.append(athlete_1)
            self.frames_since_detection['athlete_1'] = 0
        if athlete_2:
            self.athlete_2_history.append(athlete_2)
            self.frames_since_detection['athlete_2'] = 0
        
        return athlete_1, athlete_2
    
    def _distance_to_history(self, detection: Dict, history: deque) -> float:
        if len(history) == 0:
            return float('inf')
        
        centroid = detection['centroid']
        distances = []
        for hist_det in list(history)[-5:]:
            hist_centroid = hist_det['centroid']
            dist = np.sqrt((centroid[0] - hist_centroid[0])**2 + (centroid[1] - hist_centroid[1])**2)
            distances.append(dist)
        
        return np.mean(distances)
    
    def _find_best_assignment(self, detections: List[Dict]) -> Dict:
        if len(detections) < 2:
            if len(detections) == 1:
                dist_to_1 = self._distance_to_history(detections[0], self.athlete_1_history)
                dist_to_2 = self._distance_to_history(detections[0], self.athlete_2_history)
                if dist_to_1 < dist_to_2:
                    return {'athlete_1': detections[0], 'athlete_2': None}
                else:
                    return {'athlete_1': None, 'athlete_2': detections[0]}
            return {'athlete_1': None, 'athlete_2': None}
        
        d1_to_a1 = self._distance_to_history(detections[0], self.athlete_1_history)
        d1_to_a2 = self._distance_to_history(detections[0], self.athlete_2_history)
        d2_to_a1 = self._distance_to_history(detections[1], self.athlete_1_history)
        d2_to_a2 = self._distance_to_history(detections[1], self.athlete_2_history)
        
        cost_1 = d1_to_a1 + d2_to_a2
        cost_2 = d1_to_a2 + d2_to_a1
        
        if cost_1 < cost_2:
            return {'athlete_1': detections[0], 'athlete_2': detections[1]}
        else:
            return {'athlete_1': detections[1], 'athlete_2': detections[0]}


class CVPreprocessor:
    """Multi-person CV preprocessing for BJJ video analysis."""
    
    def __init__(self, video_path: str, cache_dir: str = None):
        self.video_path = video_path
        self.cache_dir = cache_dir or "outputs/experiment3"
        self.cache_file = None
        
        # Initialize YOLO11-Pose model
        if DEPENDENCIES_AVAILABLE:
            print("Initializing YOLO11-Pose model...")
            try:
                # Note: MPS has known issues with Pose models, using CPU
                # See: https://github.com/ultralytics/ultralytics/issues/4031
                self.device = 'cpu'
                print(f"  Using device: {self.device}")
                
                # Use YOLO11n-pose (nano) for speed
                self.pose_model = YOLO('yolo11n-pose.pt')
                
                # Optimize for inference
                self.pose_model.overrides['verbose'] = False
                self.pose_model.overrides['half'] = False  # FP16 not stable on CPU
                
                print("✓ YOLO11-Pose model loaded")
            except Exception as e:
                print(f"Warning: Failed to load YOLOv8-Pose: {e}")
                print("Falling back to mock mode")
                self.pose_model = None
        
        # Initialize athlete tracker
        self.athlete_tracker = AthleteTracker()
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {Path(video_path).name}")
        print(f"Duration: {self.duration:.1f}s ({self.duration/60:.1f} min)")
        print(f"FPS: {self.fps:.1f}, Total frames: {self.total_frames}")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
    
    def check_cache(self, video_name: str) -> bool:
        """Check if CV cache exists for this video."""
        cache_path = Path(self.cache_dir) / f"{video_name}_cv_cache.json"
        if cache_path.exists():
            self.cache_file = cache_path
            return True
        return False
    
    def load_cache(self) -> Dict:
        """Load cached CV analysis."""
        if self.cache_file and self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def detect_scenes(self) -> List[Tuple[float, float]]:
        """Detect scene boundaries using PySceneDetect."""
        print("\n[Stage 0.1] Scene detection...")
        
        if not DEPENDENCIES_AVAILABLE:
            print("  Skipping (dependencies not available)")
            return []
        
        try:
            scene_list = detect(self.video_path, ContentDetector(threshold=27.0))
            scenes = [(scene[0].get_seconds(), scene[1].get_seconds()) 
                     for scene in scene_list]
            print(f"  Found {len(scenes)} scenes")
            return scenes
        except Exception as e:
            print(f"  Error: {e}")
            return []
    
    def detect_multi_person_poses(self, frame: np.ndarray) -> List[Dict]:
        """Detect poses for multiple people in frame using YOLO11-Pose."""
        if not DEPENDENCIES_AVAILABLE or not hasattr(self, 'pose_model') or self.pose_model is None:
            return []
        
        try:
            # Run YOLO11-Pose inference with optimizations
            results = self.pose_model.predict(
                frame, 
                verbose=False, 
                conf=0.3,
                device=self.device,
                half=False,  # No FP16 on CPU
                max_det=2,  # Only detect top 2 people (BJJ fighters)
                classes=[0]  # Only person class
            )
            
            if len(results) == 0 or results[0].keypoints is None:
                return []
            
            detections = []
            result = results[0]  # First result (single image)
            
            # Get all detected people (limit to top 2 for BJJ - only need the fighters)
            if result.keypoints.data is not None:
                num_people = min(len(result.keypoints.data), 2)  # Only process top 2
                for i in range(num_people):
                    # Get keypoints and confidence scores
                    kpts = result.keypoints.data[i]  # Shape: (17, 3) - x, y, confidence
                    bbox = result.boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                    
                    # Calculate centroid from visible keypoints
                    visible_mask = kpts[:, 2] > 0.3
                    if not visible_mask.any():
                        continue
                    
                    visible_kpts = kpts[visible_mask]
                    centroid_x = float(visible_kpts[:, 0].mean()) / self.frame_width
                    centroid_y = float(visible_kpts[:, 1].mean()) / self.frame_height
                    
                    # Normalize bbox
                    x_min, y_min, x_max, y_max = bbox
                    bbox_norm = (
                        x_min / self.frame_width,
                        y_min / self.frame_height,
                        (x_max - x_min) / self.frame_width,
                        (y_max - y_min) / self.frame_height
                    )
                    
                    # Convert keypoints to normalized format
                    keypoints_norm = []
                    for kpt in kpts:
                        keypoints_norm.append((
                            float(kpt[0]) / self.frame_width,
                            float(kpt[1]) / self.frame_height,
                            0.0,  # z coordinate (not available)
                            float(kpt[2])  # confidence
                        ))
                    
                    detection = {
                        'keypoints': keypoints_norm,
                        'bbox': bbox_norm,
                        'centroid': (centroid_x, centroid_y),
                        'confidence': float(kpts[:, 2].mean())
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error in multi-person detection: {e}")
            return []
    
    def calculate_relative_metrics(self, athlete_1: Optional[Dict], athlete_2: Optional[Dict]) -> Dict:
        """Calculate relative interaction metrics between two athletes."""
        if not athlete_1 or not athlete_2:
            return {
                'centroid_distance': 1.0,
                'bbox_overlap': 0.0,
                'limb_entanglement': 0,
                'contact_intensity': 0.0,
                'relative_velocity': 0.0,
                'dominant_athlete': 0
            }
        
        # Centroid distance (normalized)
        c1 = np.array(athlete_1['centroid'])
        c2 = np.array(athlete_2['centroid'])
        distance = np.linalg.norm(c1 - c2)
        
        # Bbox overlap (IOU)
        bbox1 = athlete_1['bbox']
        bbox2 = athlete_2['bbox']
        overlap = self._calculate_bbox_iou(bbox1, bbox2)
        
        # Limb entanglement (count overlapping limbs)
        entanglement = self._calculate_limb_entanglement(athlete_1, athlete_2)
        
        # Contact intensity (based on overlap and distance)
        contact = min(overlap * 2, 1.0) if distance < 0.3 else 0.0
        
        # Dominant athlete (who's higher in frame)
        dominant = 1 if c1[1] < c2[1] else (2 if c2[1] < c1[1] else 0)
        
        return {
            'centroid_distance': round(float(distance), 4),
            'bbox_overlap': round(float(overlap), 4),
            'limb_entanglement': int(entanglement),
            'contact_intensity': round(float(contact), 4),
            'relative_velocity': 0.0,  # Will be calculated from frame-to-frame
            'dominant_athlete': dominant
        }
    
    def _calculate_bbox_iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1_min, y1_min, w1, h1 = bbox1
        x2_min, y2_min, w2, h2 = bbox2
        
        x1_max, y1_max = x1_min + w1, y1_min + h1
        x2_max, y2_max = x2_min + w2, y2_min + h2
        
        # Intersection
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0
        
        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        
        # Union
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _calculate_limb_entanglement(self, athlete_1: Dict, athlete_2: Dict) -> int:
        """Count how many limbs are overlapping/entangled (COCO format)."""
        keypoints_1 = athlete_1['keypoints']
        keypoints_2 = athlete_2['keypoints']
        
        # COCO key limb indices: 9=left_wrist, 10=right_wrist, 7=left_elbow, 8=right_elbow,
        # 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
        limb_indices = [9, 10, 7, 8, 13, 14, 15, 16]
        
        entanglement = 0
        threshold = 0.1  # Distance threshold for "entangled" (normalized coordinates)
        
        for idx in limb_indices:
            if idx < len(keypoints_1) and idx < len(keypoints_2):
                # Check visibility
                if keypoints_1[idx][3] < 0.3 or keypoints_2[idx][3] < 0.3:
                    continue
                
                p1 = np.array(keypoints_1[idx][:2])
                p2 = np.array(keypoints_2[idx][:2])
                dist = np.linalg.norm(p1 - p2)
                if dist < threshold:
                    entanglement += 1
        
        return entanglement
    
    def calculate_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple:
        """Calculate optical flow magnitude and spatial variance."""
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Overall motion magnitude
        flow_magnitude = np.mean(magnitude)
        
        # Spatial variance: how spread out is the motion?
        # High variance = motion across frame (scrambles, takedowns)
        # Low variance = localized motion (guard work, submissions)
        flow_std = np.std(magnitude)
        spatial_variance = flow_std / (flow_magnitude + 1e-6)  # Normalize by magnitude
        
        return flow_magnitude, spatial_variance
    
    def calculate_standing_probability(self, keypoints) -> float:
        """Calculate probability that athlete is standing vs ground (COCO format)."""
        if not keypoints or len(keypoints) < 17:
            return 0.5  # Unknown
        
        # COCO keypoint indices: 11=left_hip, 12=right_hip, 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        # Check visibility
        if left_hip[3] < 0.3 or right_hip[3] < 0.3:
            return 0.5
        
        # Hip height (normalized, 0=bottom, 1=top of frame)
        avg_hip_y = (left_hip[1] + right_hip[1]) / 2
        
        # Knee angle (straighter = more likely standing)
        def calculate_angle(a, b, c):
            """Calculate angle at point b."""
            if a[3] < 0.3 or b[3] < 0.3 or c[3] < 0.3:
                return 90  # Default if not visible
            ba = np.array([a[0] - b[0], a[1] - b[1]])
            bc = np.array([c[0] - b[0], c[1] - b[1]])
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            return np.degrees(angle)
        
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        # Standing heuristics:
        # - Hips in upper half of frame (y < 0.5)
        # - Knees relatively straight (angle > 140°)
        hip_score = 1.0 - avg_hip_y  # Higher y = lower in frame = less likely standing
        knee_score = min(avg_knee_angle / 180.0, 1.0)  # Normalize to 0-1
        
        standing_prob = 0.6 * hip_score + 0.4 * knee_score
        return np.clip(standing_prob, 0.0, 1.0)
    
    def calculate_scene_complexity(self, gray: np.ndarray) -> float:
        """Calculate scene complexity using edge detection.
        High complexity = many edges = scrambles, multiple limbs moving
        Low complexity = few edges = static positions
        """
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density (percentage of edge pixels)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Normalize to 0-1 range (typical range: 0.01-0.15)
        complexity = min(edge_density / 0.15, 1.0)
        
        return complexity
    
    def analyze_frame(self, frame: np.ndarray, prev_athletes: Tuple[Optional[Dict], Optional[Dict]] = None) -> Tuple[Dict, Tuple[Optional[Dict], Optional[Dict]]]:
        """Analyze frame with multi-person tracking and relative metrics."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect multiple people
        detections = self.detect_multi_person_poses(frame)
        
        # Debug first frame
        if not hasattr(self, '_first_detection_logged'):
            print(f"  First detection: {len(detections)} people found")
            self._first_detection_logged = True
        
        # Assign to athlete IDs
        athlete_1, athlete_2 = self.athlete_tracker.assign_athletes(detections)
        
        # Calculate standing probability per athlete
        athlete_1_standing = 0.5
        athlete_2_standing = 0.5
        
        if athlete_1 and DEPENDENCIES_AVAILABLE:
            keypoints = athlete_1['keypoints']
            if len(keypoints) >= 17:
                athlete_1_standing = self.calculate_standing_probability(keypoints)
        
        if athlete_2 and DEPENDENCIES_AVAILABLE:
            keypoints = athlete_2['keypoints']
            if len(keypoints) >= 17:
                athlete_2_standing = self.calculate_standing_probability(keypoints)
        
        # Calculate relative interaction metrics
        interaction = self.calculate_relative_metrics(athlete_1, athlete_2)
        
        # Calculate relative velocity if we have previous frame
        if prev_athletes and prev_athletes[0] and prev_athletes[1] and athlete_1 and athlete_2:
            prev_c1 = np.array(prev_athletes[0]['centroid'])
            prev_c2 = np.array(prev_athletes[1]['centroid'])
            curr_c1 = np.array(athlete_1['centroid'])
            curr_c2 = np.array(athlete_2['centroid'])
            
            vel_1 = np.linalg.norm(curr_c1 - prev_c1)
            vel_2 = np.linalg.norm(curr_c2 - prev_c2)
            interaction['relative_velocity'] = round(abs(vel_1 - vel_2), 4)
        
        # Calculate detection quality
        both_detected = athlete_1 is not None and athlete_2 is not None
        avg_confidence = 0.0
        if both_detected:
            avg_confidence = (athlete_1['confidence'] + athlete_2['confidence']) / 2
        elif athlete_1:
            avg_confidence = athlete_1['confidence']
        elif athlete_2:
            avg_confidence = athlete_2['confidence']
        
        interpolated = (athlete_1 and athlete_1.get('interpolated', False)) or \
                      (athlete_2 and athlete_2.get('interpolated', False))
        
        result = {
            'athlete_1': {
                'detected': athlete_1 is not None,
                'confidence': athlete_1['confidence'] if athlete_1 else 0.0,
                'centroid': athlete_1['centroid'] if athlete_1 else (0.0, 0.0),
                'standing_probability': athlete_1_standing,
                'interpolated': athlete_1.get('interpolated', False) if athlete_1 else False
            },
            'athlete_2': {
                'detected': athlete_2 is not None,
                'confidence': athlete_2['confidence'] if athlete_2 else 0.0,
                'centroid': athlete_2['centroid'] if athlete_2 else (0.0, 0.0),
                'standing_probability': athlete_2_standing,
                'interpolated': athlete_2.get('interpolated', False) if athlete_2 else False
            },
            'interaction': interaction,
            'detection_quality': {
                'both_detected': both_detected,
                'avg_confidence': round(avg_confidence, 3),
                'interpolated': interpolated
            }
        }
        
        return result, (athlete_1, athlete_2)
    
    def classify_segment_analysis_mode(self, segment_metrics: List[Dict]) -> str:
        """
        Classify a segment as ECONOMICAL or FULL analysis based on metrics.
        
        ECONOMICAL: Confidently boring (standing/circling, stalling, resets)
        FULL: Everything else (normal grappling, transitions, high action)
        
        Args:
            segment_metrics: List of per-frame metrics for this segment
        
        Returns:
            'ECONOMICAL' or 'FULL'
        """
        if not segment_metrics:
            return 'FULL'
        
        # Calculate segment-level statistics
        total_frames = len(segment_metrics)
        
        # Detection quality check
        both_detected_frames = sum(1 for m in segment_metrics if m['detection_quality']['both_detected'])
        avg_confidence = np.mean([m['detection_quality']['avg_confidence'] for m in segment_metrics])
        interpolated_frames = sum(1 for m in segment_metrics if m['detection_quality']['interpolated'])
        
        # Must have good detection quality (>70% both detected, avg conf > 0.6)
        if both_detected_frames / total_frames < 0.7 or avg_confidence < 0.6:
            return 'FULL'  # Uncertain data = need full analysis
        
        # Check for boring indicators across segment
        avg_distance = np.mean([m['interaction']['centroid_distance'] for m in segment_metrics])
        avg_entanglement = np.mean([m['interaction']['limb_entanglement'] for m in segment_metrics])
        avg_contact = np.mean([m['interaction']['contact_intensity'] for m in segment_metrics])
        avg_velocity = np.mean([m['interaction']['relative_velocity'] for m in segment_metrics])
        avg_action = np.mean([m['action_score'] for m in segment_metrics])
        
        # Boring criteria (all must be true for ECONOMICAL)
        boring_criteria = [
            avg_distance > 0.4,        # Athletes far apart
            avg_entanglement < 1.0,    # Minimal limb contact
            avg_contact < 0.15,        # Low contact intensity
            avg_velocity < 0.08,       # Low movement
            avg_action < 0.15          # Low action score
        ]
        
        # Need at least 4 of 5 boring criteria
        if sum(boring_criteria) >= 4:
            return 'ECONOMICAL'
        
        return 'FULL'
    
    def calculate_action_score(self, interaction: Dict, prev_interaction: Optional[Dict] = None) -> float:
        """
        Calculate composite action score from interaction metrics.
        
        NEW Formula: action_score = 0.25×relative_velocity + 0.25×limb_entanglement + 
                                    0.20×contact_change + 0.15×distance_change + 0.15×overlap
        
        Components:
        - relative_velocity: How fast athletes move relative to each other
        - limb_entanglement: Number of overlapping limbs (0-8)
        - contact_change: Rate of contact intensity change
        - distance_change: Rate of distance change
        - bbox_overlap: How much athletes overlap
        """
        # Relative velocity (already 0-1 normalized)
        rel_vel = min(interaction.get('relative_velocity', 0.0), 1.0)
        
        # Limb entanglement (normalize 0-8 to 0-1)
        entanglement = min(interaction.get('limb_entanglement', 0) / 8.0, 1.0)
        
        # Contact change rate (frame-to-frame)
        contact_change = 0.0
        if prev_interaction:
            prev_contact = prev_interaction.get('contact_intensity', 0.0)
            curr_contact = interaction.get('contact_intensity', 0.0)
            contact_change = abs(curr_contact - prev_contact)
        
        # Distance change rate (frame-to-frame)
        distance_change = 0.0
        if prev_interaction:
            prev_dist = prev_interaction.get('centroid_distance', 1.0)
            curr_dist = interaction.get('centroid_distance', 1.0)
            distance_change = abs(curr_dist - prev_dist)
        
        # Bbox overlap (already 0-1 normalized)
        overlap = interaction.get('bbox_overlap', 0.0)
        
        # Weighted combination
        action_score = (
            0.25 * rel_vel +
            0.25 * entanglement +
            0.20 * contact_change +
            0.15 * distance_change +
            0.15 * overlap
        )
        
        return np.clip(action_score, 0.0, 1.0)
    
    def process_video(self, sample_rate: int = 1) -> Dict:
        """
        Process entire video with multi-person tracking and interaction metrics.
        
        Args:
            sample_rate: Process every Nth frame (1 = every frame, 2 = every other frame)
        """
        print(f"\n[Stage 0.2] Multi-person tracking and interaction analysis...")
        
        # Detect scenes first
        scenes = self.detect_scenes()
        
        # Process frames
        per_second_metrics = []
        prev_athletes = None
        prev_interaction = None
        
        frame_idx = 0
        processed_frames = 0
        start_time = time.time()
        
        # Progress tracking
        total_to_process = self.total_frames // sample_rate
        last_progress = 0
        
        print(f"  Processing {total_to_process} frames...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue
            
            # Debug: Print first frame
            if processed_frames == 0:
                print(f"  Starting frame processing (first inference may take 10-15s)...")
            
            # Analyze frame with multi-person tracking
            try:
                metrics, curr_athletes = self.analyze_frame(frame, prev_athletes)
            except Exception as e:
                print(f"  Error analyzing frame {frame_idx}: {e}")
                import traceback
                traceback.print_exc()
                frame_idx += 1
                continue
            
            # Calculate action score from interaction
            interaction = metrics['interaction']
            action_score = self.calculate_action_score(interaction, prev_interaction)
            
            # Store per-second metrics
            timestamp = frame_idx / self.fps
            per_second_metrics.append({
                'timestamp': round(timestamp, 2),
                'frame': frame_idx,
                'athlete_1': {
                    'detected': metrics['athlete_1']['detected'],
                    'confidence': round(metrics['athlete_1']['confidence'], 4),
                    'standing_probability': round(metrics['athlete_1']['standing_probability'], 4),
                    'interpolated': metrics['athlete_1']['interpolated']
                },
                'athlete_2': {
                    'detected': metrics['athlete_2']['detected'],
                    'confidence': round(metrics['athlete_2']['confidence'], 4),
                    'standing_probability': round(metrics['athlete_2']['standing_probability'], 4),
                    'interpolated': metrics['athlete_2']['interpolated']
                },
                'interaction': {
                    'centroid_distance': interaction['centroid_distance'],
                    'bbox_overlap': interaction['bbox_overlap'],
                    'limb_entanglement': interaction['limb_entanglement'],
                    'contact_intensity': interaction['contact_intensity'],
                    'relative_velocity': interaction['relative_velocity'],
                    'dominant_athlete': interaction['dominant_athlete']
                },
                'action_score': round(action_score, 4),
                'detection_quality': metrics['detection_quality']
            })
            
            # Update for next iteration
            prev_athletes = curr_athletes
            prev_interaction = interaction
            processed_frames += 1
            frame_idx += 1
            
            # Progress update (every 5% for more feedback)
            progress = int((processed_frames / total_to_process) * 100)
            if progress >= last_progress + 5:
                elapsed = time.time() - start_time
                fps = processed_frames / elapsed if elapsed > 0 else 0
                eta = (total_to_process - processed_frames) / fps if fps > 0 else 0
                print(f"  Progress: {progress}% ({processed_frames}/{total_to_process} frames, "
                      f"{fps:.1f} fps, ETA: {eta:.0f}s)", flush=True)
                last_progress = progress
        
        self.cap.release()
        
        elapsed = time.time() - start_time
        print(f"  Completed: {processed_frames} frames in {elapsed:.1f}s ({processed_frames/elapsed:.1f} fps)")
        
        # Calculate summary statistics
        action_scores = [m['action_score'] for m in per_second_metrics]
        entanglements = [m['interaction']['limb_entanglement'] for m in per_second_metrics]
        distances = [m['interaction']['centroid_distance'] for m in per_second_metrics]
        overlaps = [m['interaction']['bbox_overlap'] for m in per_second_metrics]
        contacts = [m['interaction']['contact_intensity'] for m in per_second_metrics]
        
        athlete_1_detected = sum(1 for m in per_second_metrics if m['athlete_1']['detected'])
        athlete_2_detected = sum(1 for m in per_second_metrics if m['athlete_2']['detected'])
        both_detected = sum(1 for m in per_second_metrics if m['athlete_1']['detected'] and m['athlete_2']['detected'])
        
        # Interpolation stats
        interpolated_frames = sum(1 for m in per_second_metrics if m['detection_quality']['interpolated'])
        
        athlete_1_standing = [m['athlete_1']['standing_probability'] for m in per_second_metrics if m['athlete_1']['detected']]
        athlete_2_standing = [m['athlete_2']['standing_probability'] for m in per_second_metrics if m['athlete_2']['detected']]
        
        # Action score distribution
        low_action = sum(1 for s in action_scores if s < 0.2)
        medium_action = sum(1 for s in action_scores if 0.2 <= s < 0.6)
        high_action = sum(1 for s in action_scores if s >= 0.6)
        
        # Entanglement distribution
        separated = sum(1 for e in entanglements if e <= 2)
        engaged = sum(1 for e in entanglements if 3 <= e <= 5)
        scrambling = sum(1 for e in entanglements if e >= 6)
        
        # Dominance tracking
        athlete_1_dominant = sum(1 for m in per_second_metrics if m['interaction']['dominant_athlete'] == 1)
        athlete_2_dominant = sum(1 for m in per_second_metrics if m['interaction']['dominant_athlete'] == 2)
        neutral = sum(1 for m in per_second_metrics if m['interaction']['dominant_athlete'] == 0)
        
        summary = {
            'avg_action': round(np.mean(action_scores), 4),
            'max_action': round(np.max(action_scores), 4),
            'min_action': round(np.min(action_scores), 4),
            'std_action': round(np.std(action_scores), 4),
            'action_distribution': {
                'low': low_action,
                'medium': medium_action,
                'high': high_action,
                'low_pct': round(low_action / len(action_scores) * 100, 1),
                'medium_pct': round(medium_action / len(action_scores) * 100, 1),
                'high_pct': round(high_action / len(action_scores) * 100, 1)
            },
            'athlete_detection': {
                'athlete_1_rate': round(athlete_1_detected / len(per_second_metrics), 4),
                'athlete_2_rate': round(athlete_2_detected / len(per_second_metrics), 4),
                'both_rate': round(both_detected / len(per_second_metrics), 4),
                'interpolated_rate': round(interpolated_frames / len(per_second_metrics), 4),
                'interpolated_count': interpolated_frames
            },
            'athlete_standing': {
                'athlete_1_avg': round(np.mean(athlete_1_standing), 4) if athlete_1_standing else 0.5,
                'athlete_2_avg': round(np.mean(athlete_2_standing), 4) if athlete_2_standing else 0.5
            },
            'interaction': {
                'avg_entanglement': round(np.mean(entanglements), 2),
                'avg_distance': round(np.mean(distances), 4),
                'avg_overlap': round(np.mean(overlaps), 4),
                'avg_contact': round(np.mean(contacts), 4),
                'entanglement_distribution': {
                    'separated': separated,
                    'engaged': engaged,
                    'scrambling': scrambling,
                    'separated_pct': round(separated / len(entanglements) * 100, 1),
                    'engaged_pct': round(engaged / len(entanglements) * 100, 1),
                    'scrambling_pct': round(scrambling / len(entanglements) * 100, 1)
                }
            },
            'dominance': {
                'athlete_1_frames': athlete_1_dominant,
                'athlete_2_frames': athlete_2_dominant,
                'neutral_frames': neutral,
                'athlete_1_pct': round(athlete_1_dominant / len(per_second_metrics) * 100, 1),
                'athlete_2_pct': round(athlete_2_dominant / len(per_second_metrics) * 100, 1),
                'neutral_pct': round(neutral / len(per_second_metrics) * 100, 1)
            }
        }
        
        print(f"\n  Summary:")
        print(f"    Avg action score: {summary['avg_action']:.3f} (std: {summary['std_action']:.3f})")
        print(f"    Action range: {summary['min_action']:.3f} - {summary['max_action']:.3f}")
        print(f"    Distribution: Low {summary['action_distribution']['low_pct']:.1f}%, Medium {summary['action_distribution']['medium_pct']:.1f}%, High {summary['action_distribution']['high_pct']:.1f}%")
        print(f"    Avg entanglement: {summary['interaction']['avg_entanglement']:.1f} limbs")
        print(f"    Entanglement: Separated {summary['interaction']['entanglement_distribution']['separated_pct']:.1f}%, Engaged {summary['interaction']['entanglement_distribution']['engaged_pct']:.1f}%, Scrambling {summary['interaction']['entanglement_distribution']['scrambling_pct']:.1f}%")
        print(f"    Both athletes detected: {summary['athlete_detection']['both_rate']:.1%}")
        print(f"    Dominance: A1 {summary['dominance']['athlete_1_pct']:.1f}%, A2 {summary['dominance']['athlete_2_pct']:.1f}%, Neutral {summary['dominance']['neutral_pct']:.1f}%")
        
        return {
            'video_path': self.video_path,
            'duration': self.duration,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'resolution': f"{self.frame_width}x{self.frame_height}",
            'scenes': scenes,
            'per_second_metrics': per_second_metrics,
            'summary': summary,
            'formula': 'action_score = 0.25×relative_velocity + 0.25×limb_entanglement + 0.20×contact_change + 0.15×distance_change + 0.15×overlap',
            'weights': {'relative_velocity': 0.25, 'limb_entanglement': 0.25, 'contact_change': 0.20, 'distance_change': 0.15, 'overlap': 0.15},
            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def save_cache(self, data: Dict, video_name: str):
        """Save CV analysis to cache file."""
        cache_path = Path(self.cache_dir) / f"{video_name}_cv_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python native types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        data = convert_numpy(data)
        
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n[Stage 0] Saved CV cache: {cache_path}")
        self.cache_file = cache_path
    
    def create_smart_segments(self, per_second_metrics: List[Dict], target_segments: int = 60) -> List[Dict]:
        """
        Create intelligent segment suggestions based on CV signals.
        Target: 30-100 segments for the full match (default 60).
        
        Strategy: Create baseline time-based segments, then split on major transitions.
        """
        if not per_second_metrics:
            return []
        
        # Sample to get actual per-second data (metrics are per-frame at 30fps)
        fps = 30
        sampled_metrics = [per_second_metrics[i] for i in range(0, len(per_second_metrics), fps) if i < len(per_second_metrics)]
        
        total_duration = len(sampled_metrics)  # Now in seconds
        baseline_segment_length = total_duration // target_segments  # ~16s for 60 segments in 16min
        
        segments = []
        current_start = 0
        prev_metrics = None
        
        for i, metrics in enumerate(sampled_metrics):
            should_segment = False
            reason = []
            
            if i == 0:
                prev_metrics = metrics
                continue
            
            segment_length = i - current_start
            
            # Calculate average action for current segment
            avg_action = np.mean([m['action_score'] for m in sampled_metrics[current_start:i]])
            
            # Rule 1: Check for major transitions first (minimum 8s segment)
            if segment_length >= 8:
                # Standing <-> Ground transition
                if metrics.get('athlete_1') and prev_metrics.get('athlete_1'):
                    curr_standing = metrics['athlete_1'].get('standing_probability', 0.5)
                    prev_standing = prev_metrics['athlete_1'].get('standing_probability', 0.5)
                    
                    if (prev_standing > 0.7 and curr_standing < 0.3):
                        should_segment = True
                        reason.append('takedown')
                    elif (prev_standing < 0.3 and curr_standing > 0.7):
                        should_segment = True
                        reason.append('standup')
                
                # High action spike (submission/scramble)
                if metrics['action_score'] > 0.5 and prev_metrics['action_score'] < 0.2:
                    should_segment = True
                    reason.append('action_spike')
            
            # Rule 2: Time-based segmentation
            if not should_segment:
                if avg_action < 0.10:  # Very static - longer segments
                    if segment_length >= baseline_segment_length * 1.5:  # ~24s
                        should_segment = True
                        reason.append('static_segment')
                elif avg_action < 0.30:  # Medium - baseline segments
                    if segment_length >= baseline_segment_length:  # ~16s
                        should_segment = True
                        reason.append('medium_segment')
                else:  # Active - shorter segments
                    if segment_length >= baseline_segment_length * 0.7:  # ~11s
                        should_segment = True
                        reason.append('active_segment')
            
            if should_segment:
                # Calculate segment statistics
                segment_metrics = sampled_metrics[current_start:i]
                avg_action = np.mean([m['action_score'] for m in segment_metrics])
                max_action = max([m['action_score'] for m in segment_metrics])
                
                # Get standing probabilities for position hints
                standing_probs = []
                for m in segment_metrics:
                    if m.get('athlete_1'):
                        standing_probs.append(m['athlete_1'].get('standing_probability', 0.5))
                avg_standing = np.mean(standing_probs) if standing_probs else 0.5
                
                segments.append({
                    'start_sec': current_start,
                    'end_sec': i,
                    'start_time': self._format_timestamp(current_start),
                    'end_time': self._format_timestamp(i),
                    'duration_sec': i - current_start,
                    'cv_avg_action': round(float(avg_action), 4),
                    'cv_max_action': round(float(max_action), 4),
                    'cv_avg_standing': round(float(avg_standing), 4),
                    'cv_signals': reason
                })
                current_start = i
            
            prev_metrics = metrics
        
        # Add final segment
        if current_start < len(sampled_metrics):
            segment_metrics = sampled_metrics[current_start:]
            avg_action = np.mean([m['action_score'] for m in segment_metrics])
            max_action = max([m['action_score'] for m in segment_metrics])
            standing_probs = []
            for m in segment_metrics:
                if m.get('athlete_1'):
                    standing_probs.append(m['athlete_1'].get('standing_probability', 0.5))
            avg_standing = np.mean(standing_probs) if standing_probs else 0.5
            
            segments.append({
                'start_sec': current_start,
                'end_sec': len(sampled_metrics),
                'start_time': self._format_timestamp(current_start),
                'end_time': self._format_timestamp(len(sampled_metrics)),
                'duration_sec': len(sampled_metrics) - current_start,
                'cv_avg_action': round(float(avg_action), 4),
                'cv_max_action': round(float(max_action), 4),
                'cv_avg_standing': round(float(avg_standing), 4),
                'cv_signals': ['end_of_video']
            })
        
        print(f"\n[Stage 0] Created {len(segments)} suggested segments")
        print(f"  Target range: 30-100 segments")
        print(f"  Average segment duration: {np.mean([s['duration_sec'] for s in segments]):.1f}s")
        
        return segments
    
    def _format_timestamp(self, seconds: int) -> str:
        """Format seconds as MM:SS timestamp."""
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}:{secs:02d}"


def add_segments_to_cache(cache_path: str) -> Dict:
    """
    Add smart segments to existing CV cache without reprocessing video.
    Useful for updating segmentation logic without re-running CV.
    """
    print(f"\n[Stage 0] Adding smart segments to existing cache...")
    
    with open(cache_path, 'r') as f:
        data = json.load(f)
    
    # Create preprocessor instance (just for the segmentation method)
    preprocessor = CVPreprocessor(data['video_path'])
    
    # Generate segments
    segments = preprocessor.create_smart_segments(data['per_second_metrics'])
    data['suggested_segments'] = segments
    
    # Save updated cache
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Updated cache: {cache_path}")
    return data


def run_cv_preprocessing(video_path: str, cache_dir: str = None, force: bool = False) -> Dict:
    """
    Run CV preprocessing on video.
    
    Args:
        video_path: Path to video file
        cache_dir: Directory to store cache
        force: Force reprocessing even if cache exists
    
    Returns:
        CV analysis data
    """
    video_name = Path(video_path).stem
    
    preprocessor = CVPreprocessor(video_path, cache_dir)
    
    # Check cache
    if not force and preprocessor.check_cache(video_name):
        print(f"\n[Stage 0] Loading cached CV analysis...")
        data = preprocessor.load_cache()
        print(f"  Loaded from: {preprocessor.cache_file}")
        print(f"  Processed at: {data.get('processed_at', 'unknown')}")
        return data
    
    # Process video
    print(f"\n[Stage 0] Starting CV preprocessing...")
    data = preprocessor.process_video(sample_rate=1)
    
    # Create smart segments
    print(f"\n[Stage 0] Creating smart segments...")
    segments = preprocessor.create_smart_segments(data['per_second_metrics'])
    data['suggested_segments'] = segments
    
    # Save cache
    preprocessor.save_cache(data, video_name)
    
    return data


def generate_compressed_5s_checkpoints(cv_cache_path: str, output_dir: str = "outputs/experiment4.0") -> Dict:
    """
    Generate compressed 5-second CV checkpoints for Experiment 4 Stage 1.
    
    Takes the full granular CV data from Experiment 3 and creates:
    - 5-second interval checkpoints
    - Compressed format (~45 chars per checkpoint)
    - Both human-readable (.md) and structured (.json) outputs
    
    Args:
        cv_cache_path: Path to Experiment 3 CV cache JSON
        output_dir: Directory to save Experiment 4 outputs
    
    Returns:
        Dictionary with compressed checkpoints
    """
    print(f"\n[Stage 0 → Exp4] Generating compressed 5s checkpoints...")
    print(f"  Input: {cv_cache_path}")
    
    # Load Experiment 3 CV cache
    with open(cv_cache_path, 'r') as f:
        cv_data = json.load(f)
    
    metrics = cv_data['per_second_metrics']
    duration = cv_data['duration']
    
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  Total metrics: {len(metrics)}")
    
    # Sample every 5 seconds
    checkpoints = []
    checkpoint_interval = 5  # seconds
    
    for i in range(0, int(duration), checkpoint_interval):
        # Find metric closest to this timestamp
        target_time = i
        closest_metric = min(metrics, key=lambda m: abs(m['timestamp'] - target_time))
        
        # Extract data
        a1 = closest_metric['athlete_1']
        a2 = closest_metric['athlete_2']
        interaction = closest_metric['interaction']
        
        # Classify standing/ground/mixed
        def classify_standing(prob):
            if prob > 0.7:
                return 'S'
            elif prob < 0.3:
                return 'G'
            else:
                return 'M'
        
        a1_class = classify_standing(a1['standing_probability']) if a1['detected'] else '?'
        a2_class = classify_standing(a2['standing_probability']) if a2['detected'] else '?'
        
        # Classify contact
        contact = interaction['contact_intensity']
        if contact > 0.6:
            contact_class = 'H'
        elif contact > 0.3:
            contact_class = 'M'
        else:
            contact_class = 'L'
        
        # Determine dominant athlete
        dominant = interaction['dominant_athlete']
        if dominant == 1:
            dom_class = 'A1'
        elif dominant == 2:
            dom_class = 'A2'
        else:
            dom_class = 'N'
        
        # Format timestamp
        mm = i // 60
        ss = i % 60
        timestamp = f"{mm}:{ss:02d}"
        
        # Compressed format: "0:00 S/S c:L e:0.2 A1"
        compressed = f"{timestamp} {a1_class}/{a2_class} c:{contact_class} e:{interaction['limb_entanglement']:.1f} {dom_class}"
        
        # Full format for JSON
        checkpoint = {
            'timestamp': timestamp,
            'time_sec': i,
            'athlete_1': {
                'standing': a1_class,
                'standing_prob': round(a1['standing_probability'], 2) if a1['detected'] else None,
                'detected': a1['detected']
            },
            'athlete_2': {
                'standing': a2_class,
                'standing_prob': round(a2['standing_probability'], 2) if a2['detected'] else None,
                'detected': a2['detected']
            },
            'interaction': {
                'contact': contact_class,
                'contact_intensity': round(contact, 2),
                'entanglement': round(interaction['limb_entanglement'], 1),
                'distance': round(interaction['centroid_distance'], 2),
                'dominant': dom_class
            },
            'action_score': round(closest_metric['action_score'], 3),
            'compressed': compressed
        }
        
        checkpoints.append(checkpoint)
    
    print(f"  Generated {len(checkpoints)} checkpoints (every {checkpoint_interval}s)")
    
    # Calculate token estimate
    total_chars = sum(len(cp['compressed']) for cp in checkpoints)
    estimated_tokens = total_chars // 4
    print(f"  Compressed format: {total_chars} chars (~{estimated_tokens} tokens)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate markdown output
    md_lines = [
        "# Compressed 5-Second CV Checkpoints for Experiment 4",
        "",
        f"**Video Duration:** {duration:.1f}s ({duration/60:.1f} min)",
        f"**Checkpoints:** {len(checkpoints)} (every {checkpoint_interval}s)",
        f"**Token Cost:** ~{estimated_tokens} tokens",
        "",
        "**Legend:**",
        "- S=Standing, G=Ground, M=Mixed, ?=Not detected",
        "- c=Contact (L=Low, M=Medium, H=High)",
        "- e=Entanglement (limbs)",
        "- A1/A2/N=Dominant athlete (Athlete1/Athlete2/Neutral)",
        "",
        "**Format:** `timestamp athlete1/athlete2 c:contact e:entanglement dominant`",
        "",
        "---",
        ""
    ]
    
    # Add checkpoints in rows of 5 for readability
    for i in range(0, len(checkpoints), 5):
        row_checkpoints = checkpoints[i:i+5]
        row = " | ".join(cp['compressed'] for cp in row_checkpoints)
        md_lines.append(row)
    
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("**Usage in Stage 1 Prompt:**")
    md_lines.append("```")
    md_lines.append("CV CHECKPOINTS (verify at these timestamps):")
    for i in range(0, min(10, len(checkpoints))):
        md_lines.append(checkpoints[i]['compressed'])
    if len(checkpoints) > 10:
        md_lines.append("...")
    md_lines.append("```")
    
    md_content = "\n".join(md_lines)
    md_path = output_path / "stage0_cv_checkpoints_5s.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    print(f"  Saved markdown: {md_path}")
    
    # Generate JSON output
    json_data = {
        'source': cv_cache_path,
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'video_duration': duration,
        'checkpoint_interval': checkpoint_interval,
        'num_checkpoints': len(checkpoints),
        'estimated_tokens': estimated_tokens,
        'legend': {
            'standing': {'S': 'Standing', 'G': 'Ground', 'M': 'Mixed', '?': 'Not detected'},
            'contact': {'L': 'Low', 'M': 'Medium', 'H': 'High'},
            'dominant': {'A1': 'Athlete 1', 'A2': 'Athlete 2', 'N': 'Neutral'}
        },
        'checkpoints': checkpoints
    }
    
    json_path = output_path / "stage0_cv_checkpoints_5s.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"  Saved JSON: {json_path}")
    
    return json_data


if __name__ == '__main__':
    # Test with sample video
    video_path = "data/videos/youtube_SMRbZEbxepA.mp4"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    result = run_cv_preprocessing(video_path, cache_dir="outputs/experiment3")
    
    print(f"\n✓ CV preprocessing complete")
    print(f"  Metrics per second: {len(result['per_second_metrics'])}")
    print(f"  Scenes detected: {len(result['scenes'])}")
