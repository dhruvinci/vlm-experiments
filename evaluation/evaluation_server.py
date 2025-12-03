#!/usr/bin/env python3
"""
HITL Evaluation Server

FastAPI backend for Human-in-the-Loop evaluation of BJJ analysis experiments.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from functools import lru_cache
import gc

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from accuracy_calculator import AccuracyCalculator, compare_experiments
from bjj_ontology import BJJ_ONTOLOGY, RATING_SCALE, get_all_labels, get_position_hierarchy


app = FastAPI(title="BJJ HITL Evaluation Server")

# Cache for experiments list (5 minute TTL)
_experiments_cache = None
_cache_timestamp = None
CACHE_TTL = 300  # 5 minutes

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class SubSegment(BaseModel):
    start_time: str
    end_time: str
    note: str
    labels: List[str] = []

class AlternativePrediction(BaseModel):
    position: str
    sub_position: Optional[str] = None
    confidence: float
    rank: int  # 2 for secondary, 3 for tertiary

class PositionLabel(BaseModel):
    start_time: str
    end_time: str
    position: str
    sub_position: Optional[str] = None
    top_athlete: Optional[str] = None
    bottom_athlete: Optional[str] = None
    confidence: float = 1.0
    notes: Optional[str] = None
    is_correct: Optional[bool] = None  # For validation
    rating: Optional[int] = None  # 0-5 rating scale
    completed: Optional[bool] = False  # Track completion status
    labels: List[str] = []  # Multi-select labels from ontology
    sub_segments: List[SubSegment] = []  # Split segments with detailed labels
    alternative_predictions: List[AlternativePrediction] = []  # AI's other predictions

class TransitionLabel(BaseModel):
    timestamp: str
    from_position: str
    to_position: str
    technique: Optional[str] = None
    notes: Optional[str] = None
    is_correct: Optional[bool] = None

class ScoringLabel(BaseModel):
    athlete_name: str
    takedowns: int = 0
    guard_passes: int = 0
    knee_on_belly: int = 0
    mounts: int = 0
    back_controls: int = 0
    total_points: int = 0

class GroundTruth(BaseModel):
    video_path: str
    video_duration: str
    labeled_by: str
    labeled_date: str
    ground_truth_positions: List[PositionLabel]
    ground_truth_transitions: List[TransitionLabel]
    ground_truth_scoring_adcc: Dict[str, Any]
    notes: Optional[str] = None


# API Endpoints

@app.get("/")
def root():
    """API health check."""
    return {
        "status": "ok",
        "service": "BJJ HITL Evaluation Server",
        "version": "2.0"
    }


@app.get("/ontology")
def get_ontology() -> Dict[str, Any]:
    """
    Get the complete BJJ ontology.

    Returns:
        Complete ontology with positions, techniques, transitions, etc.
    """
    return BJJ_ONTOLOGY


@app.get("/ontology/labels")
def get_ontology_labels() -> Dict[str, Any]:
    """
    Get flat list of all labels for autocomplete.

    Returns:
        List of all possible labels
    """
    return {
        "labels": get_all_labels(),
        "total": len(get_all_labels())
    }


@app.get("/ontology/positions")
def get_position_hierarchy_endpoint() -> Dict[str, Any]:
    """
    Get position hierarchy for structured dropdowns.

    Returns:
        Positions with their sub-positions
    """
    return {
        "positions": get_position_hierarchy()
    }


@app.get("/rating-scale")
def get_rating_scale_endpoint() -> Dict[str, Any]:
    """
    Get rating scale definitions.

    Returns:
        Rating scale with 0-5 definitions
    """
    return RATING_SCALE


@app.get("/experiments/list")
def list_experiments(results_dir: str = "results") -> Dict[str, Any]:
    """
    List all available experiment results (with caching).

    Args:
        results_dir: Base directory containing results

    Returns:
        List of experiments with metadata
    """
    global _experiments_cache, _cache_timestamp
    
    # Check cache validity
    now = datetime.now().timestamp()
    if _experiments_cache is not None and _cache_timestamp is not None:
        if (now - _cache_timestamp) < CACHE_TTL:
            return _experiments_cache
    
    # Handle relative paths from project root
    results_path = Path(results_dir)
    if not results_path.is_absolute():
        results_path = Path(__file__).parent.parent / results_dir
    
    if not results_path.exists():
        raise HTTPException(status_code=404, detail=f"Results directory not found: {results_dir}")

    experiments = []

    # Scan for experiment directories (limit depth to avoid deep recursion)
    for exp_dir in results_path.glob("*"):
        if not exp_dir.is_dir():
            continue

        analysis_file = exp_dir / "analysis.json"
        metadata_file = exp_dir / "metadata.json"

        if analysis_file.exists():
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass

            experiments.append({
                "name": exp_dir.name,
                "path": str(exp_dir),
                "analysis_path": str(analysis_file),
                "metadata": metadata,
                "timestamp": metadata.get("timestamp", "unknown")
            })

    # Scan outputs/experiment3 for Exp3 results
    outputs_path = Path(__file__).parent.parent / "outputs/experiment3"
    if outputs_path.exists():
        for exp_file in outputs_path.glob("stage1_timeline*.json"):
            experiments.append({
                "name": f"exp3_{exp_file.stem}",
                "path": str(outputs_path),
                "analysis_path": str(exp_file),
                "metadata": {
                    "experiment_name": f"Experiment 3 - {exp_file.stem}",
                    "type": "exp3"
                },
                "timestamp": datetime.fromtimestamp(exp_file.stat().st_mtime).isoformat()
            })
    
    # Scan outputs/experiment4.0 for Exp4 results
    exp4_path = Path(__file__).parent.parent / "outputs/experiment4.0"
    if exp4_path.exists():
        for exp_file in exp4_path.glob("combined_*.json"):
            experiments.append({
                "name": f"exp4_{exp_file.stem}",
                "path": str(exp4_path),
                "analysis_path": str(exp_file),
                "metadata": {
                    "experiment_name": f"Experiment 4 - {exp_file.stem.replace('combined_', '')}",
                    "type": "exp4"
                },
                "timestamp": datetime.fromtimestamp(exp_file.stat().st_mtime).isoformat()
            })
        
        for exp_file in exp4_path.glob("stage1v*.json"):
            experiments.append({
                "name": f"exp4_{exp_file.stem}",
                "path": str(exp4_path),
                "analysis_path": str(exp_file),
                "metadata": {
                    "experiment_name": f"Experiment 4 - {exp_file.stem.replace('stage1v', '')}",
                    "type": "exp4"
                },
                "timestamp": datetime.fromtimestamp(exp_file.stat().st_mtime).isoformat()
            })

    # Sort by timestamp (newest first)
    experiments.sort(key=lambda x: x["timestamp"], reverse=True)

    result = {
        "total": len(experiments),
        "experiments": experiments
    }
    
    # Cache the result
    _experiments_cache = result
    _cache_timestamp = now
    
    return result


def _detect_experiment_format(analysis: Dict[str, Any]) -> str:
    """
    Detect experiment format based on JSON structure.
    
    Args:
        analysis: Analysis JSON data
        
    Returns:
        Format type: "exp4", "exp3", "exp1_2", or "unknown"
    """
    # Exp4 combined format: has "meta" with "stage1_meta" and "segments" with "has_detail"
    if "meta" in analysis and "segments" in analysis:
        segments = analysis.get("segments", [])
        if segments and any("has_detail" in seg for seg in segments):
            return "exp4"
        # Exp4 v3 format: has "meta" with "video_path" and "segments" with "strategy"
        if segments and any("strategy" in seg for seg in segments):
            return "exp4"
    # Exp3 has "segments" and "athlete_profiles"
    if "segments" in analysis and "athlete_profiles" in analysis:
        return "exp3"
    # Exp1/2 has "position_timeline"
    elif "position_timeline" in analysis:
        return "exp1_2"
    else:
        return "unknown"


def _normalize_exp4_to_standard(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize Experiment 4 format to standard format for compatibility.
    Handles both combined_*.json and stage1v*.json formats.
    Optimized to avoid loading unnecessary data.
    
    Args:
        analysis: Exp4 analysis data
        
    Returns:
        Normalized analysis with position_timeline
    """
    meta = analysis.get("meta", {})
    stage1_meta = meta.get("stage1_meta", {})
    
    # Extract video metadata (handle both combined and stage1v formats)
    video_metadata = {
        "video_path": meta.get("video_path", "data/videos/youtube_SMRbZEbxepA.mp4"),
        "duration": meta.get("duration", stage1_meta.get("duration", "")),
        "context": meta.get("context", stage1_meta.get("context", "")),
        "athletes": meta.get("athletes", stage1_meta.get("athletes", ""))
    }
    
    normalized = {
        "video_metadata": video_metadata,
        "position_timeline": [],
        "experiment_format": "exp4",
        "meta": meta
    }
    
    # Convert segments to position_timeline format (only include necessary fields)
    for segment in analysis.get("segments", []):
        time_str = segment.get("time", "")
        start_time = time_str.split("-")[0] if "-" in time_str else ""
        end_time = time_str.split("-")[1] if "-" in time_str else ""
        
        position = {
            "start_time": start_time,
            "end_time": end_time,
            "position": segment.get("position", ""),
            "sub_position": "N/A",
            "top_athlete": segment.get("top", "-"),
            "bottom_athlete": "-",
            "confidence": float(segment.get("control", 0.0)),
            "action_score": float(segment.get("action", 0.0)),
            "reasons": segment.get("reasons", ""),
            "focus": segment.get("focus", ""),
            "notes": segment.get("notes", ""),
            "strategy": segment.get("strategy", ""),
            "setup": segment.get("setup", ""),
            "execution": segment.get("execution", ""),
            "outcome": segment.get("outcome", ""),
            "coaching": segment.get("coaching", ""),
            "has_detail": segment.get("has_detail", False),
            "detail": segment.get("detail", {})
        }
        normalized["position_timeline"].append(position)
    
    return normalized


def _normalize_exp3_to_standard(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize Experiment 3 format to standard format for compatibility.
    
    Args:
        analysis: Exp3 analysis data
        
    Returns:
        Normalized analysis with position_timeline
    """
    # Add video metadata if missing (Exp3 doesn't include it in JSON)
    video_metadata = analysis.get("video_metadata", {})
    if not video_metadata.get("video_path"):
        # Default to the standard video path
        video_metadata["video_path"] = "data/videos/youtube_SMRbZEbxepA.mp4"
    
    normalized = {
        "video_metadata": video_metadata,
        "position_timeline": [],
        "athlete_profiles": analysis.get("athlete_profiles", {}),
        "experiment_format": "exp3",
        "original_data": analysis  # Keep original for reference
    }
    
    # Convert segments to position_timeline format
    for segment in analysis.get("segments", []):
        position = {
            "start_time": segment.get("start", ""),
            "end_time": segment.get("end", ""),
            "position": segment.get("position", ""),
            "sub_position": segment.get("sub_position", "N/A"),
            "top_athlete": segment.get("top_athlete"),
            "bottom_athlete": segment.get("bottom_athlete"),
            "confidence": segment.get("confidence", 0.0),
            # Exp3-specific fields (don't duplicate narrative in notes)
            "avg_action": segment.get("avg_action", 0.0),
            "transition": segment.get("transition", ""),
            "key_moments": segment.get("key_moments", []),
            "key_actions": segment.get("key_actions", []),
            "narrative": segment.get("narrative", "")
        }
        normalized["position_timeline"].append(position)
    
    return normalized


@app.get("/experiments/{experiment_name}")
def get_experiment(experiment_name: str, results_dir: str = "results") -> Dict[str, Any]:
    """
    Load a specific experiment's results.

    Args:
        experiment_name: Name of the experiment directory
        results_dir: Base results directory

    Returns:
        Experiment analysis data (normalized for compatibility)
    """
    # Check if this is an Exp4 file reference
    if experiment_name.startswith("exp4_"):
        # Load from outputs/experiment4.0 directory
        outputs_path = Path("outputs/experiment4.0")
        if not outputs_path.is_absolute():
            outputs_path = Path(__file__).parent.parent / "outputs/experiment4.0"
        
        file_stem = experiment_name.replace("exp4_", "")
        analysis_file = outputs_path / f"{file_stem}.json"
        
        if not analysis_file.exists():
            raise HTTPException(status_code=404, detail=f"Exp4 file not found: {analysis_file}")
        
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        # Normalize Exp4 format
        format_type = _detect_experiment_format(analysis)
        if format_type == "exp4":
            analysis = _normalize_exp4_to_standard(analysis)
        
        # Load ground truth if exists
        ground_truth = None
        gt_file = outputs_path / f"{file_stem}_ground_truth.json"
        if gt_file.exists():
            with open(gt_file, 'r') as f:
                ground_truth = json.load(f)
        
        return {
            "experiment_name": experiment_name,
            "experiment_path": str(outputs_path),
            "analysis": analysis,
            "metadata": {"experiment_name": f"Experiment 4 - {file_stem.replace('combined_', '')}", "type": "exp4"},
            "ground_truth": ground_truth,
            "ground_truth_path": str(gt_file) if gt_file.exists() else None,
            "format": format_type
        }
    
    # Check if this is an Exp3 file reference
    if experiment_name.startswith("exp3_"):
        # Load from outputs/experiment3 directory
        outputs_path = Path("outputs/experiment3")
        if not outputs_path.is_absolute():
            outputs_path = Path(__file__).parent.parent / "outputs/experiment3"
        
        file_stem = experiment_name.replace("exp3_", "")
        analysis_file = outputs_path / f"{file_stem}.json"
        
        if not analysis_file.exists():
            raise HTTPException(status_code=404, detail=f"Exp3 file not found: {analysis_file}")
        
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        # Normalize Exp3 format
        format_type = _detect_experiment_format(analysis)
        if format_type == "exp3":
            analysis = _normalize_exp3_to_standard(analysis)
        
        # Load ground truth if exists
        ground_truth = None
        gt_file = outputs_path / f"{file_stem}_ground_truth.json"
        if gt_file.exists():
            with open(gt_file, 'r') as f:
                ground_truth = json.load(f)
        
        return {
            "experiment_name": experiment_name,
            "experiment_path": str(outputs_path),
            "analysis": analysis,
            "metadata": {"experiment_name": f"Experiment 3 - {file_stem}", "type": "exp3"},
            "ground_truth": ground_truth,
            "ground_truth_path": str(gt_file) if gt_file.exists() else None,
            "format": format_type
        }
    
    # Find experiment directory (Exp1/2)
    results_path = Path(results_dir)
    if not results_path.is_absolute():
        results_path = Path(__file__).parent.parent / results_dir
    
    exp_dir = None

    for candidate in results_path.glob(f"**/{experiment_name}"):
        if candidate.is_dir():
            exp_dir = candidate
            break

    if not exp_dir:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_name}")

    analysis_file = exp_dir / "analysis.json"
    if not analysis_file.exists():
        raise HTTPException(status_code=404, detail=f"Analysis file not found in {experiment_name}")

    with open(analysis_file, 'r') as f:
        analysis = json.load(f)

    # Detect format and normalize if needed
    format_type = _detect_experiment_format(analysis)
    if format_type == "exp3":
        analysis = _normalize_exp3_to_standard(analysis)
    else:
        # Add format marker for exp1/2
        analysis["experiment_format"] = "exp1_2"

    # Load metadata if available
    metadata = {}
    metadata_file = exp_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    # Load experiment-specific ground truth if available
    ground_truth = None
    gt_file = exp_dir / "ground_truth.json"
    if gt_file.exists():
        with open(gt_file, 'r') as f:
            ground_truth = json.load(f)

    return {
        "experiment_name": experiment_name,
        "experiment_path": str(exp_dir),
        "analysis": analysis,
        "metadata": metadata,
        "ground_truth": ground_truth,
        "ground_truth_path": str(gt_file) if gt_file.exists() else None,
        "format": format_type
    }


@app.post("/ground_truth/save")
def save_ground_truth(ground_truth: GroundTruth, output_path: str = "evaluation/ground_truth.json"):
    """
    Save ground truth labels.

    Args:
        ground_truth: Ground truth data
        output_path: Where to save the file

    Returns:
        Success confirmation
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict and save
    gt_dict = ground_truth.dict()

    with open(output_file, 'w') as f:
        json.dump(gt_dict, f, indent=2)

    return {
        "status": "success",
        "message": f"Ground truth saved to {output_path}",
        "path": str(output_file.absolute()),
        "positions_labeled": len(ground_truth.ground_truth_positions),
        "transitions_labeled": len(ground_truth.ground_truth_transitions)
    }


@app.get("/ground_truth/load")
def load_ground_truth(path: str = "evaluation/ground_truth.json") -> Dict[str, Any]:
    """
    Load existing ground truth labels.

    Args:
        path: Path to ground truth file

    Returns:
        Ground truth data
    """
    gt_file = Path(path)

    if not gt_file.exists():
        raise HTTPException(status_code=404, detail=f"Ground truth file not found: {path}")

    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)

    return ground_truth


@app.post("/evaluate")
def evaluate_experiment(
    experiment_path: str,
    ground_truth_path: str = "evaluation/ground_truth.json"
) -> Dict[str, Any]:
    """
    Evaluate an experiment against ground truth.

    Args:
        experiment_path: Path to experiment's analysis.json
        ground_truth_path: Path to ground truth file

    Returns:
        Evaluation metrics
    """
    gt_file = Path(ground_truth_path)
    exp_file = Path(experiment_path)

    if not gt_file.exists():
        raise HTTPException(status_code=404, detail=f"Ground truth not found: {ground_truth_path}")

    if not exp_file.exists():
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_path}")

    calculator = AccuracyCalculator(str(gt_file))
    metrics = calculator.calculate_all_metrics(str(exp_file))

    return metrics


@app.post("/evaluate/compare")
def compare_multiple_experiments(
    experiment_paths: List[str],
    ground_truth_path: str = "evaluation/ground_truth.json"
) -> Dict[str, Any]:
    """
    Compare multiple experiments against ground truth.

    Args:
        experiment_paths: List of paths to experiment analysis.json files
        ground_truth_path: Path to ground truth file

    Returns:
        Comparison results with rankings
    """
    gt_file = Path(ground_truth_path)

    if not gt_file.exists():
        raise HTTPException(status_code=404, detail=f"Ground truth not found: {ground_truth_path}")

    comparison = compare_experiments(str(gt_file), experiment_paths)

    return comparison


@app.get("/videos/info")
def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        Video information
    """
    video_file = Path(video_path)

    # Try adding .mp4 extension if file doesn't exist
    if not video_file.exists():
        video_file_with_ext = Path(str(video_path) + ".mp4")
        if video_file_with_ext.exists():
            video_file = video_file_with_ext

    if not video_file.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    # Get basic file info
    stat = video_file.stat()

    # Try to get video duration using cv2
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
    except:
        duration_str = "unknown"
        fps = 0
        frame_count = 0

    return {
        "path": str(video_file.absolute()),
        "name": video_file.name,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "duration": duration_str,
        "fps": fps,
        "frame_count": frame_count
    }


@app.get("/videos/stream")
def stream_video(video_path: str):
    """
    Stream video file to frontend.

    Args:
        video_path: Path to video file (relative or absolute)

    Returns:
        Video file response
    """
    video_file = Path(video_path)
    
    # Handle relative paths from project root
    if not video_file.is_absolute():
        video_file = Path(__file__).parent.parent / video_path

    # Try adding .mp4 extension if file doesn't exist
    if not video_file.exists():
        video_file_with_ext = Path(str(video_file) + ".mp4")
        if video_file_with_ext.exists():
            video_file = video_file_with_ext

    if not video_file.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path} (resolved to {video_file})")

    return FileResponse(
        path=str(video_file.absolute()),
        media_type="video/mp4",
        filename=video_file.name
    )


@app.on_event("startup")
async def startup_event():
    """Optimize server on startup."""
    gc.collect()
    print("Server started - memory optimized")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    global _experiments_cache, _cache_timestamp
    _experiments_cache = None
    _cache_timestamp = None
    gc.collect()


if __name__ == "__main__":
    import uvicorn

    print("Starting HITL Evaluation Server...")
    print("API docs available at: http://localhost:5002/docs")
    print("Memory optimization enabled")

    uvicorn.run(app, host="0.0.0.0", port=5002, workers=1)
