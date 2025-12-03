# HITL Annotation Tool - VideoEvaluatorV3

Human-in-the-loop tool for creating ground truth annotations of BJJ matches.

## Features

- **Segment-by-segment review** - Navigate through 32 AI-detected positions
- **Detailed feedback** - Add sub-segments with technique labels and notes
- **Rating system** - Rate AI prediction quality 1-5 stars
- **Progress tracking** - Track completion status across all segments
- **Video playback control** - Precise seeking, play/pause, skip controls
- **Data persistence** - Auto-save to backend, track completion

## Architecture

```
Frontend (React)              Backend (FastAPI)
├── VideoEvaluatorV3.js  →   evaluation_server.py
├── VideoPlayer.js            ├── Load experiment results
└── [UI components]           ├── Save ground truth
                              └── Validate annotations
```

## Setup

### Prerequisites
- Node.js 14+
- Python 3.8+
- FFmpeg (for video processing)

### Backend Setup
```bash
cd ../evaluation
python evaluation_server.py
# Server runs on http://localhost:5002
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
# App runs on http://localhost:3000
```

## Usage

1. **Start servers** - Run both backend and frontend
2. **Load experiment** - Tool auto-loads from `results/gemini_exp5/`
3. **Review segments** - Navigate through each AI-detected position
4. **Add feedback** - For each segment:
   - Rate AI prediction (1-5 stars)
   - Add sub-segments with:
     - Start/end timestamps (5-15 second windows)
     - Position label
     - Technique labels (multiple allowed)
     - Notes
5. **Validate & Save** - Click "Save & Next" (validates before saving)
6. **Track progress** - Progress bar shows X/32 completed

## Validation Rules

Each sub-segment must have:
- Valid start/end times (within segment boundaries)
- At least one label OR note
- Start time < end time

## Keyboard Shortcuts

- **Arrow Keys** - Navigate between segments
- **Space** - Play/pause video
- **Shift + Left/Right** - Skip 5 seconds

## Data Format

### Input (Experiment Results)
```json
{
  "position_timeline": [
    {
      "start_time": "00:00",
      "end_time": "00:08",
      "position": "standing",
      "sub_position": "grip_fighting",
      "notes": "AI prediction notes"
    }
  ]
}
```

### Output (Ground Truth)
```json
{
  "ground_truth_positions": [
    {
      "start_time": "00:00",
      "end_time": "00:08",
      "position": "standing",
      "rating": 3,
      "completed": true,
      "sub_segments": [
        {
          "start_time": "00:00",
          "end_time": "00:05",
          "position": "Standing",
          "labels": ["grip_fighting", "underhooks"],
          "notes": "Both athletes establishing grips"
        }
      ]
    }
  ]
}
```

## Troubleshooting

**Issue: Data not saving**
- Check backend server is running (http://localhost:5002/health)
- Check browser console for errors
- Verify ground_truth.json has write permissions

**Issue: Video not loading**
- Ensure video path in experiment results is correct
- Check video format is supported (mp4, webm)
- Verify browser can play the video codec

**Issue: Validation errors persist**
- Validation errors are cleared when loading new segments
- If stuck, check console for React state errors

## Known Limitations

- Video must be locally accessible (no remote URLs yet)
- Maximum 200 sub-segments total (React performance)
- Auto-save triggers on navigation (not auto-timed)

## Future Improvements

- Batch export to multiple formats (CSV, JSON, annotations)
- Video frame extraction for ML training
- Multi-user annotation comparison
- Inter-annotator agreement metrics
