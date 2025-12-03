import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  ButtonGroup,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  TextField,
  Grid,
  Paper,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import EditIcon from '@mui/icons-material/Edit';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore';
import SaveIcon from '@mui/icons-material/Save';
import AssessmentIcon from '@mui/icons-material/Assessment';

import VideoPlayer from './VideoPlayer';
import PositionMarker from './PositionMarker';

const API_BASE_URL = 'http://localhost:8001';

const VideoEvaluator = () => {
  // State
  const [experiments, setExperiments] = useState([]);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [experimentData, setExperimentData] = useState(null);
  const [videoPath, setVideoPath] = useState('');
  const [currentTime, setCurrentTime] = useState(0);

  // Ground truth state
  const [groundTruth, setGroundTruth] = useState({
    positions: [],
    transitions: [],
    scoring: {}
  });

  // Evaluation state
  const [currentPositionIndex, setCurrentPositionIndex] = useState(0);
  const [labeledCount, setLabeledCount] = useState(0);
  const [editMode, setEditMode] = useState(false);
  const [editedPosition, setEditedPosition] = useState(null);
  const [feedbackNotes, setFeedbackNotes] = useState('');

  // Metrics state
  const [showMetrics, setShowMetrics] = useState(false);
  const [metrics, setMetrics] = useState(null);

  // Load experiments on mount
  useEffect(() => {
    loadExperiments();
  }, []);

  // Load ground truth if it exists
  useEffect(() => {
    loadGroundTruth();
  }, []);

  // Auto-select position based on current video time
  useEffect(() => {
    if (!experimentData || !experimentData.analysis || !experimentData.analysis.position_timeline) {
      return;
    }

    const positions = experimentData.analysis.position_timeline;

    // Find the position that contains the current time
    const positionIndex = positions.findIndex(pos => {
      const startSeconds = timeToSeconds(pos.start_time);
      const endSeconds = timeToSeconds(pos.end_time);
      return currentTime >= startSeconds && currentTime <= endSeconds;
    });

    if (positionIndex !== -1 && positionIndex !== currentPositionIndex) {
      setCurrentPositionIndex(positionIndex);
    }
  }, [currentTime, experimentData, currentPositionIndex]);

  const loadExperiments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/experiments/list?results_dir=results`);
      const data = await response.json();
      setExperiments(data.experiments || []);
    } catch (error) {
      console.error('Error loading experiments:', error);
    }
  };

  const loadExperimentData = async (experimentName) => {
    try {
      const response = await fetch(`${API_BASE_URL}/experiments/${experimentName}?results_dir=results`);
      const data = await response.json();

      setExperimentData(data);
      setSelectedExperiment(experimentName);

      // Extract video path from metadata and convert to streaming URL
      if (data.metadata && data.metadata.video_path) {
        const videoPath = data.metadata.video_path;
        // Create streaming URL
        const streamingUrl = `${API_BASE_URL}/videos/stream?video_path=${encodeURIComponent(videoPath)}`;
        setVideoPath(streamingUrl);
      }

      setCurrentPositionIndex(0);
    } catch (error) {
      console.error('Error loading experiment:', error);
    }
  };

  const loadGroundTruth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/ground_truth/load`);
      if (response.ok) {
        const data = await response.json();
        setGroundTruth({
          positions: data.ground_truth_positions || [],
          transitions: data.ground_truth_transitions || [],
          scoring: data.ground_truth_scoring_adcc || {}
        });

        // Calculate labeled count
        const labeled = data.ground_truth_positions?.filter(p => p.is_correct !== undefined).length || 0;
        setLabeledCount(labeled);
      }
    } catch (error) {
      console.log('No existing ground truth found');
    }
  };

  const saveGroundTruth = async () => {
    try {
      const gtData = {
        video_path: videoPath,
        video_duration: experimentData?.analysis?.video_metadata?.duration || '00:00:00',
        labeled_by: 'dhruva',
        labeled_date: new Date().toISOString(),
        ground_truth_positions: groundTruth.positions,
        ground_truth_transitions: groundTruth.transitions,
        ground_truth_scoring_adcc: groundTruth.scoring
      };

      const response = await fetch(`${API_BASE_URL}/ground_truth/save`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(gtData),
      });

      if (response.ok) {
        alert('Ground truth saved successfully!');
      }
    } catch (error) {
      console.error('Error saving ground truth:', error);
      alert('Error saving ground truth');
    }
  };

  const calculateMetrics = async () => {
    if (!experimentData) return;

    try {
      const analysisPath = experimentData.analysis_path;
      const response = await fetch(`${API_BASE_URL}/evaluate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          experiment_path: analysisPath,
          ground_truth_path: 'evaluation/ground_truth.json'
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
        setShowMetrics(true);
      }
    } catch (error) {
      console.error('Error calculating metrics:', error);
    }
  };

  const markAsCorrect = () => {
    if (!experimentData || currentPositionIndex < 0) return;

    const positions = experimentData.analysis.position_timeline || [];
    const currentPos = positions[currentPositionIndex];

    // Add to ground truth with feedback notes
    const gtPosition = {
      ...currentPos,
      is_correct: true,
      feedback_notes: feedbackNotes || currentPos.notes
    };

    const newGroundTruth = { ...groundTruth };
    newGroundTruth.positions.push(gtPosition);
    setGroundTruth(newGroundTruth);

    setLabeledCount(labeledCount + 1);
    setFeedbackNotes(''); // Clear notes for next position
    goToNext();
  };

  const markAsWrong = () => {
    setEditMode(true);

    const positions = experimentData.analysis.position_timeline || [];
    const currentPos = positions[currentPositionIndex];

    setEditedPosition({
      ...currentPos,
      is_correct: false
    });
  };

  const saveEditedPosition = () => {
    const newGroundTruth = { ...groundTruth };
    // Include feedback notes in the edited position
    const positionWithNotes = {
      ...editedPosition,
      feedback_notes: feedbackNotes || editedPosition.notes
    };
    newGroundTruth.positions.push(positionWithNotes);
    setGroundTruth(newGroundTruth);

    setLabeledCount(labeledCount + 1);
    setEditMode(false);
    setEditedPosition(null);
    setFeedbackNotes(''); // Clear notes for next position
    goToNext();
  };

  const goToNext = () => {
    const positions = experimentData?.analysis?.position_timeline || [];
    if (currentPositionIndex < positions.length - 1) {
      const nextIndex = currentPositionIndex + 1;
      setCurrentPositionIndex(nextIndex);

      // Update video time
      const nextPos = positions[nextIndex];
      if (nextPos.start_time) {
        const seconds = timeToSeconds(nextPos.start_time);
        setCurrentTime(seconds);
      }
    }
  };

  const goToPrevious = () => {
    if (currentPositionIndex > 0) {
      const prevIndex = currentPositionIndex - 1;
      setCurrentPositionIndex(prevIndex);

      // Update video time
      const positions = experimentData?.analysis?.position_timeline || [];
      const prevPos = positions[prevIndex];
      if (prevPos.start_time) {
        const seconds = timeToSeconds(prevPos.start_time);
        setCurrentTime(seconds);
      }
    }
  };

  const jumpToPosition = (index) => {
    setCurrentPositionIndex(index);

    const positions = experimentData?.analysis?.position_timeline || [];
    const pos = positions[index];
    if (pos.start_time) {
      const seconds = timeToSeconds(pos.start_time);
      setCurrentTime(seconds);
    }
  };

  const timeToSeconds = (timeStr) => {
    const parts = timeStr.split(':');
    if (parts.length === 2) {
      return parseInt(parts[0]) * 60 + parseInt(parts[1]);
    } else if (parts.length === 3) {
      return parseInt(parts[0]) * 3600 + parseInt(parts[1]) * 60 + parseInt(parts[2]);
    }
    return 0;
  };

  // Render
  const positions = experimentData?.analysis?.position_timeline || [];
  const currentPosition = positions[currentPositionIndex];
  const progress = positions.length > 0 ? (labeledCount / positions.length) * 100 : 0;

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Typography variant="h4" gutterBottom>
        BJJ Analysis Evaluation (HITL)
      </Typography>

      {/* Experiment Selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Select Experiment</InputLabel>
                <Select
                  value={selectedExperiment || ''}
                  onChange={(e) => loadExperimentData(e.target.value)}
                >
                  {experiments.map((exp) => (
                    <MenuItem key={exp.name} value={exp.name}>
                      {exp.name} ({exp.timestamp})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={saveGroundTruth}
                  disabled={labeledCount === 0}
                >
                  Save Ground Truth ({labeledCount})
                </Button>

                <Button
                  variant="outlined"
                  startIcon={<AssessmentIcon />}
                  onClick={calculateMetrics}
                  disabled={labeledCount === 0}
                >
                  Calculate Metrics
                </Button>
              </Box>
            </Grid>
          </Grid>

          {/* Progress Bar */}
          {experimentData && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" gutterBottom>
                Progress: {labeledCount} / {positions.length} positions labeled ({Math.round(progress)}%)
              </Typography>
              <LinearProgress variant="determinate" value={progress} />
            </Box>
          )}
        </CardContent>
      </Card>

      {experimentData && (
        <Grid container spacing={3}>
          {/* Video Player */}
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <VideoPlayer
                  videoUrl={videoPath}
                  currentTime={currentTime}
                  onTimeUpdate={setCurrentTime}
                />

                {/* Position Markers Timeline */}
                <Box sx={{ mt: 2 }}>
                  {positions.map((pos, index) => (
                    <PositionMarker
                      key={index}
                      position={pos}
                      isActive={index === currentPositionIndex}
                      onClick={() => jumpToPosition(index)}
                      isLabeled={groundTruth.positions.some(
                        gt => gt.start_time === pos.start_time && gt.position === pos.position
                      )}
                    />
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Annotation Panel */}
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Current Position ({currentPositionIndex + 1}/{positions.length})
                </Typography>

                {currentPosition && (
                  <Box>
                    <Paper sx={{ p: 2, mb: 2, bgcolor: 'primary.main', color: 'white' }}>
                      <Typography variant="subtitle2" sx={{ color: 'rgba(255,255,255,0.9)', fontWeight: 600 }}>
                        AI Prediction
                      </Typography>

                      <Box sx={{ mt: 1 }}>
                        <Chip
                          label={`${currentPosition.start_time} - ${currentPosition.end_time}`}
                          size="small"
                          sx={{ mr: 1, bgcolor: 'rgba(255,255,255,0.2)', color: 'white', fontWeight: 500 }}
                        />
                        <Chip
                          label={currentPosition.position}
                          sx={{ mr: 1, bgcolor: 'rgba(255,255,255,0.3)', color: 'white', fontWeight: 600 }}
                        />
                        {currentPosition.sub_position && (
                          <Chip
                            label={currentPosition.sub_position}
                            size="small"
                            sx={{ bgcolor: 'rgba(255,255,255,0.15)', color: 'white' }}
                          />
                        )}
                      </Box>

                      <Typography variant="body2" sx={{ mt: 1.5, color: 'white' }}>
                        <strong>Top:</strong> {currentPosition.top_athlete || 'N/A'}
                      </Typography>

                      <Typography variant="body2" sx={{ color: 'white' }}>
                        <strong>Bottom:</strong> {currentPosition.bottom_athlete || 'N/A'}
                      </Typography>

                      <Typography variant="body2" sx={{ color: 'white' }}>
                        <strong>Confidence:</strong> {(currentPosition.confidence * 100).toFixed(0)}%
                      </Typography>

                      {currentPosition.notes && (
                        <Typography variant="body2" sx={{ mt: 1.5, fontStyle: 'italic', color: 'rgba(255,255,255,0.95)' }}>
                          {currentPosition.notes}
                        </Typography>
                      )}
                    </Paper>

                    {/* Annotation Controls */}
                    {!editMode ? (
                      <Box>
                        <Typography variant="subtitle2" gutterBottom>
                          Is this correct?
                        </Typography>

                        <TextField
                          fullWidth
                          label="Feedback / Notes (optional)"
                          multiline
                          rows={2}
                          value={feedbackNotes}
                          onChange={(e) => setFeedbackNotes(e.target.value)}
                          placeholder="e.g., 'Actually showing handshake, not grip fighting yet'"
                          sx={{ mb: 2 }}
                          size="small"
                        />

                        <ButtonGroup fullWidth variant="contained" sx={{ mb: 2 }}>
                          <Button
                            color="success"
                            startIcon={<CheckIcon />}
                            onClick={markAsCorrect}
                          >
                            Correct
                          </Button>

                          <Button
                            color="error"
                            startIcon={<CloseIcon />}
                            onClick={markAsWrong}
                          >
                            Wrong
                          </Button>
                        </ButtonGroup>

                        {/* Navigation */}
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Button
                            startIcon={<NavigateBeforeIcon />}
                            onClick={goToPrevious}
                            disabled={currentPositionIndex === 0}
                          >
                            Previous
                          </Button>

                          <Button
                            endIcon={<NavigateNextIcon />}
                            onClick={goToNext}
                            disabled={currentPositionIndex >= positions.length - 1}
                          >
                            Next
                          </Button>
                        </Box>
                      </Box>
                    ) : (
                      // Edit Mode
                      <Box>
                        <Typography variant="subtitle2" gutterBottom color="error">
                          Correct the label:
                        </Typography>

                        <FormControl fullWidth sx={{ mb: 2 }}>
                          <InputLabel>Position</InputLabel>
                          <Select
                            value={editedPosition?.position || ''}
                            onChange={(e) => setEditedPosition({
                              ...editedPosition,
                              position: e.target.value
                            })}
                          >
                            <MenuItem value="standing">Standing</MenuItem>
                            <MenuItem value="guard">Guard</MenuItem>
                            <MenuItem value="mount">Mount</MenuItem>
                            <MenuItem value="side_control">Side Control</MenuItem>
                            <MenuItem value="back_control">Back Control</MenuItem>
                            <MenuItem value="half_guard">Half Guard</MenuItem>
                            <MenuItem value="turtle">Turtle</MenuItem>
                            <MenuItem value="north_south">North-South</MenuItem>
                            <MenuItem value="knee_on_belly">Knee on Belly</MenuItem>
                          </Select>
                        </FormControl>

                        <TextField
                          fullWidth
                          label="Notes (optional)"
                          multiline
                          rows={2}
                          value={editedPosition?.notes || ''}
                          onChange={(e) => setEditedPosition({
                            ...editedPosition,
                            notes: e.target.value
                          })}
                          sx={{ mb: 2 }}
                        />

                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Button
                            variant="contained"
                            color="primary"
                            onClick={saveEditedPosition}
                            fullWidth
                          >
                            Save Correction
                          </Button>

                          <Button
                            variant="outlined"
                            onClick={() => {
                              setEditMode(false);
                              setEditedPosition(null);
                            }}
                          >
                            Cancel
                          </Button>
                        </Box>
                      </Box>
                    )}
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Metrics Dialog */}
      <Dialog
        open={showMetrics}
        onClose={() => setShowMetrics(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Experiment Metrics</DialogTitle>
        <DialogContent>
          {metrics && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Overall Score: {(metrics.overall_score * 100).toFixed(1)}%
              </Typography>

              <Typography variant="subtitle1" sx={{ mt: 2 }}>
                Position Metrics:
              </Typography>
              <pre>{JSON.stringify(metrics.position_metrics, null, 2)}</pre>

              <Typography variant="subtitle1" sx={{ mt: 2 }}>
                Transition Metrics:
              </Typography>
              <pre>{JSON.stringify(metrics.transition_metrics, null, 2)}</pre>

              <Typography variant="subtitle1" sx={{ mt: 2 }}>
                Scoring Metrics:
              </Typography>
              <pre>{JSON.stringify(metrics.scoring_metrics, null, 2)}</pre>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowMetrics(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default VideoEvaluator;
