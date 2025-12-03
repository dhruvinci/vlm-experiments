import React, { useState, useEffect, useRef, useCallback } from 'react';
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
  Autocomplete,
  Rating,
  ToggleButton,
  ToggleButtonGroup,
  Divider,
  Alert,
  Stack,
  Badge,
} from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore';
import SaveIcon from '@mui/icons-material/Save';
import AssessmentIcon from '@mui/icons-material/Assessment';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import LoopIcon from '@mui/icons-material/Loop';
import SkipNextIcon from '@mui/icons-material/SkipNext';
import SkipPreviousIcon from '@mui/icons-material/SkipPrevious';

import VideoPlayer from './VideoPlayer';
import PositionMarker from './PositionMarker';

const API_BASE_URL = 'http://localhost:8001';

const VideoEvaluatorV2 = () => {
  // State
  const [experiments, setExperiments] = useState([]);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [experimentData, setExperimentData] = useState(null);
  const [videoPath, setVideoPath] = useState('');
  const [currentTime, setCurrentTime] = useState(0);

  // Ontology data
  const [ontology, setOntology] = useState(null);
  const [allLabels, setAllLabels] = useState([]);
  const [ratingScale, setRatingScale] = useState({});

  // Ground truth state
  const [groundTruth, setGroundTruth] = useState({
    positions: [],
    transitions: [],
    scoring: {}
  });

  // Evaluation state
  const [currentPositionIndex, setCurrentPositionIndex] = useState(0);
  const [labeledCount, setLabeledCount] = useState(0);

  // New granular feedback state
  const [rating, setRating] = useState(null);
  const [selectedLabels, setSelectedLabels] = useState([]);
  const [subSegments, setSubSegments] = useState([]);
  const [notes, setNotes] = useState('');
  const [saveStatus, setSaveStatus] = useState(''); // 'saving', 'saved', 'error'
  const [validationErrors, setValidationErrors] = useState([]);

  // Video controls
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [isLooping, setIsLooping] = useState(false);
  const [pauseAtBoundaries, setPauseAtBoundaries] = useState(true);

  // Metrics state
  const [showMetrics, setShowMetrics] = useState(false);
  const [metrics, setMetrics] = useState(null);

  // Auto-save debouncing
  const saveTimeoutRef = useRef(null);
  const videoPlayerRef = useRef(null);

  // Load ontology and experiments on mount
  useEffect(() => {
    loadOntology();
    loadExperiments();
    loadGroundTruth();
  }, []);

  // Auto-select position based on current video time
  useEffect(() => {
    if (!experimentData || !experimentData.analysis || !experimentData.analysis.position_timeline) {
      return;
    }

    const positions = experimentData.analysis.position_timeline;
    const positionIndex = positions.findIndex(pos => {
      const startSeconds = timeToSeconds(pos.start_time);
      const endSeconds = timeToSeconds(pos.end_time);
      return currentTime >= startSeconds && currentTime <= endSeconds;
    });

    if (positionIndex !== -1 && positionIndex !== currentPositionIndex) {
      setCurrentPositionIndex(positionIndex);
      // Load saved feedback for this position
      loadFeedbackForPosition(positionIndex);
    }

    // Pause at segment boundaries if enabled
    if (pauseAtBoundaries && positionIndex !== -1) {
      const pos = positions[positionIndex];
      const endSeconds = timeToSeconds(pos.end_time);
      if (Math.abs(currentTime - endSeconds) < 0.5) {
        // Near end of segment, pause
        videoPlayerRef.current?.pause();
      }
    }
  }, [currentTime, experimentData, currentPositionIndex, pauseAtBoundaries]);

  // Auto-save when feedback changes (debounced)
  useEffect(() => {
    if (rating !== null || selectedLabels.length > 0 || notes || subSegments.length > 0) {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }

      setSaveStatus('saving');
      saveTimeoutRef.current = setTimeout(() => {
        autoSaveCurrentFeedback();
      }, 2000); // Save after 2 seconds of no changes
    }

    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [rating, selectedLabels, notes, subSegments]);

  const loadOntology = async () => {
    try {
      const [ontologyRes, labelsRes, ratingScaleRes] = await Promise.all([
        fetch(`${API_BASE_URL}/ontology`),
        fetch(`${API_BASE_URL}/ontology/labels`),
        fetch(`${API_BASE_URL}/rating-scale`)
      ]);

      const ontologyData = await ontologyRes.json();
      const labelsData = await labelsRes.json();
      const ratingScaleData = await ratingScaleRes.json();

      setOntology(ontologyData);
      setAllLabels(labelsData.labels || []);
      setRatingScale(ratingScaleData);
    } catch (error) {
      console.error('Error loading ontology:', error);
    }
  };

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

      if (data.metadata && data.metadata.video_path) {
        const videoPath = data.metadata.video_path;
        const streamingUrl = `${API_BASE_URL}/videos/stream?video_path=${encodeURIComponent(videoPath)}`;
        setVideoPath(streamingUrl);
      }

      setCurrentPositionIndex(0);
      loadFeedbackForPosition(0);
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

        const labeled = data.ground_truth_positions?.filter(p => p.rating !== undefined).length || 0;
        setLabeledCount(labeled);
      }
    } catch (error) {
      console.log('No existing ground truth found');
    }
  };

  const loadFeedbackForPosition = (index) => {
    const positions = experimentData?.analysis?.position_timeline || [];
    const currentPos = positions[index];

    if (!currentPos) return;

    // Check if we have saved feedback for this position
    const savedFeedback = groundTruth.positions.find(
      gt => gt.start_time === currentPos.start_time && gt.end_time === currentPos.end_time
    );

    if (savedFeedback) {
      setRating(savedFeedback.rating || null);
      setSelectedLabels(savedFeedback.labels || []);
      setSubSegments(savedFeedback.sub_segments || []);
      setNotes(savedFeedback.notes || '');
    } else {
      // Reset to defaults
      setRating(null);
      setSelectedLabels([]);
      setSubSegments([]);
      setNotes('');
    }
  };

  const autoSaveCurrentFeedback = async () => {
    if (!experimentData || currentPositionIndex < 0) return;

    const positions = experimentData.analysis.position_timeline || [];
    const currentPos = positions[currentPositionIndex];

    if (!currentPos) return;

    // Validate before saving
    const errors = validateFeedback();
    if (errors.length > 0) {
      setValidationErrors(errors);
      setSaveStatus('error');
      return;
    }

    setValidationErrors([]);

    // Update or add position to ground truth
    const updatedPosition = {
      ...currentPos,
      rating,
      labels: selectedLabels,
      sub_segments: subSegments,
      notes,
      last_updated: new Date().toISOString()
    };

    const newGroundTruth = { ...groundTruth };
    const existingIndex = newGroundTruth.positions.findIndex(
      gt => gt.start_time === currentPos.start_time && gt.end_time === currentPos.end_time
    );

    if (existingIndex >= 0) {
      newGroundTruth.positions[existingIndex] = updatedPosition;
    } else {
      newGroundTruth.positions.push(updatedPosition);
    }

    setGroundTruth(newGroundTruth);

    // Update labeled count
    const labeled = newGroundTruth.positions.filter(p => p.rating !== undefined).length;
    setLabeledCount(labeled);

    setSaveStatus('saved');
    setTimeout(() => setSaveStatus(''), 2000);
  };

  const validateFeedback = () => {
    const errors = [];

    // Validate sub-segments don't overlap
    if (subSegments.length > 1) {
      for (let i = 0; i < subSegments.length - 1; i++) {
        const current = subSegments[i];
        const next = subSegments[i + 1];

        const currentEnd = timeToSeconds(current.end_time);
        const nextStart = timeToSeconds(next.start_time);

        if (currentEnd > nextStart) {
          errors.push(`Sub-segments ${i + 1} and ${i + 2} overlap`);
        }
      }
    }

    // Validate sub-segments are within parent segment
    if (subSegments.length > 0 && experimentData) {
      const positions = experimentData.analysis.position_timeline || [];
      const currentPos = positions[currentPositionIndex];

      const parentStart = timeToSeconds(currentPos.start_time);
      const parentEnd = timeToSeconds(currentPos.end_time);

      subSegments.forEach((sub, index) => {
        const subStart = timeToSeconds(sub.start_time);
        const subEnd = timeToSeconds(sub.end_time);

        if (subStart < parentStart || subEnd > parentEnd) {
          errors.push(`Sub-segment ${index + 1} is outside parent segment bounds`);
        }
      });
    }

    return errors;
  };

  const addSubSegment = () => {
    const positions = experimentData?.analysis?.position_timeline || [];
    const currentPos = positions[currentPositionIndex];

    if (!currentPos) return;

    // Default new sub-segment to last few seconds of current segment
    const parentEnd = currentPos.end_time;
    const parentStart = currentPos.start_time;

    setSubSegments([...subSegments, {
      start_time: parentStart,
      end_time: parentEnd,
      note: '',
      labels: []
    }]);
  };

  const updateSubSegment = (index, field, value) => {
    const updated = [...subSegments];
    updated[index] = { ...updated[index], [field]: value };
    setSubSegments(updated);
  };

  const deleteSubSegment = (index) => {
    setSubSegments(subSegments.filter((_, i) => i !== index));
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

  const goToNext = () => {
    const positions = experimentData?.analysis?.position_timeline || [];
    if (currentPositionIndex < positions.length - 1) {
      const nextIndex = currentPositionIndex + 1;
      setCurrentPositionIndex(nextIndex);

      const nextPos = positions[nextIndex];
      if (nextPos.start_time) {
        const seconds = timeToSeconds(nextPos.start_time);
        setCurrentTime(seconds);
      }

      loadFeedbackForPosition(nextIndex);
    }
  };

  const goToPrevious = () => {
    if (currentPositionIndex > 0) {
      const prevIndex = currentPositionIndex - 1;
      setCurrentPositionIndex(prevIndex);

      const positions = experimentData?.analysis?.position_timeline || [];
      const prevPos = positions[prevIndex];
      if (prevPos.start_time) {
        const seconds = timeToSeconds(prevPos.start_time);
        setCurrentTime(seconds);
      }

      loadFeedbackForPosition(prevIndex);
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

    loadFeedbackForPosition(index);
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

  const secondsToTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Don't trigger if user is typing in a text field
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
      }

      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          // Frame step backward (0.1 seconds)
          setCurrentTime(Math.max(0, currentTime - 0.1));
          break;
        case 'ArrowRight':
          e.preventDefault();
          // Frame step forward (0.1 seconds)
          setCurrentTime(currentTime + 0.1);
          break;
        case 'j':
          // Jump back 5 seconds
          setCurrentTime(Math.max(0, currentTime - 5));
          break;
        case 'l':
          // Jump forward 5 seconds
          setCurrentTime(currentTime + 5);
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentTime]);

  // Render
  const positions = experimentData?.analysis?.position_timeline || [];
  const currentPosition = positions[currentPositionIndex];
  const progress = positions.length > 0 ? (labeledCount / positions.length) * 100 : 0;

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Typography variant="h4" gutterBottom>
        BJJ Analysis Evaluation (HITL) v2.0
      </Typography>

      {/* Save Status Indicator */}
      {saveStatus && (
        <Alert
          severity={saveStatus === 'saved' ? 'success' : saveStatus === 'saving' ? 'info' : 'error'}
          sx={{ mb: 2 }}
        >
          {saveStatus === 'saved' ? 'Feedback auto-saved' : saveStatus === 'saving' ? 'Saving...' : 'Error saving feedback'}
        </Alert>
      )}

      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <Alert severity="error" sx={{ mb: 2 }}>
          <Typography variant="subtitle2">Validation Errors:</Typography>
          <ul>
            {validationErrors.map((error, i) => (
              <li key={i}>{error}</li>
            ))}
          </ul>
        </Alert>
      )}

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
                  ref={videoPlayerRef}
                  videoUrl={videoPath}
                  currentTime={currentTime}
                  onTimeUpdate={setCurrentTime}
                  playbackRate={playbackSpeed}
                  loop={isLooping}
                />

                {/* Video Controls */}
                <Box sx={{ mt: 2, display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
                  <FormControl size="small">
                    <InputLabel>Speed</InputLabel>
                    <Select
                      value={playbackSpeed}
                      onChange={(e) => setPlaybackSpeed(e.target.value)}
                      label="Speed"
                      sx={{ minWidth: 100 }}
                    >
                      <MenuItem value={0.25}>0.25x</MenuItem>
                      <MenuItem value={0.5}>0.5x</MenuItem>
                      <MenuItem value={1.0}>1x</MenuItem>
                      <MenuItem value={1.5}>1.5x</MenuItem>
                      <MenuItem value={2.0}>2x</MenuItem>
                    </Select>
                  </FormControl>

                  <Tooltip title="Loop current segment">
                    <IconButton
                      color={isLooping ? 'primary' : 'default'}
                      onClick={() => setIsLooping(!isLooping)}
                    >
                      <LoopIcon />
                    </IconButton>
                  </Tooltip>

                  <Tooltip title="Pause at segment boundaries">
                    <Button
                      size="small"
                      variant={pauseAtBoundaries ? 'contained' : 'outlined'}
                      onClick={() => setPauseAtBoundaries(!pauseAtBoundaries)}
                    >
                      Pause at Boundaries
                    </Button>
                  </Tooltip>

                  <Typography variant="caption" sx={{ ml: 'auto', color: 'text.secondary' }}>
                    ← → : Frame step | J/L: Jump 5s
                  </Typography>
                </Box>

                {/* Position Markers Timeline */}
                <Box sx={{ mt: 2 }}>
                  {positions.map((pos, index) => (
                    <PositionMarker
                      key={index}
                      position={pos}
                      isActive={index === currentPositionIndex}
                      onClick={() => jumpToPosition(index)}
                      isLabeled={groundTruth.positions.some(
                        gt => gt.start_time === pos.start_time && gt.rating !== undefined
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
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    Position {currentPositionIndex + 1}/{positions.length}
                  </Typography>

                  <Box>
                    <IconButton size="small" onClick={goToPrevious} disabled={currentPositionIndex === 0}>
                      <NavigateBeforeIcon />
                    </IconButton>
                    <IconButton size="small" onClick={goToNext} disabled={currentPositionIndex >= positions.length - 1}>
                      <NavigateNextIcon />
                    </IconButton>
                  </Box>
                </Box>

                {currentPosition && (
                  <Box>
                    {/* AI Prediction */}
                    <Paper sx={{ p: 2, mb: 2, bgcolor: 'primary.main', color: 'white' }}>
                      <Typography variant="subtitle2" sx={{ color: 'rgba(255,255,255,0.9)', fontWeight: 600 }}>
                        AI Prediction
                      </Typography>

                      <Box sx={{ mt: 1 }}>
                        <Chip
                          label={`${currentPosition.start_time} - ${currentPosition.end_time}`}
                          size="small"
                          sx={{ mr: 1, mb: 0.5, bgcolor: 'rgba(255,255,255,0.2)', color: 'white', fontWeight: 500 }}
                        />
                        <Chip
                          label={currentPosition.position}
                          sx={{ mr: 1, mb: 0.5, bgcolor: 'rgba(255,255,255,0.3)', color: 'white', fontWeight: 600 }}
                        />
                        {currentPosition.sub_position && (
                          <Chip
                            label={currentPosition.sub_position}
                            size="small"
                            sx={{ mb: 0.5, bgcolor: 'rgba(255,255,255,0.15)', color: 'white' }}
                          />
                        )}
                      </Box>

                      <Typography variant="body2" sx={{ mt: 1.5, color: 'white' }}>
                        <strong>Confidence:</strong> {(currentPosition.confidence * 100).toFixed(0)}%
                      </Typography>
                    </Paper>

                    <Divider sx={{ my: 2 }} />

                    {/* Rating Scale */}
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Rate AI Prediction (0-5)
                      </Typography>
                      <Rating
                        value={rating}
                        onChange={(e, newValue) => setRating(newValue)}
                        max={5}
                        size="large"
                        sx={{ mb: 1 }}
                      />
                      {rating !== null && ratingScale[rating] && (
                        <Typography variant="caption" color="text.secondary">
                          {rating}: {ratingScale[rating].label} - {ratingScale[rating].description}
                        </Typography>
                      )}
                    </Box>

                    {/* Labels Picker */}
                    <Box sx={{ mb: 3 }}>
                      <Autocomplete
                        multiple
                        options={allLabels}
                        value={selectedLabels}
                        onChange={(e, newValue) => setSelectedLabels(newValue)}
                        renderInput={(params) => (
                          <TextField
                            {...params}
                            label="Labels"
                            placeholder="Add labels..."
                            helperText="Multi-select from ontology"
                          />
                        )}
                        renderTags={(value, getTagProps) =>
                          value.map((option, index) => (
                            <Chip
                              label={option}
                              size="small"
                              {...getTagProps({ index })}
                            />
                          ))
                        }
                      />
                    </Box>

                    {/* Notes */}
                    <Box sx={{ mb: 3 }}>
                      <TextField
                        fullWidth
                        label="Notes"
                        multiline
                        rows={3}
                        value={notes}
                        onChange={(e) => setNotes(e.target.value)}
                        placeholder="Detailed feedback about this segment..."
                        helperText="Notes are auto-saved"
                      />
                    </Box>

                    {/* Sub-segments */}
                    <Box sx={{ mb: 3 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="subtitle2">
                          Sub-segments ({subSegments.length})
                        </Typography>
                        <Button
                          size="small"
                          startIcon={<AddIcon />}
                          onClick={addSubSegment}
                        >
                          Add
                        </Button>
                      </Box>

                      {subSegments.map((sub, index) => (
                        <Paper key={index} sx={{ p: 1.5, mb: 1.5, bgcolor: 'grey.50' }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant="caption" fontWeight={600}>
                              Sub-segment {index + 1}
                            </Typography>
                            <IconButton size="small" onClick={() => deleteSubSegment(index)}>
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Box>

                          <Grid container spacing={1}>
                            <Grid item xs={6}>
                              <TextField
                                fullWidth
                                size="small"
                                label="Start"
                                value={sub.start_time}
                                onChange={(e) => updateSubSegment(index, 'start_time', e.target.value)}
                                placeholder="00:00"
                              />
                            </Grid>
                            <Grid item xs={6}>
                              <TextField
                                fullWidth
                                size="small"
                                label="End"
                                value={sub.end_time}
                                onChange={(e) => updateSubSegment(index, 'end_time', e.target.value)}
                                placeholder="00:05"
                              />
                            </Grid>
                            <Grid item xs={12}>
                              <TextField
                                fullWidth
                                size="small"
                                label="Note"
                                value={sub.note}
                                onChange={(e) => updateSubSegment(index, 'note', e.target.value)}
                                placeholder="e.g., Collar tie snap down attempt"
                              />
                            </Grid>
                            <Grid item xs={12}>
                              <Autocomplete
                                multiple
                                options={allLabels}
                                value={sub.labels}
                                onChange={(e, newValue) => updateSubSegment(index, 'labels', newValue)}
                                size="small"
                                renderInput={(params) => (
                                  <TextField
                                    {...params}
                                    label="Labels"
                                    placeholder="Add labels..."
                                  />
                                )}
                                renderTags={(value, getTagProps) =>
                                  value.map((option, idx) => (
                                    <Chip
                                      label={option}
                                      size="small"
                                      {...getTagProps({ index: idx })}
                                    />
                                  ))
                                }
                              />
                            </Grid>
                          </Grid>
                        </Paper>
                      ))}
                    </Box>
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

export default VideoEvaluatorV2;
