import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
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
  Autocomplete,
  Rating,
  Divider,
  Alert,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore';
import SaveIcon from '@mui/icons-material/Save';
import AssessmentIcon from '@mui/icons-material/Assessment';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import LoopIcon from '@mui/icons-material/Loop';

import VideoPlayer from './VideoPlayer';
import PositionMarker from './PositionMarker';
import AthleteProfiles from './AthleteProfiles';
import KeyMomentsPanel from './KeyMomentsPanel';
import ActionScoreIndicator from './ActionScoreIndicator';
import config from '../config';

const API_BASE_URL = config.api.baseUrl || 'http://localhost:5002';

// Hardcoded BJJ labels - no API needed
const ALL_LABELS = ["50_50_guard","Back Control","Guard","Half Guard","Knee on Belly","Mount","North South","Side Control","Standing","Transitional","Turtle","advantage","aggressive_turtle","americana","anaconda","ankle_lock","ankle_pick","arm_triangle","armbar","attacking","back_control","back_control_to_mount","body_lock","body_triangle","bow","bow_and_arrow","bridge_and_roll","butterfly_guard","butterfly_sweep","closed_guard","collar_tie","controlling","crucifix","darce","de_la_riva","de_la_riva_sweep","deep_half","deep_half_guard","defending","defensive_turtle","double_leg","dump","elbow_escape","escaping","ezekiel","front_headlock","full_mount","gift_wrap","granby_roll","grip_fighting","guard_pass","guard_to_back_control","guard_to_half_guard","guard_to_mount","guard_to_side_control","guillotine","half_guard_bottom","half_guard_to_back_control","half_guard_to_guard","half_guard_to_mount","half_guard_to_side_control","half_guard_top","handshake","headquarters_position","heel_hook","high_crotch","high_mount","hip_bump_sweep","hooks_in","kesa_gatame","kimura","knee_bar","knee_on_belly","knee_shield","knee_slice","lasso_guard","leg_drag","lockdown","long_step_pass","low_mount","match_end","match_start","mid_transition","modified_kesa_gatame","mount","mount_to_back_control","neutral_stance","one_hook_in","open_guard","out_of_bounds","over_under_pass","overhooks","passing","penalty","points_scored","pre_match","pull_guard","quarter_guard","rear_mount","rear_naked_choke","referee_restart","referee_stop","reset","retaining_guard","reversal","reverse_de_la_riva","reverse_knee_on_belly","reverse_north_south","reverse_side_control","s_mount","sacrifice_throw","scissor_sweep","scramble","scramble_reversal","seat_belt","side_control_to_knee_on_belly","side_control_to_mount","side_control_to_north_south","single_leg","single_leg_x","sit_up_escape","snap_down","spider_guard","stack_pass","stalling","standard_knee_on_belly","standard_north_south","standard_side_control","standing_to_back_control","standing_to_guard","standing_to_mount","standing_to_side_control","submission_attempt","submission_success","sweep","takedown","technical_mount","technical_standup","throw","toe_hold","toreando_pass","triangle","trip","turtle_to_back_control","turtle_to_side_control","underhooks","unstable_position","x_guard","x_guard_sweep","x_pass","z_guard"];

const RATING_SCALE = {
  0: { label: "Completely Wrong", description: "All aspects incorrect" },
  1: { label: "Mostly Wrong", description: "Major errors" },
  2: { label: "Partially Wrong", description: "Position correct but details wrong" },
  3: { label: "Mostly Correct", description: "Minor details missing" },
  4: { label: "Almost Perfect", description: "Trivial details off" },
  5: { label: "Perfect", description: "Complete accuracy" }
};

const VideoEvaluatorV3 = () => {
  // State
  const [experiments, setExperiments] = useState([]);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [experimentData, setExperimentData] = useState(null);
  const [experimentFormat, setExperimentFormat] = useState('exp1_2'); // 'exp1_2', 'exp3', or 'exp4'
  const [athleteProfiles, setAthleteProfiles] = useState({});
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

  // Feedback segments (the key change - this is now the primary feedback format)
  const [feedbackSegments, setFeedbackSegments] = useState([]);
  const [rating, setRating] = useState(null);
  const [isPositionCompleted, setIsPositionCompleted] = useState(false);

  // UI state
  const [saveStatus, setSaveStatus] = useState('');
  const [validationErrors, setValidationErrors] = useState([]);

  // Video controls
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [isLooping, setIsLooping] = useState(false);
  const [pauseAtBoundaries, setPauseAtBoundaries] = useState(true);

  // Refs
  const saveTimeoutRef = useRef(null);
  const videoPlayerRef = useRef(null);
  const pendingSeekRef = useRef(null);
  const isManualNavigationRef = useRef(false);

  // Load data on mount
  useEffect(() => {
    loadExperiments();
    loadGroundTruth();
  }, []);

  // Auto-select position based on current video time (disabled during manual navigation)
  useEffect(() => {
    if (!experimentData || !experimentData.analysis || !experimentData.analysis.position_timeline) {
      return;
    }

    // Skip auto-select during manual navigation
    if (isManualNavigationRef.current) {
      return;
    }

    const positions = experimentData.analysis.position_timeline;

    // Check if we're near the end of the current segment
    // This prevents accidental auto-switching when video overshoots segment boundary before pausing
    if (currentPositionIndex >= 0 && currentPositionIndex < positions.length) {
      const currentPos = positions[currentPositionIndex];
      const currentEndSeconds = timeToSeconds(currentPos.end_time);

      // If we're past the segment end but within 0.5s of it, don't auto-switch
      // The video should pause soon and user hasn't manually navigated
      if (currentTime >= currentEndSeconds && currentTime <= currentEndSeconds + 0.5) {
        return;
      }
    }

    const positionIndex = positions.findIndex(pos => {
      const startSeconds = timeToSeconds(pos.start_time);
      const endSeconds = timeToSeconds(pos.end_time);
      return currentTime >= startSeconds && currentTime <= endSeconds;
    });

    if (positionIndex !== -1 && positionIndex !== currentPositionIndex) {
      setCurrentPositionIndex(positionIndex);
      loadFeedbackForPosition(positionIndex);
    }
  }, [currentTime, experimentData, currentPositionIndex]);

  // Handle pending seeks after position index updates
  useEffect(() => {
    if (pendingSeekRef.current !== null && videoPlayerRef.current) {
      videoPlayerRef.current.seekTo(pendingSeekRef.current);
      setCurrentTime(pendingSeekRef.current);
      pendingSeekRef.current = null;

      // Re-enable auto-select after a short delay to let the seek complete
      setTimeout(() => {
        isManualNavigationRef.current = false;
      }, 300);
    }
  }, [currentPositionIndex]);

  // Auto-save when feedback changes (debounced) - only after at least 2 positions are labeled
  useEffect(() => {
    // Only auto-save if user has labeled at least 2 positions
    if (labeledCount >= 2 && (rating !== null || feedbackSegments.length > 0)) {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }

      setSaveStatus('saving');
      saveTimeoutRef.current = setTimeout(() => {
        autoSaveCurrentFeedback();
      }, 2000);
    }

    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [rating, feedbackSegments, labeledCount]);

  const loadExperiments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/experiments/list?results_dir=results`);
      const data = await response.json();
      const sortedExperiments = (data.experiments || []).sort((a, b) => {
        const nameA = (a.metadata?.experiment_name || a.name).toLowerCase();
        const nameB = (b.metadata?.experiment_name || b.name).toLowerCase();
        return nameA.localeCompare(nameB);
      });
      setExperiments(sortedExperiments);
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

      // Detect and set experiment format
      const format = data.format || data.analysis?.experiment_format || 'exp1_2';
      setExperimentFormat(format);

      // Load athlete profiles if Exp3
      if (data.analysis?.athlete_profiles) {
        setAthleteProfiles(data.analysis.athlete_profiles);
      } else {
        setAthleteProfiles({});
      }

      // Handle video path - try multiple sources
      let videoPathToUse = null;
      if (data.metadata && data.metadata.video_path) {
        videoPathToUse = data.metadata.video_path;
      } else if (data.analysis?.video_metadata?.video_path) {
        videoPathToUse = data.analysis.video_metadata.video_path;
      } else if (data.analysis?.meta?.video_path) {
        // v3 format stores video_path in meta
        videoPathToUse = data.analysis.meta.video_path;
      }

      if (videoPathToUse) {
        const streamingUrl = `${API_BASE_URL}/videos/stream?video_path=${encodeURIComponent(videoPathToUse)}`;
        setVideoPath(streamingUrl);
        console.log('[DEBUG] Video path loaded:', videoPathToUse);
      } else {
        console.log('[DEBUG] No video path found in:', { metadata: data.metadata, video_metadata: data.analysis?.video_metadata, meta: data.analysis?.meta });
      }
      
      console.log('[DEBUG] Experiment loaded:', {
        name: experimentName,
        format: format,
        segmentsCount: data.analysis?.position_timeline?.length || 0,
        hasVideoPath: !!videoPathToUse,
        meta: data.analysis?.meta
      });

      // Load experiment-specific ground truth
      if (data.ground_truth) {
        setGroundTruth({
          positions: data.ground_truth.ground_truth_positions || [],
          transitions: data.ground_truth.ground_truth_transitions || [],
          scoring: data.ground_truth.ground_truth_scoring_adcc || {}
        });

        // Count completed positions
        const labeled = data.ground_truth.ground_truth_positions?.filter(p => p.completed === true).length || 0;
        setLabeledCount(labeled);
      } else {
        // Reset ground truth if none exists for this experiment
        setGroundTruth({
          positions: [],
          transitions: [],
          scoring: {}
        });
        setLabeledCount(0);
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

        // Count completed positions (not just those with ratings)
        const labeled = data.ground_truth_positions?.filter(p => p.completed === true).length || 0;
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

    // Clear any previous validation errors when loading a new position
    setValidationErrors([]);

    // Check if we have saved feedback for this position
    const savedFeedback = groundTruth.positions.find(
      gt => gt.start_time === currentPos.start_time && gt.end_time === currentPos.end_time
    );

    console.log(`[DEBUG] Loading position ${index}:`, {
      start_time: currentPos.start_time,
      end_time: currentPos.end_time,
      hasSavedFeedback: !!savedFeedback,
      completedStatus: savedFeedback?.completed,
      rating: savedFeedback?.rating,
      subSegmentsCount: savedFeedback?.sub_segments?.length || 0
    });

    // Set completed status from saved feedback
    setIsPositionCompleted(savedFeedback?.completed || false);

    if (savedFeedback && savedFeedback.sub_segments && savedFeedback.sub_segments.length > 0) {
      // Load saved sub-segments as feedback segments
      setFeedbackSegments(savedFeedback.sub_segments);
      setRating(savedFeedback.rating || null);
    } else {
      // Initialize with one segment covering the entire position
      setFeedbackSegments([{
        start_time: currentPos.start_time,
        end_time: currentPos.end_time,
        labels: [],
        note: ''
      }]);
      setRating(savedFeedback?.rating || null);
    }
  };

  const autoSaveCurrentFeedback = async () => {
    if (!experimentData || currentPositionIndex < 0) return;

    const positions = experimentData.analysis.position_timeline || [];
    const currentPos = positions[currentPositionIndex];

    if (!currentPos) return;

    // Validate before saving
    const errors = validateFeedback(currentPos);
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
      sub_segments: feedbackSegments,
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

  const validateFeedback = (currentPos) => {
    const errors = [];
    const parentStart = timeToSeconds(currentPos.start_time);
    const parentEnd = timeToSeconds(currentPos.end_time);

    if (feedbackSegments.length === 0) {
      errors.push('At least one feedback segment is required');
      return errors;
    }

    // Sort segments by start time for validation
    const sortedSegments = [...feedbackSegments].sort((a, b) =>
      timeToSeconds(a.start_time) - timeToSeconds(b.start_time)
    );

    // Check each segment
    sortedSegments.forEach((seg, index) => {
      const segStart = timeToSeconds(seg.start_time);
      const segEnd = timeToSeconds(seg.end_time);

      // Validate that segment has either labels or notes
      const hasLabels = seg.labels && seg.labels.length > 0;
      const hasNotes = seg.note && seg.note.trim().length > 0;

      if (!hasLabels && !hasNotes) {
        errors.push(`Segment ${index + 1} must have at least one label or a note`);
      }

      // Check segment is within parent bounds
      if (segStart < parentStart) {
        errors.push(`Segment ${index + 1} starts before the position start time`);
      }
      if (segEnd > parentEnd) {
        errors.push(`Segment ${index + 1} ends after the position end time (${seg.end_time} > ${currentPos.end_time})`);
      }

      // Check segment has valid duration
      if (segEnd <= segStart) {
        errors.push(`Segment ${index + 1} has invalid time range`);
      }

      // Check for overlaps with next segment
      if (index < sortedSegments.length - 1) {
        const nextSegStart = timeToSeconds(sortedSegments[index + 1].start_time);
        if (segEnd > nextSegStart) {
          errors.push(`Segment ${index + 1} overlaps with segment ${index + 2}`);
        }
      }
    });

    // Check for complete coverage
    if (sortedSegments.length > 0) {
      const firstStart = timeToSeconds(sortedSegments[0].start_time);
      const lastEnd = timeToSeconds(sortedSegments[sortedSegments.length - 1].end_time);

      if (firstStart > parentStart) {
        const gap = secondsToTime(firstStart - parentStart);
        errors.push(`Gap at start: ${currentPos.start_time} to ${sortedSegments[0].start_time} (${gap}) needs feedback`);
      }

      if (lastEnd < parentEnd) {
        const gap = secondsToTime(parentEnd - lastEnd);
        errors.push(`Gap at end: ${sortedSegments[sortedSegments.length - 1].end_time} to ${currentPos.end_time} (${gap}) needs feedback`);
      }

      // Check for gaps between segments
      for (let i = 0; i < sortedSegments.length - 1; i++) {
        const currentEnd = timeToSeconds(sortedSegments[i].end_time);
        const nextStart = timeToSeconds(sortedSegments[i + 1].start_time);
        if (nextStart > currentEnd) {
          const gap = secondsToTime(nextStart - currentEnd);
          errors.push(`Gap between segment ${i + 1} and ${i + 2}: ${sortedSegments[i].end_time} to ${sortedSegments[i + 1].start_time} (${gap}) needs feedback`);
        }
      }
    }

    return errors;
  };

  const addFeedbackSegment = () => {
    const positions = experimentData?.analysis?.position_timeline || [];
    const currentPos = positions[currentPositionIndex];

    if (!currentPos) return;

    // Find the end time of the last segment
    let newStartTime = currentPos.start_time;
    if (feedbackSegments.length > 0) {
      const lastSegment = feedbackSegments[feedbackSegments.length - 1];
      newStartTime = lastSegment.end_time;
    }

    // New segment starts from last segment's end, goes to position end
    setFeedbackSegments([...feedbackSegments, {
      start_time: newStartTime,
      end_time: currentPos.end_time,
      labels: [],
      note: ''
    }]);
  };

  const updateFeedbackSegment = (index, field, value) => {
    const updated = [...feedbackSegments];
    updated[index] = { ...updated[index], [field]: value };

    // If end_time changed and this isn't the last segment, update next segment's start_time
    if (field === 'end_time' && index < feedbackSegments.length - 1) {
      updated[index + 1] = { ...updated[index + 1], start_time: value };
    }

    setFeedbackSegments(updated);
  };

  const deleteFeedbackSegment = (index) => {
    if (feedbackSegments.length === 1) {
      alert('Cannot delete the only feedback segment');
      return;
    }
    setFeedbackSegments(feedbackSegments.filter((_, i) => i !== index));
  };

  const saveGroundTruth = async (gtOverride = null) => {
    try {
      // Use provided ground truth or fall back to state
      const gt = gtOverride || groundTruth;

      // Get experiment-specific ground truth path
      const outputPath = experimentData?.ground_truth_path ||
                        (experimentData?.experiment_path ?
                          `${experimentData.experiment_path}/ground_truth.json` :
                          'evaluation/ground_truth.json');

      const gtData = {
        video_path: experimentData?.analysis?.video_metadata?.video_path || videoPath,
        video_duration: experimentData?.analysis?.video_metadata?.duration || '00:00:00',
        labeled_by: 'dhruva',
        labeled_date: new Date().toISOString(),
        ground_truth_positions: gt.positions,
        ground_truth_transitions: gt.transitions,
        ground_truth_scoring_adcc: gt.scoring
      };

      const response = await fetch(`${API_BASE_URL}/ground_truth/save?output_path=${encodeURIComponent(outputPath)}`, {
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

  const saveCurrentPositionToState = (markAsCompleted = null) => {
    if (currentPositionIndex < 0 || !experimentData) return null;

    const positions = experimentData.analysis.position_timeline || [];
    const currentPos = positions[currentPositionIndex];
    if (!currentPos) return null;

    const newGroundTruth = { ...groundTruth };
    const existingIndex = newGroundTruth.positions.findIndex(
      gt => gt.start_time === currentPos.start_time && gt.end_time === currentPos.end_time
    );

    // Preserve existing completed status if not explicitly changing it
    // Use isPositionCompleted state as the source of truth, fallback to existing if needed
    const existingCompleted = existingIndex >= 0 ? newGroundTruth.positions[existingIndex].completed : false;
    const completedStatus = markAsCompleted !== null ? markAsCompleted : (isPositionCompleted || existingCompleted);

    // Save current feedback to ground truth state
    const updatedPosition = {
      ...currentPos,
      rating,
      sub_segments: feedbackSegments,
      completed: completedStatus,
      last_updated: new Date().toISOString()
    };

    if (existingIndex >= 0) {
      newGroundTruth.positions[existingIndex] = updatedPosition;
    } else {
      newGroundTruth.positions.push(updatedPosition);
    }

    setGroundTruth(newGroundTruth);

    console.log(`[DEBUG] Saved position to state:`, {
      start_time: currentPos.start_time,
      end_time: currentPos.end_time,
      completedStatus: completedStatus,
      markAsCompleted: markAsCompleted,
      existingCompleted: existingCompleted,
      rating: rating,
      subSegmentsCount: feedbackSegments.length
    });

    // Update labeled count (now counts completed positions)
    const labeled = newGroundTruth.positions.filter(p => p.completed === true).length;
    setLabeledCount(labeled);

    // Return the updated ground truth for immediate use
    return newGroundTruth;
  };

  const goToNext = (skipSave = false) => {
    // Save current position before navigating (unless explicitly skipped)
    if (!skipSave) {
      saveCurrentPositionToState();
    }

    const positions = experimentData?.analysis?.position_timeline || [];
    if (currentPositionIndex < positions.length - 1) {
      const nextIndex = currentPositionIndex + 1;
      const nextPos = positions[nextIndex];

      if (nextPos.start_time) {
        const seconds = timeToSeconds(nextPos.start_time);
        isManualNavigationRef.current = true;
        pendingSeekRef.current = seconds;
      }

      setCurrentPositionIndex(nextIndex);
      loadFeedbackForPosition(nextIndex);
    }
  };

  const goToPrevious = () => {
    // Save current position before navigating
    saveCurrentPositionToState();

    if (currentPositionIndex > 0) {
      const prevIndex = currentPositionIndex - 1;
      const positions = experimentData?.analysis?.position_timeline || [];
      const prevPos = positions[prevIndex];

      if (prevPos.start_time) {
        const seconds = timeToSeconds(prevPos.start_time);
        isManualNavigationRef.current = true;
        pendingSeekRef.current = seconds;
      }

      setCurrentPositionIndex(prevIndex);
      loadFeedbackForPosition(prevIndex);
    }
  };

  const jumpToPosition = (index) => {
    console.log(`[DEBUG] jumpToPosition called:`, {
      fromIndex: currentPositionIndex,
      toIndex: index
    });

    // Don't save if jumping to same position or from invalid position
    if (currentPositionIndex >= 0 && currentPositionIndex !== index) {
      console.log(`[DEBUG] Saving current position ${currentPositionIndex} before jumping`);
      saveCurrentPositionToState();
    }

    const positions = experimentData?.analysis?.position_timeline || [];
    const pos = positions[index];

    if (pos.start_time) {
      const seconds = timeToSeconds(pos.start_time);
      isManualNavigationRef.current = true;
      pendingSeekRef.current = seconds;
    }

    setCurrentPositionIndex(index);
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
      // Don't trigger if user is typing
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
      }

      // Get current segment boundaries
      const positions = experimentData?.analysis?.position_timeline || [];
      const currentPos = positions[currentPositionIndex];

      if (!currentPos) return;

      const segmentStart = timeToSeconds(currentPos.start_time);
      const segmentEnd = timeToSeconds(currentPos.end_time);

      switch (e.key) {
        case ' ': // Spacebar for play/pause
          e.preventDefault();
          videoPlayerRef.current?.togglePlayPause();
          break;
        case 'ArrowLeft': // Left arrow for -1 second (within segment bounds)
          e.preventDefault();
          const newTimeLeft = Math.max(segmentStart, currentTime - 1);
          setCurrentTime(newTimeLeft);
          videoPlayerRef.current?.seekTo(newTimeLeft);
          break;
        case 'ArrowRight': // Right arrow for +1 second (within segment bounds)
          e.preventDefault();
          const newTimeRight = Math.min(segmentEnd, currentTime + 1);
          setCurrentTime(newTimeRight);
          videoPlayerRef.current?.seekTo(newTimeRight);
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentTime, experimentData, currentPositionIndex]);

  // Render
  const positions = experimentData?.analysis?.position_timeline || [];
  const currentPosition = positions[currentPositionIndex];
  const progress = positions.length > 0 ? (labeledCount / positions.length) * 100 : 0;

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        BJJ Analysis Evaluation (HITL) v3.0
      </Typography>

      {/* Experiment Selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel id="experiment-select-label">Select Experiment</InputLabel>
                <Select
                  labelId="experiment-select-label"
                  value={selectedExperiment || ''}
                  onChange={(e) => loadExperimentData(e.target.value)}
                  label="Select Experiment"
                >
                  {experiments.map((exp) => (
                    <MenuItem key={exp.name} value={exp.name}>
                      {exp.metadata?.experiment_name || exp.name}
                      {exp.metadata?.type === 'exp3' && (
                        <Chip label="Exp3" size="small" color="secondary" sx={{ ml: 1 }} />
                      )}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

          </Grid>

          {experimentData && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" gutterBottom>
                Review Progress: {labeledCount} / {positions.length} positions ({Math.round(progress)}%)
                {experimentFormat === 'exp3' && (
                  <Chip label="Experiment 3" size="small" color="secondary" sx={{ ml: 1 }} />
                )}
                {experimentFormat === 'exp4' && (
                  <Chip label="Experiment 4" size="small" color="primary" sx={{ ml: 1 }} />
                )}
              </Typography>
              <LinearProgress variant="determinate" value={progress} />
              {experimentFormat === 'exp4' && experimentData.analysis?.meta && (
                <Box sx={{ mt: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    {experimentData.analysis.meta.total_segments ? (
                      <>
                        {experimentData.analysis.meta.total_segments} segments total 
                        ({experimentData.analysis.meta.segments_with_detail} with detail, 
                        {experimentData.analysis.meta.segments_skeleton_only} skeleton only)
                      </>
                    ) : (
                      <>
                        {positions.length} segments (v3 skeleton + micro-analysis)
                      </>
                    )}
                  </Typography>
                </Box>
              )}
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Athlete Profiles (Exp3 only) */}
      {experimentFormat === 'exp3' && <AthleteProfiles athleteProfiles={athleteProfiles} />}

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
                  segmentStart={currentPosition ? timeToSeconds(currentPosition.start_time) : null}
                  segmentEnd={currentPosition ? timeToSeconds(currentPosition.end_time) : null}
                  pauseAtSegmentEnd={pauseAtBoundaries}
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
                      <MenuItem value={2.0}>2x</MenuItem>
                    </Select>
                  </FormControl>

                  <Button
                    size="small"
                    variant={isLooping ? 'contained' : 'outlined'}
                    startIcon={<LoopIcon />}
                    onClick={() => setIsLooping(!isLooping)}
                  >
                    Loop
                  </Button>

                  <Button
                    size="small"
                    variant={pauseAtBoundaries ? 'contained' : 'outlined'}
                    onClick={() => setPauseAtBoundaries(!pauseAtBoundaries)}
                  >
                    Pause at Boundaries
                  </Button>

                  <Typography variant="caption" sx={{ ml: 'auto', color: 'text.secondary' }}>
                    Space: Play/Pause | ← →: 1s jumps
                  </Typography>
                </Box>

                {/* Timeline */}
                <Box sx={{ mt: 2 }}>
                  {positions.map((pos, index) => (
                    <PositionMarker
                      key={index}
                      position={pos}
                      isActive={index === currentPositionIndex}
                      onClick={() => jumpToPosition(index)}
                      isLabeled={groundTruth.positions.some(gt =>
                        gt.start_time === pos.start_time && gt.completed === true
                      )}
                    />
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Feedback Panel */}
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
                      <Typography variant="subtitle2" sx={{ color: 'rgba(255,255,255,0.9)', fontWeight: 600, mb: 1 }}>
                        AI Prediction
                      </Typography>
                      <Box>
                        <Chip
                          label={`${currentPosition.start_time} - ${currentPosition.end_time}`}
                          size="small"
                          sx={{ mr: 1, mb: 0.5, bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }}
                        />
                        <Chip
                          label={currentPosition.position}
                          sx={{ mr: 1, mb: 0.5, bgcolor: 'rgba(255,255,255,0.3)', color: 'white', fontWeight: 600 }}
                        />
                        {currentPosition.sub_position && currentPosition.sub_position !== 'N/A' && (
                          <Chip
                            label={currentPosition.sub_position}
                            size="small"
                            sx={{ mb: 0.5, bgcolor: 'rgba(255,255,255,0.15)', color: 'white' }}
                          />
                        )}
                      </Box>
                      {/* Exp3: Athletes */}
                      {experimentFormat === 'exp3' && (currentPosition.top_athlete || currentPosition.bottom_athlete) && (
                        <Box sx={{ mt: 1.5 }}>
                          <Typography variant="caption" sx={{ opacity: 0.8, display: 'block', mb: 0.5 }}>
                            Athletes:
                          </Typography>
                          {currentPosition.top_athlete && (
                            <Chip
                              label={`Top: ${currentPosition.top_athlete}`}
                              size="small"
                              sx={{ mr: 0.5, mb: 0.5, bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }}
                            />
                          )}
                          {currentPosition.bottom_athlete && (
                            <Chip
                              label={`Bottom: ${currentPosition.bottom_athlete}`}
                              size="small"
                              sx={{ mb: 0.5, bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }}
                            />
                          )}
                        </Box>
                      )}
                      {/* Exp3: Action Score & Transition */}
                      {experimentFormat === 'exp3' && (
                        <ActionScoreIndicator
                          avgAction={currentPosition.avg_action}
                          transition={currentPosition.transition}
                        />
                      )}
                      {/* Exp4: Action Score, Control, Reasons, Focus */}
                      {experimentFormat === 'exp4' && (
                        <Box sx={{ mt: 1.5 }}>
                          {currentPosition.top_athlete && currentPosition.top_athlete !== '-' && (
                            <Box sx={{ mb: 1 }}>
                              <Chip
                                label={`Top: ${currentPosition.top_athlete}`}
                                size="small"
                                sx={{ bgcolor: 'rgba(255,255,255,0.3)', color: 'white', fontWeight: 600 }}
                              />
                            </Box>
                          )}
                          <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                            {currentPosition.action_score !== undefined && (
                              <Chip
                                label={`Action: ${currentPosition.action_score}`}
                                size="small"
                                sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }}
                              />
                            )}
                            {currentPosition.confidence !== undefined && (
                              <Chip
                                label={`Control: ${currentPosition.confidence}`}
                                size="small"
                                sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }}
                              />
                            )}
                          </Box>
                          {currentPosition.reasons && (
                            <Typography variant="caption" sx={{ opacity: 0.9, display: 'block', mb: 0.5 }}>
                              <strong>Reasons:</strong> {currentPosition.reasons}
                            </Typography>
                          )}
                          {currentPosition.focus && currentPosition.focus !== '-' && (
                            <Typography variant="caption" sx={{ opacity: 0.9, display: 'block', mb: 0.5 }}>
                              <strong>Focus:</strong> {currentPosition.focus}
                            </Typography>
                          )}
                          {currentPosition.notes && (
                            <Typography variant="caption" sx={{ opacity: 0.9, display: 'block' }}>
                              <strong>Notes:</strong> {currentPosition.notes}
                            </Typography>
                          )}
                          {/* Exp4 v3: Micro-analysis fields (strategy, setup, execution, outcome, coaching) */}
                          {(currentPosition.strategy && currentPosition.strategy !== 'N/A') && (
                            <Box sx={{ mt: 1.5, p: 1.5, bgcolor: 'rgba(255,255,255,0.05)', borderRadius: 1, border: '1px solid rgba(255,255,255,0.1)' }}>
                              <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600, color: 'secondary.main' }}>
                                📊 Micro-Analysis
                              </Typography>
                              {currentPosition.strategy && currentPosition.strategy !== 'N/A' && (
                                <Box sx={{ mb: 1.5 }}>
                                  <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.5, color: 'primary.light' }}>
                                    Strategy:
                                  </Typography>
                                  <Typography variant="body2" sx={{ opacity: 0.95 }}>
                                    {currentPosition.strategy}
                                  </Typography>
                                </Box>
                              )}
                              {currentPosition.setup && currentPosition.setup !== 'N/A' && (
                                <Box sx={{ mb: 1.5 }}>
                                  <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.5, color: 'primary.light' }}>
                                    Setup:
                                  </Typography>
                                  <Typography variant="body2" sx={{ opacity: 0.95 }}>
                                    {currentPosition.setup}
                                  </Typography>
                                </Box>
                              )}
                              {currentPosition.execution && currentPosition.execution !== 'N/A' && (
                                <Box sx={{ mb: 1.5 }}>
                                  <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.5, color: 'primary.light' }}>
                                    Execution:
                                  </Typography>
                                  <Typography variant="body2" sx={{ opacity: 0.95 }}>
                                    {currentPosition.execution}
                                  </Typography>
                                </Box>
                              )}
                              {currentPosition.outcome && currentPosition.outcome !== 'N/A' && (
                                <Box sx={{ mb: 1.5 }}>
                                  <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.5, color: 'primary.light' }}>
                                    Outcome:
                                  </Typography>
                                  <Typography variant="body2" sx={{ opacity: 0.95 }}>
                                    {currentPosition.outcome}
                                  </Typography>
                                </Box>
                              )}
                              {currentPosition.coaching && currentPosition.coaching !== 'N/A' && (
                                <Box>
                                  <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.5, color: 'secondary.main' }}>
                                    💡 Coaching:
                                  </Typography>
                                  <Typography variant="body2" sx={{ opacity: 0.95, fontStyle: 'italic' }}>
                                    {currentPosition.coaching}
                                  </Typography>
                                </Box>
                              )}
                            </Box>
                          )}
                          {/* Exp4 combined format: Detailed Analysis from Stage 2 */}
                          {currentPosition.has_detail && currentPosition.detail && (
                            <Box sx={{ mt: 1, p: 1, bgcolor: 'rgba(255,255,255,0.1)', borderRadius: 1 }}>
                              <Typography variant="caption" sx={{ opacity: 0.8, display: 'block', mb: 0.5, fontWeight: 600 }}>
                                Detailed Analysis:
                              </Typography>
                              {currentPosition.detail.setup && (
                                <Typography variant="caption" sx={{ opacity: 0.9, display: 'block', mb: 0.5 }}>
                                  <strong>Setup:</strong> {currentPosition.detail.setup}
                                </Typography>
                              )}
                              {currentPosition.detail.execution && (
                                <Typography variant="caption" sx={{ opacity: 0.9, display: 'block', mb: 0.5 }}>
                                  <strong>Execution:</strong> {currentPosition.detail.execution}
                                </Typography>
                              )}
                              {currentPosition.detail.outcome && (
                                <Typography variant="caption" sx={{ opacity: 0.9, display: 'block', mb: 0.5 }}>
                                  <strong>Outcome:</strong> {currentPosition.detail.outcome}
                                </Typography>
                              )}
                              {currentPosition.detail.coaching && (
                                <Typography variant="caption" sx={{ opacity: 0.9, display: 'block' }}>
                                  <strong>Coaching:</strong> {currentPosition.detail.coaching}
                                </Typography>
                              )}
                            </Box>
                          )}
                        </Box>
                      )}
                      {/* Exp3: Key Actions */}
                      {experimentFormat === 'exp3' && currentPosition.key_actions && currentPosition.key_actions.length > 0 && (
                        <Box sx={{ mt: 1.5 }}>
                          <Typography variant="caption" sx={{ opacity: 0.8, display: 'block', mb: 0.5 }}>
                            Key Actions:
                          </Typography>
                          {currentPosition.key_actions.map((action, idx) => (
                            <Chip
                              key={idx}
                              label={action.replace(/_/g, ' ')}
                              size="small"
                              sx={{ mr: 0.5, mb: 0.5, bgcolor: 'rgba(255,255,255,0.25)', color: 'white' }}
                            />
                          ))}
                        </Box>
                      )}
                      {/* Exp2 Enhanced Fields - conditionally displayed */}
                      {(currentPosition.labels && currentPosition.labels.length > 0) && (
                        <Box sx={{ mt: 1.5 }}>
                          <Typography variant="caption" sx={{ opacity: 0.8, display: 'block', mb: 0.5 }}>
                            AI Labels:
                          </Typography>
                          {currentPosition.labels.map((label, idx) => (
                            <Chip
                              key={idx}
                              label={label}
                              size="small"
                              sx={{ mr: 0.5, mb: 0.5, bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }}
                            />
                          ))}
                        </Box>
                      )}
                      {(currentPosition.dominance_athlete || currentPosition.execution_quality) && (
                        <Box sx={{ mt: 1.5, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                          {currentPosition.dominance_athlete && (
                            <Chip
                              label={`Dominance: ${currentPosition.dominance_athlete} (${currentPosition.dominance_score || 'N/A'}/5)`}
                              size="small"
                              sx={{ bgcolor: 'rgba(255,255,255,0.25)', color: 'white', fontWeight: 500 }}
                            />
                          )}
                          {currentPosition.execution_quality && (
                            <Chip
                              label={`Quality: ${currentPosition.execution_quality}/5`}
                              size="small"
                              sx={{ bgcolor: 'rgba(255,255,255,0.25)', color: 'white', fontWeight: 500 }}
                            />
                          )}
                          {currentPosition.control_quality && (
                            <Chip
                              label={`Control: ${currentPosition.control_quality}/5`}
                              size="small"
                              sx={{ bgcolor: 'rgba(255,255,255,0.25)', color: 'white', fontWeight: 500 }}
                            />
                          )}
                        </Box>
                      )}
                      {currentPosition.notes && (
                        <Typography variant="body2" sx={{ mt: 1.5, fontStyle: 'italic', opacity: 0.9 }}>
                          "{currentPosition.notes}"
                        </Typography>
                      )}
                      {/* Exp3: Narrative */}
                      {experimentFormat === 'exp3' && currentPosition.narrative && (
                        <Typography variant="body2" sx={{ mt: 1.5, fontStyle: 'italic', opacity: 0.9 }}>
                          📖 {currentPosition.narrative}
                        </Typography>
                      )}
                    </Paper>

                    {/* Exp3: Key Moments Panel */}
                    {experimentFormat === 'exp3' && currentPosition.key_moments && (
                      <KeyMomentsPanel
                        keyMoments={currentPosition.key_moments}
                        onMomentClick={(seconds) => {
                          setCurrentTime(seconds);
                          videoPlayerRef.current?.seekTo(seconds);
                        }}
                      />
                    )}

                    <Divider sx={{ my: 2 }} />

                    {/* Overall Rating */}
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Overall Rating (0-5)
                      </Typography>
                      <Rating
                        value={rating}
                        onChange={(e, newValue) => setRating(newValue)}
                        max={5}
                        size="large"
                        sx={{ mb: 1 }}
                      />
                      {rating !== null && RATING_SCALE[rating] && (
                        <Typography variant="caption" color="text.secondary">
                          {rating}: {RATING_SCALE[rating].label} - {RATING_SCALE[rating].description}
                        </Typography>
                      )}
                    </Box>

                    <Divider sx={{ my: 2 }} />

                    {/* Feedback Segments */}
                    <Box sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                        <Typography variant="subtitle2" fontWeight={600}>
                          Detailed Feedback (Labels: {ALL_LABELS.length})
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Button
                            variant="outlined"
                            size="small"
                            startIcon={<AddIcon />}
                            onClick={addFeedbackSegment}
                          >
                            Add Segment
                          </Button>
                          <Button
                            variant="contained"
                            size="small"
                            startIcon={<SaveIcon />}
                            onClick={async () => {
                              // Validate current feedback
                              const positions = experimentData?.analysis?.position_timeline || [];
                              const currentPos = positions[currentPositionIndex];
                              if (currentPos) {
                                const errors = validateFeedback(currentPos);
                                if (errors.length > 0) {
                                  setValidationErrors(errors);
                                  return;
                                }
                              }

                              // Clear validation errors
                              setValidationErrors([]);

                              // Save and mark as completed (validation passed)
                              // Update local completed state first
                              setIsPositionCompleted(true);
                              // This returns the updated ground truth immediately
                              const updatedGroundTruth = saveCurrentPositionToState(true);

                              // Save to backend immediately with the updated data
                              await saveGroundTruth(updatedGroundTruth);

                              // Move to next position if not at end
                              if (currentPositionIndex < positions.length - 1) {
                                goToNext(true); // Skip double save
                              }
                            }}
                            disabled={
                              !rating &&
                              !feedbackSegments.some(seg =>
                                (seg.labels && seg.labels.length > 0) ||
                                (seg.note && seg.note.trim().length > 0)
                              )
                            }
                          >
                            {currentPositionIndex < (experimentData?.analysis?.position_timeline || []).length - 1
                              ? 'Save & Next'
                              : 'Save & Finish'}
                          </Button>
                        </Box>
                      </Box>

                      {[...feedbackSegments].reverse().map((segment, displayIndex) => {
                        const actualIndex = feedbackSegments.length - 1 - displayIndex;
                        return (
                        <Paper
                          key={actualIndex}
                          elevation={2}
                          sx={{
                            p: 1.5,
                            mb: 1.5,
                            bgcolor: 'background.paper',
                            border: '1px solid',
                            borderColor: 'divider',
                            '&:hover': {
                              borderColor: 'primary.main',
                              boxShadow: 2
                            }
                          }}
                        >
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant="caption" fontWeight={600} color="primary">
                              Segment {actualIndex + 1} {displayIndex === 0 ? '(Latest)' : ''}
                            </Typography>
                            {feedbackSegments.length > 1 && (
                              <IconButton size="small" onClick={() => deleteFeedbackSegment(actualIndex)}>
                                <DeleteIcon fontSize="small" />
                              </IconButton>
                            )}
                          </Box>

                          <Grid container spacing={1}>
                            <Grid item xs={6}>
                              <TextField
                                fullWidth
                                size="small"
                                label="Start"
                                value={segment.start_time}
                                onChange={(e) => updateFeedbackSegment(actualIndex, 'start_time', e.target.value)}
                                disabled={actualIndex > 0}
                                helperText={actualIndex > 0 ? "Auto from prev" : ""}
                              />
                            </Grid>
                            <Grid item xs={6}>
                              <TextField
                                fullWidth
                                size="small"
                                label="End"
                                value={segment.end_time}
                                onChange={(e) => updateFeedbackSegment(actualIndex, 'end_time', e.target.value)}
                              />
                            </Grid>
                            <Grid item xs={12}>
                              <Autocomplete
                                multiple
                                options={ALL_LABELS}
                                value={segment.labels || []}
                                onChange={(e, newValue) => {
                                  updateFeedbackSegment(actualIndex, 'labels', newValue);
                                }}
                                size="small"
                                renderInput={(params) => (
                                  <TextField
                                    {...params}
                                    label={`Labels (${ALL_LABELS.length} available)`}
                                    placeholder="Select labels..."
                                  />
                                )}
                                renderTags={(value, getTagProps) =>
                                  value.map((option, idx) => {
                                    const { key, ...tagProps } = getTagProps({ index: idx });
                                    return (
                                      <Chip
                                        key={key}
                                        label={option}
                                        size="small"
                                        {...tagProps}
                                      />
                                    );
                                  })
                                }
                              />
                            </Grid>
                            <Grid item xs={12}>
                              <TextField
                                fullWidth
                                size="small"
                                multiline
                                rows={2}
                                label="Notes"
                                value={segment.note}
                                onChange={(e) => updateFeedbackSegment(actualIndex, 'note', e.target.value)}
                                placeholder="Detailed feedback..."
                              />
                            </Grid>
                          </Grid>
                        </Paper>
                        );
                      })}
                    </Box>

                    {/* Validation Errors */}
                    {validationErrors.length > 0 && (
                      <Alert severity="warning" sx={{ mt: 2 }} onClose={() => setValidationErrors([])}>
                        <Typography variant="subtitle2" fontWeight={600}>Validation Issues:</Typography>
                        <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                          {validationErrors.map((error, i) => (
                            <li key={i}><Typography variant="body2">{error}</Typography></li>
                          ))}
                        </ul>
                      </Alert>
                    )}

                    {/* Save Status */}
                    {saveStatus && (
                      <Alert
                        severity={saveStatus === 'saved' ? 'success' : saveStatus === 'saving' ? 'info' : 'error'}
                        sx={{ mt: 2 }}
                      >
                        {saveStatus === 'saved' ? 'Feedback auto-saved' : saveStatus === 'saving' ? 'Saving...' : 'Error saving'}
                      </Alert>
                    )}
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default VideoEvaluatorV3;
