import React, { useState } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  Paper, 
  Tooltip, 
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';
import axios from 'axios';
import config from '../config';

const Timeline = ({ 
  analysisData, 
  currentTime, 
  onEventClick, 
  videoId, 
  onAnalysisStart,
  isAnalyzing 
}) => {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [analysisConfig, setAnalysisConfig] = useState({
    every_n_frames: config.analysis.defaultFrameSamplingRate,
    batch_size: config.analysis.defaultBatchSize,
    user_prompt: config.analysis.defaultPrompt
  });

  // Calculate timeline width based on video duration
  const videoDuration = analysisData?.metadata?.video_duration || 0;
  
  // Start analysis dialog
  const handleOpenDialog = () => {
    setDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
  };

  const handleConfigChange = (event) => {
    const { name, value } = event.target;
    setAnalysisConfig({
      ...analysisConfig,
      [name]: name === 'every_n_frames' || name === 'batch_size' ? parseInt(value) : value
    });
  };

  const handleStartAnalysis = async () => {
    if (!videoId) return;
    
    try {
      const response = await axios.post(`${config.api.baseUrl}${config.api.endpoints.analyze}`, {
        video_id: videoId,
        config: analysisConfig
      });
      
      if (response.data && response.data.job_id) {
        onAnalysisStart(response.data.job_id);
        handleCloseDialog();
      }
    } catch (error) {
      console.error('Analysis error:', error);
    }
  };

  // Render timeline events
  const renderTimelineEvents = () => {
    if (!analysisData || !analysisData.timeline || analysisData.timeline.length === 0) {
      return null;
    }

    return analysisData.timeline.map((event, index) => {
      // Calculate position on timeline
      const startPercent = (event.estimated_time_start / videoDuration) * 100;
      const endPercent = (event.estimated_time_end / videoDuration) * 100;
      const width = endPercent - startPercent;
      
      // Determine color based on positions/transitions
      let color = config.ui.timeline.positionColors.default;
      if (event.positions && event.positions.length > 0) {
        // Different colors for different positions
        const position = event.positions[0].toLowerCase();
        if (position.includes('mount')) color = config.ui.timeline.positionColors.mount;
        else if (position.includes('guard')) color = config.ui.timeline.positionColors.guard;
        else if (position.includes('back')) color = config.ui.timeline.positionColors.back;
        else if (position.includes('side')) color = config.ui.timeline.positionColors.side;
        else if (position.includes('half')) color = config.ui.timeline.positionColors.halfguard;
        else if (position.includes('turtle')) color = config.ui.timeline.positionColors.turtle;
      }
      
      // Create tooltip content
      const tooltipContent = (
        <>
          <Typography variant="subtitle2">
            {formatTime(event.estimated_time_start)} - {formatTime(event.estimated_time_end)}
          </Typography>
          {event.positions && event.positions.length > 0 && (
            <Typography variant="body2">
              Positions: {event.positions.join(', ')}
            </Typography>
          )}
          {event.transitions && event.transitions.length > 0 && (
            <Typography variant="body2">
              Transitions: {event.transitions.join(', ')}
            </Typography>
          )}
        </>
      );
      
      return (
        <Tooltip 
          key={index} 
          title={tooltipContent} 
          arrow
          placement="top"
        >
          <Box
            sx={{
              position: 'absolute',
              left: `${startPercent}%`,
              width: `${Math.max(width, 1)}%`,
              height: '100%',
              backgroundColor: color,
              cursor: 'pointer',
              '&:hover': {
                opacity: 0.8,
                height: '120%',
                top: '-10%',
                zIndex: 2
              }
            }}
            onClick={() => onEventClick(event)}
          />
        </Tooltip>
      );
    });
  };

  // Render current time marker
  const renderCurrentTimeMarker = () => {
    if (!videoDuration) return null;
    
    const position = (currentTime / videoDuration) * 100;
    
    return (
      <Box
        sx={{
          position: 'absolute',
          left: `${position}%`,
          width: '2px',
          height: '120%',
          top: '-10%',
          backgroundColor: '#f50057',
          zIndex: 3
        }}
      />
    );
  };

  // Format time as MM:SS
  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Timeline</Typography>
        
        {videoId && !isAnalyzing && (
          <Button 
            variant="contained" 
            color="primary"
            startIcon={<PlayCircleOutlineIcon />}
            onClick={handleOpenDialog}
            disabled={isAnalyzing}
          >
            Analyze Video
          </Button>
        )}
      </Box>
      
      {/* Timeline track */}
      <Box 
        sx={{ 
          position: 'relative', 
          height: '40px', 
          backgroundColor: '#333',
          borderRadius: '4px',
          overflow: 'hidden'
        }}
      >
        {/* Time markers */}
        {videoDuration > 0 && Array.from({ length: 11 }).map((_, i) => {
          const percent = i * 10;
          const timeValue = (videoDuration * percent) / 100;
          
          return (
            <Box
              key={i}
              sx={{
                position: 'absolute',
                left: `${percent}%`,
                top: 0,
                height: '10px',
                width: '1px',
                backgroundColor: '#666'
              }}
            >
              <Typography 
                variant="caption" 
                sx={{ 
                  position: 'absolute', 
                  top: '10px', 
                  left: '-10px',
                  color: '#999'
                }}
              >
                {formatTime(timeValue)}
              </Typography>
            </Box>
          );
        })}
        
        {/* Events */}
        {renderTimelineEvents()}
        
        {/* Current time marker */}
        {renderCurrentTimeMarker()}
      </Box>
      
      {/* Analysis Configuration Dialog */}
      <Dialog open={dialogOpen} onClose={handleCloseDialog}>
        <DialogTitle>Analysis Configuration</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <TextField
              fullWidth
              label="Frame Sampling Rate"
              name="every_n_frames"
              type="number"
              value={analysisConfig.every_n_frames}
              onChange={handleConfigChange}
              helperText="Process every Nth frame (higher = faster but less detailed)"
              margin="normal"
            />
            
            <TextField
              fullWidth
              label="Batch Size"
              name="batch_size"
              type="number"
              value={analysisConfig.batch_size}
              onChange={handleConfigChange}
              helperText="Number of frames to process in each batch"
              margin="normal"
            />
            
            <TextField
              fullWidth
              label="Analysis Prompt"
              name="user_prompt"
              multiline
              rows={3}
              value={analysisConfig.user_prompt}
              onChange={handleConfigChange}
              helperText="Instructions for the model"
              margin="normal"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button onClick={handleStartAnalysis} variant="contained" color="primary">
            Start Analysis
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Timeline;
