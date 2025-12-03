import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Tabs, 
  Tab, 
  List, 
  ListItem, 
  ListItemText,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Paper
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import JSONPretty from 'react-json-pretty';
import 'react-json-pretty/themes/monikai.css';

const AnalysisPanel = ({ analysisData, selectedEvent }) => {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Find current analysis based on selected event
  const getCurrentAnalysis = () => {
    if (!analysisData || !analysisData.analysis || !selectedEvent) {
      return null;
    }

    return analysisData.analysis.find(
      batch => batch.batch_index === selectedEvent.batch_index
    );
  };

  const currentAnalysis = getCurrentAnalysis();

  // Render position information
  const renderPositions = () => {
    if (!selectedEvent || !selectedEvent.positions || selectedEvent.positions.length === 0) {
      return (
        <Typography variant="body2" color="text.secondary">
          No position data available for this segment
        </Typography>
      );
    }

    return (
      <List dense>
        {selectedEvent.positions.map((position, index) => (
          <ListItem key={index}>
            <ListItemText 
              primary={position} 
              secondary={`Confidence: ${currentAnalysis?.confidence || 'N/A'}`} 
            />
          </ListItem>
        ))}
      </List>
    );
  };

  // Render transitions information
  const renderTransitions = () => {
    if (!selectedEvent || !selectedEvent.transitions || selectedEvent.transitions.length === 0) {
      return (
        <Typography variant="body2" color="text.secondary">
          No transition data available for this segment
        </Typography>
      );
    }

    return (
      <List dense>
        {selectedEvent.transitions.map((transition, index) => (
          <ListItem key={index}>
            <ListItemText 
              primary={transition} 
              secondary={`At ${formatTime(selectedEvent.estimated_time_start)}`} 
            />
          </ListItem>
        ))}
      </List>
    );
  };

  // Render transcription if available
  const renderTranscription = () => {
    if (!currentAnalysis || 
        !currentAnalysis.transcription || 
        !currentAnalysis.transcription.segments || 
        currentAnalysis.transcription.segments.length === 0) {
      return (
        <Typography variant="body2" color="text.secondary">
          No transcription available for this segment
        </Typography>
      );
    }

    return (
      <Box>
        <List dense>
          {currentAnalysis.transcription.segments.map((segment, index) => (
            <ListItem key={index}>
              <ListItemText 
                primary={segment.text} 
                secondary={
                  <>
                    <Typography variant="caption" component="span">
                      {`${formatTime(segment.start)} - ${formatTime(segment.end)}`}
                    </Typography>
                    {segment.bjj_terms && segment.bjj_terms.length > 0 && (
                      <Box sx={{ mt: 0.5, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {segment.bjj_terms.map((term, i) => (
                          <Chip 
                            key={i} 
                            label={term} 
                            size="small" 
                            color="secondary" 
                            variant="outlined"
                          />
                        ))}
                      </Box>
                    )}
                  </>
                }
              />
            </ListItem>
          ))}
        </List>
      </Box>
    );
  };

  // Render raw JSON data
  const renderRawData = () => {
    if (!currentAnalysis) {
      return (
        <Typography variant="body2" color="text.secondary">
          No analysis data available
        </Typography>
      );
    }

    return (
      <Box sx={{ maxHeight: '400px', overflow: 'auto' }}>
        <JSONPretty id="json-pretty" data={currentAnalysis}></JSONPretty>
      </Box>
    );
  };

  // Format time as MM:SS
  const formatTime = (seconds) => {
    if (seconds === undefined) return 'N/A';
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
  };

  // Render metadata summary
  const renderMetadataSummary = () => {
    if (!analysisData || !analysisData.metadata) {
      return null;
    }

    // Get video info if available
    const videoInfo = analysisData.metadata.video_info || {};
    const analysisConfig = analysisData.metadata.analysis_config || {};

    return (
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          Analysis Summary
        </Typography>
        
        {/* Processing stats */}
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
          <Chip 
            label={`${analysisData.metadata.frames_processed || 0} frames`} 
            size="small" 
            color="primary" 
          />
          <Chip 
            label={`${analysisData.metadata.total_batches || 0} batches`} 
            size="small" 
            color="primary" 
          />
          <Chip 
            label={`${Math.round(analysisData.metadata.processing_time || 0)}s processing time`} 
            size="small" 
            color="primary" 
          />
          {analysisConfig.transcription_available && (
            <Chip 
              label="Audio transcription" 
              size="small" 
              color="success" 
            />
          )}
        </Box>
        
        {/* Video metadata if available */}
        {Object.keys(videoInfo).length > 0 && (
          <Accordion sx={{ mb: 1 }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>Video Details</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                {videoInfo.duration && (
                  <Typography variant="body2">
                    Duration: {formatTime(videoInfo.duration)}
                  </Typography>
                )}
                {videoInfo.fps && (
                  <Typography variant="body2">
                    FPS: {videoInfo.fps.toFixed(2)}
                  </Typography>
                )}
                {videoInfo.width && videoInfo.height && (
                  <Typography variant="body2">
                    Resolution: {videoInfo.width}×{videoInfo.height}
                  </Typography>
                )}
                {analysisConfig.every_n_frames && (
                  <Typography variant="body2">
                    Frame sampling: Every {analysisConfig.every_n_frames} frames
                  </Typography>
                )}
              </Box>
            </AccordionDetails>
          </Accordion>
        )}
      </Paper>
    );
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h6" gutterBottom>
        Analysis Results
      </Typography>
      
      {renderMetadataSummary()}
      
      {selectedEvent ? (
        <>
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle1">
              Segment: {formatTime(selectedEvent.estimated_time_start)} - {formatTime(selectedEvent.estimated_time_end)}
            </Typography>
          </Box>
          
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
            <Tabs value={tabValue} onChange={handleTabChange} aria-label="analysis tabs">
              <Tab label="Positions" />
              <Tab label="Transitions" />
              <Tab label="Transcription" />
              <Tab label="Raw Data" />
            </Tabs>
          </Box>
          
          <Box sx={{ flex: 1, overflow: 'auto' }}>
            {tabValue === 0 && renderPositions()}
            {tabValue === 1 && renderTransitions()}
            {tabValue === 2 && renderTranscription()}
            {tabValue === 3 && renderRawData()}
          </Box>
        </>
      ) : (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flex: 1 }}>
          <Typography variant="body1" color="text.secondary">
            Select a segment on the timeline to view analysis
          </Typography>
        </Box>
      )}
      
      {/* Additional details accordion */}
      {selectedEvent && (
        <Accordion sx={{ mt: 'auto' }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography>Additional Details</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="body2" gutterBottom>
              Batch Index: {selectedEvent.batch_index}
            </Typography>
            {currentAnalysis && currentAnalysis.is_structured !== false && (
              <Typography variant="body2" color="text.secondary">
                Structured analysis available. View Raw Data tab for complete details.
              </Typography>
            )}
          </AccordionDetails>
        </Accordion>
      )}
    </Box>
  );
};

export default AnalysisPanel;
