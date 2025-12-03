import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  LinearProgress, 
  CircularProgress,
  Alert,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import axios from 'axios';
import config from '../config';

const AnalysisStatus = ({ videoId, jobId, onAnalysisComplete }) => {
  const [status, setStatus] = useState('queued');
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('Initializing analysis...');
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [logs, setLogs] = useState([]);
  const [eta, setEta] = useState(null);

  // Poll for status updates
  useEffect(() => {
    if (!jobId) return;
    
    const pollInterval = setInterval(async () => {
      try {
        const response = await axios.get(`${config.api.baseUrl}${config.api.endpoints.status(jobId)}`);
        const data = response.data;
        
        setStatus(data.status);
        setProgress(data.progress * 100);
        setMessage(data.message);
        
        // Update logs if available
        if (data.logs && Array.isArray(data.logs)) {
          setLogs(prevLogs => [...prevLogs, ...data.logs]);
        } else if (data.log && typeof data.log === 'string') {
          setLogs(prevLogs => [...prevLogs, data.log]);
        }
        
        // Update ETA if available
        if (data.eta) {
          setEta(data.eta);
        }
        
        // Check if analysis is complete
        if (data.status === 'completed') {
          clearInterval(pollInterval);
          fetchResults();
        } else if (data.status === 'failed') {
          clearInterval(pollInterval);
          setError(data.message || 'Analysis failed');
        }
      } catch (err) {
        console.error('Error polling status:', err);
        setError(`Error checking status: ${err.message}`);
      }
    }, 2000); // Poll every 2 seconds
    
    return () => clearInterval(pollInterval);
  }, [jobId]);

  // Fetch results when analysis is complete
  const fetchResults = async () => {
    try {
      const response = await axios.get(`${config.api.baseUrl}${config.api.endpoints.results(jobId)}`);
      setResults(response.data);
      onAnalysisComplete(response.data);
    } catch (err) {
      console.error('Error fetching results:', err);
      setError(`Error fetching results: ${err.message}`);
    }
  };

  // Render status message based on current state
  const renderStatusMessage = () => {
    switch (status) {
      case 'queued':
        return 'Your analysis is queued and will start soon...';
      case 'processing':
        return `Processing: ${message}`;
      case 'completed':
        return 'Analysis complete! Loading results...';
      case 'failed':
        return `Analysis failed: ${message}`;
      default:
        return 'Waiting for status update...';
    }
  };

  // Render progress indicator
  const renderProgress = () => {
    if (status === 'queued') {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      );
    }
    
    return (
      <Box sx={{ my: 4 }}>
        <LinearProgress 
          variant="determinate" 
          value={progress} 
          sx={{ height: 10, borderRadius: 5 }}
        />
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
          <Typography variant="body2" color="text.secondary">
            {Math.round(progress)}%
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {status}
          </Typography>
        </Box>
      </Box>
    );
  };

  // Render processing steps
  const renderProcessingSteps = () => {
    const steps = [
      { label: 'Extracting video frames', complete: progress >= 20 },
      { label: 'Transcribing audio with Whisper', complete: progress >= 40 },
      { label: 'Running Qwen2.5-VL model inference with BJJ-specific prompts', complete: progress >= 60 },
      { label: 'Processing results and detecting positions/transitions', complete: progress >= 80 },
      { label: 'Generating interactive timeline with position markers', complete: progress >= 95 }
    ];
    
    return (
      <Box sx={{ mt: 4 }}>
        <Typography variant="subtitle2" gutterBottom>
          Processing Steps:
        </Typography>
        
        {steps.map((step, index) => (
          <Box 
            key={index}
            sx={{ 
              display: 'flex', 
              alignItems: 'center',
              mb: 1,
              opacity: step.complete ? 1 : 0.5
            }}
          >
            <Box 
              sx={{ 
                width: 20, 
                height: 20, 
                borderRadius: '50%',
                backgroundColor: step.complete ? '#4caf50' : '#757575',
                mr: 2,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                color: 'white',
                fontSize: '12px'
              }}
            >
              {step.complete ? '✓' : index + 1}
            </Box>
            <Typography variant="body2">
              {step.label}
            </Typography>
          </Box>
        ))}
      </Box>
    );
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        BJJ Video Analysis
      </Typography>
      
      <Typography variant="body1" gutterBottom>
        {renderStatusMessage()}
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ my: 2 }}>
          {error}
        </Alert>
      )}
      
      {renderProgress()}
      
      {status === 'processing' && renderProcessingSteps()}
      
      {/* Output Console for Debugging and Tracing */}
      <Accordion sx={{ mt: 3 }}>
        <AccordionSummary
          expandIcon={<ExpandMoreIcon />}
          aria-controls="output-console-content"
          id="output-console-header"
        >
          <Typography>Output Console {logs.length > 0 && `(${logs.length} entries)`}</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Paper 
            elevation={0} 
            variant="outlined" 
            sx={{ 
              p: 2, 
              backgroundColor: '#1e1e1e', 
              color: '#f1f1f1',
              maxHeight: '300px',
              overflow: 'auto',
              fontFamily: 'monospace',
              fontSize: '0.85rem'
            }}
          >
            {logs.length > 0 ? (
              logs.map((log, index) => (
                <Box key={index} sx={{ mb: 0.5 }}>
                  <Typography variant="body2" component="div" sx={{ whiteSpace: 'pre-wrap' }}>
                    {log}
                  </Typography>
                </Box>
              ))
            ) : (
              <Typography variant="body2">No logs available yet...</Typography>
            )}
          </Paper>
        </AccordionDetails>
      </Accordion>
      
      <Box sx={{ mt: 4 }}>
        {eta && (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Estimated time remaining: {typeof eta === 'number' ? `${Math.ceil(eta)} seconds` : eta}
          </Typography>
        )}
        <Typography variant="body2" color="text.secondary">
          Job ID: {jobId}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Video ID: {videoId}
        </Typography>
      </Box>
    </Box>
  );
};

export default AnalysisStatus;
