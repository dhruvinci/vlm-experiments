import React, { useState } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  Paper, 
  LinearProgress,
  Alert
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import axios from 'axios';
import config from '../config';

const VideoUploader = ({ onUploadSuccess }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Check if file is a video
      if (!file.type.startsWith('video/')) {
        setError('Please select a video file');
        setSelectedFile(null);
        return;
      }
      
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${config.api.baseUrl}${config.api.endpoints.upload}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        }
      });

      if (response.data && response.data.video_id) {
        // Create a local URL for the uploaded file
        const videoUrl = URL.createObjectURL(selectedFile);
        onUploadSuccess(response.data.video_id, videoUrl);
      } else {
        setError('Upload failed: Invalid server response');
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError(`Upload failed: ${err.response?.data?.detail || err.message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <Paper 
      sx={{ 
        p: 4, 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '60vh',
        textAlign: 'center'
      }}
    >
      <Typography variant="h4" gutterBottom>
        BJJ Video Analyzer
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Upload a BJJ video for detailed position tracking and analysis using Qwen2.5-VL model
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2, width: '100%', maxWidth: 500 }}>
          {error}
        </Alert>
      )}
      
      <Box 
        sx={{ 
          border: '2px dashed #3f51b5', 
          borderRadius: 2,
          p: 5,
          mb: 3,
          width: '100%',
          maxWidth: 500,
          cursor: 'pointer',
          '&:hover': {
            backgroundColor: 'rgba(63, 81, 181, 0.04)'
          }
        }}
        onClick={() => document.getElementById('video-upload').click()}
      >
        <input
          id="video-upload"
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <CloudUploadIcon sx={{ fontSize: 60, color: '#3f51b5', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          Drag & Drop or Click to Upload
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Supported formats: MP4, MOV, AVI, etc.
        </Typography>
        
        {selectedFile && (
          <Typography variant="body2" sx={{ mt: 2 }}>
            Selected: {selectedFile.name} ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
          </Typography>
        )}
      </Box>
      
      {uploading && (
        <Box sx={{ width: '100%', maxWidth: 500, mb: 2 }}>
          <LinearProgress variant="determinate" value={uploadProgress} />
          <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 1 }}>
            Uploading: {uploadProgress}%
          </Typography>
        </Box>
      )}
      
      <Button 
        variant="contained" 
        color="primary" 
        size="large"
        disabled={!selectedFile || uploading}
        onClick={handleUpload}
        startIcon={<CloudUploadIcon />}
      >
        {uploading ? 'Uploading...' : 'Upload Video'}
      </Button>
    </Paper>
  );
};

export default VideoUploader;
