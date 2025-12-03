import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { Container, Box, Paper, Typography, AppBar, Toolbar, Tabs, Tab, IconButton } from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import theme from './theme';
import LandingPage from './components/LandingPage';
import VideoUploader from './components/VideoUploader';
import VideoPlayer from './components/VideoPlayer';
import Timeline from './components/Timeline';
import AnalysisPanel from './components/AnalysisPanel';
import AnalysisStatus from './components/AnalysisStatus';
import VideoEvaluator from './components/VideoEvaluator';
import VideoEvaluatorV2 from './components/VideoEvaluatorV2';
import VideoEvaluatorV3 from './components/VideoEvaluatorV3';
import './App.css';

function HITLTool() {
  const navigate = useNavigate();
  // Tab navigation state (0 = HITL Evaluation, 1 = Video Analyzer)
  const [currentTab, setCurrentTab] = useState(0);

  // Main analyzer state
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [videoId, setVideoId] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Handle video upload success
  const handleUploadSuccess = (uploadedVideoId, uploadedVideoUrl) => {
    setVideoId(uploadedVideoId);
    setVideoUrl(uploadedVideoUrl);
    setIsAnalyzing(false);
    setAnalysisData(null);
    setJobId(null);
  };

  // Handle analysis start
  const handleAnalysisStart = (newJobId) => {
    setJobId(newJobId);
    setIsAnalyzing(true);
  };

  // Handle analysis complete
  const handleAnalysisComplete = (data) => {
    setAnalysisData(data);
    setIsAnalyzing(false);
  };

  // Handle timeline event click
  const handleEventClick = (event) => {
    setSelectedEvent(event);
    if (event.estimated_time_start) {
      setCurrentTime(event.estimated_time_start);
    }
  };

  // Handle video time update
  const handleTimeUpdate = (time) => {
    setCurrentTime(time);
    
    // Find event at current time if no event is selected
    if (!selectedEvent && analysisData && analysisData.timeline) {
      const eventAtTime = analysisData.timeline.find(event => 
        event.estimated_time_start <= time && event.estimated_time_end >= time
      );
      
      if (eventAtTime) {
        setSelectedEvent(eventAtTime);
      }
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      {/* Header */}
      <AppBar position="static" sx={{ bgcolor: 'primary.main', mb: 3, boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
        <Toolbar>
          <IconButton
            onClick={() => navigate('/')}
            sx={{ color: 'white', mr: 1 }}
            title="Back to Home"
          >
            <HomeIcon />
          </IconButton>
          <Box
            component="img"
            src="/logo-brown.png"
            alt="Sensai"
            sx={{ height: 35, mr: 2 }}
          />
          <Typography variant="h6" sx={{ flexGrow: 1, color: 'white' }}>
            HITL Tool
          </Typography>
          <Tabs
            value={currentTab}
            onChange={(e, newValue) => setCurrentTab(newValue)}
            sx={{ ml: 'auto' }}
            textColor="inherit"
            indicatorColor="secondary"
          >
            <Tab label="HITL Evaluation" />
            <Tab label="Video Analyzer" />
          </Tabs>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        {currentTab === 0 ? (
          // HITL Evaluation Tab - V3 with segment-based feedback
          <VideoEvaluatorV3 />
        ) : (
          // Video Analyzer Tab
          !videoUrl ? (
            <VideoUploader onUploadSuccess={handleUploadSuccess} />
          ) : (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', gap: 2, height: '60vh' }}>
                {/* Left side - Video Player */}
                <Box sx={{ flex: 3 }}>
                  <Paper sx={{ p: 2, height: '100%' }}>
                    <VideoPlayer
                      videoUrl={videoUrl}
                      currentTime={currentTime}
                      onTimeUpdate={handleTimeUpdate}
                    />
                  </Paper>
                </Box>

                {/* Right side - Analysis Panel */}
                <Box sx={{ flex: 2 }}>
                  <Paper sx={{ p: 2, height: '100%', overflow: 'auto' }}>
                    {isAnalyzing ? (
                      <AnalysisStatus
                        videoId={videoId}
                        jobId={jobId}
                        onAnalysisComplete={handleAnalysisComplete}
                      />
                    ) : (
                      analysisData ? (
                        <AnalysisPanel
                          analysisData={analysisData}
                          selectedEvent={selectedEvent}
                        />
                      ) : (
                        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                          <Typography variant="h6" color="text.secondary">
                            Click "Analyze Video" to start analysis
                          </Typography>
                        </Box>
                      )
                    )}
                  </Paper>
                </Box>
              </Box>

              {/* Timeline */}
              <Paper sx={{ p: 2 }}>
                <Timeline
                  analysisData={analysisData}
                  currentTime={currentTime}
                  onEventClick={handleEventClick}
                  videoId={videoId}
                  onAnalysisStart={handleAnalysisStart}
                  isAnalyzing={isAnalyzing}
                />
              </Paper>
            </Box>
          )
        )}
      </Container>
    </Box>
  );
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/hitl" element={<HITLTool />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
