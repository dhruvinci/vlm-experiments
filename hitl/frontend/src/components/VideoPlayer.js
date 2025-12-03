import React, { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react';
import { Box, IconButton, Slider, Typography, Grid } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import SkipNextIcon from '@mui/icons-material/SkipNext';
import SkipPreviousIcon from '@mui/icons-material/SkipPrevious';
import ReactPlayer from 'react-player';

const VideoPlayer = forwardRef(({
  videoUrl,
  currentTime,
  onTimeUpdate,
  playbackRate: externalPlaybackRate = 1,
  loop = false,
  segmentStart = null,
  segmentEnd = null,
  pauseAtSegmentEnd = false
}, ref) => {
  const playerRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [playedSeconds, setPlayedSeconds] = useState(0);
  const [seeking, setSeeking] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(externalPlaybackRate);

  // Update playback rate when external prop changes
  useEffect(() => {
    setPlaybackRate(externalPlaybackRate);
  }, [externalPlaybackRate]);

  // Expose play/pause methods to parent via ref
  useImperativeHandle(ref, () => ({
    play: () => setPlaying(true),
    pause: () => setPlaying(false),
    togglePlayPause: () => setPlaying(prev => !prev),
    seekTo: (time) => {
      if (playerRef.current) {
        playerRef.current.seekTo(time);
        setPlayedSeconds(time);
        onTimeUpdate(time);
      }
    }
  }));

  // Handle external time update (e.g., from timeline)
  useEffect(() => {
    if (!seeking && playerRef.current && Math.abs(playedSeconds - currentTime) > 0.5) {
      playerRef.current.seekTo(currentTime);
    }
  }, [currentTime, seeking, playedSeconds]);

  const handlePlayPause = () => {
    setPlaying(!playing);
  };

  const handleDuration = (duration) => {
    setDuration(duration);
  };

  const handleProgress = (state) => {
    if (!seeking) {
      setPlayedSeconds(state.playedSeconds);
      onTimeUpdate(state.playedSeconds);

      // Handle segment boundary
      if (segmentEnd !== null && state.playedSeconds >= segmentEnd - 0.1) {
        if (loop) {
          // Loop back to current segment start
          if (playerRef.current) {
            playerRef.current.seekTo(currentTime);
          }
        } else if (pauseAtSegmentEnd) {
          // Just pause at segment end
          setPlaying(false);
        }
      }
    }
  };

  const handleSeekChange = (event, newValue) => {
    setSeeking(true);
    setPlayedSeconds(newValue * duration / 100);
  };

  const handleSeekMouseUp = (event, newValue) => {
    setSeeking(false);
    playerRef.current.seekTo(newValue / 100);
    onTimeUpdate(newValue * duration / 100);
  };

  const handleSkipForward = () => {
    let newTime = Math.min(playedSeconds + 5, duration);

    // If segment boundaries are set, constrain to segment end
    if (segmentEnd !== null) {
      newTime = Math.min(newTime, segmentEnd);
    }

    playerRef.current.seekTo(newTime);
    setPlayedSeconds(newTime);
    onTimeUpdate(newTime);
  };

  const handleSkipBackward = () => {
    let newTime = Math.max(playedSeconds - 5, 0);

    // If segment boundaries are set, constrain to segment start
    if (segmentStart !== null) {
      newTime = Math.max(newTime, segmentStart);
    }

    playerRef.current.seekTo(newTime);
    setPlayedSeconds(newTime);
    onTimeUpdate(newTime);
  };

  // Format time as MM:SS
  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
  };

  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Video Container */}
      <Box sx={{ flex: 1, position: 'relative', backgroundColor: '#000', mb: 2 }}>
        <ReactPlayer
          ref={playerRef}
          url={videoUrl}
          width="100%"
          height="100%"
          playing={playing}
          playbackRate={playbackRate}
          onDuration={handleDuration}
          onProgress={handleProgress}
          progressInterval={100}
        />
        
        {/* Video Overlay - Can be used for position markers, etc. */}
        <Box 
          sx={{ 
            position: 'absolute', 
            top: 0, 
            left: 0, 
            right: 0, 
            bottom: 0, 
            pointerEvents: 'none',
            zIndex: 1
          }}
        >
          {/* Overlay content will be added here */}
        </Box>
      </Box>
      
      {/* Controls */}
      <Box>
        {/* Time Slider */}
        <Slider
          min={0}
          max={100}
          value={duration ? (playedSeconds / duration) * 100 : 0}
          onChange={handleSeekChange}
          onChangeCommitted={handleSeekMouseUp}
          sx={{ mb: 1 }}
        />
        
        <Grid container alignItems="center" spacing={1}>
          {/* Time Display */}
          <Grid item xs={3}>
            <Typography variant="body2">
              {formatTime(playedSeconds)} / {formatTime(duration)}
            </Typography>
          </Grid>

          {/* Playback Controls - Centered */}
          <Grid item xs={6} sx={{ display: 'flex', justifyContent: 'center' }}>
            <IconButton onClick={handleSkipBackward}>
              <SkipPreviousIcon />
            </IconButton>

            <IconButton onClick={handlePlayPause} size="large">
              {playing ? <PauseIcon fontSize="large" /> : <PlayArrowIcon fontSize="large" />}
            </IconButton>

            <IconButton onClick={handleSkipForward}>
              <SkipNextIcon />
            </IconButton>
          </Grid>
          
          {/* Empty spacer for symmetry */}
          <Grid item xs={3} />
        </Grid>
      </Box>
    </Box>
  );
});

export default VideoPlayer;
