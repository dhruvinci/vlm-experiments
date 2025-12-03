import React from 'react';
import { Box, Typography, LinearProgress, Chip } from '@mui/material';

const ActionScoreIndicator = ({ avgAction, transition }) => {
  if (avgAction === undefined && !transition) {
    return null;
  }

  // Color based on action intensity
  const getColor = (score) => {
    if (score >= 0.7) return 'error';
    if (score >= 0.4) return 'warning';
    return 'success';
  };

  const getIntensityLabel = (score) => {
    if (score >= 0.7) return 'High Intensity';
    if (score >= 0.4) return 'Medium Intensity';
    return 'Low Intensity';
  };

  return (
    <Box sx={{ mt: 1.5 }}>
      {avgAction !== undefined && (
        <Box sx={{ mb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
            <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              🔥 Action Score
            </Typography>
            <Typography variant="caption" fontWeight={600}>
              {(avgAction * 100).toFixed(0)}% - {getIntensityLabel(avgAction)}
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={avgAction * 100}
            color={getColor(avgAction)}
            sx={{ height: 8, borderRadius: 1 }}
          />
        </Box>
      )}
      {transition && (
        <Chip
          label={`Transition: ${transition.replace(/_/g, ' ')}`}
          size="small"
          color="secondary"
          variant="outlined"
          sx={{ mt: 0.5 }}
        />
      )}
    </Box>
  );
};

export default ActionScoreIndicator;
