import React from 'react';
import { Box, Chip, Typography, LinearProgress, Tooltip, Paper } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import RadioButtonUncheckedIcon from '@mui/icons-material/RadioButtonUnchecked';

const PositionMarker = ({ position, isActive, onClick, isLabeled }) => {
  const getPositionColor = (positionName) => {
    const colorMap = {
      'standing': 'default',
      'guard': 'primary',
      'mount': 'error',
      'side_control': 'warning',
      'back_control': 'error',
      'half_guard': 'info',
      'turtle': 'secondary',
      'north_south': 'warning',
      'knee_on_belly': 'warning',
    };

    return colorMap[positionName] || 'default';
  };

  const confidence = position.confidence || 0;
  const confidenceColor = confidence >= 0.8 ? 'success' : confidence >= 0.5 ? 'warning' : 'error';

  return (
    <Tooltip
      title={
        <div>
          <div><strong>Time:</strong> {position.start_time} - {position.end_time}</div>
          <div><strong>Position:</strong> {position.position}</div>
          {position.sub_position && <div><strong>Sub:</strong> {position.sub_position}</div>}
          <div><strong>Top:</strong> {position.top_athlete || 'N/A'}</div>
          <div><strong>Bottom:</strong> {position.bottom_athlete || 'N/A'}</div>
          <div><strong>Confidence:</strong> {(confidence * 100).toFixed(0)}%</div>
          {position.avg_action !== undefined && (
            <div><strong>Action Score:</strong> {(position.avg_action * 100).toFixed(0)}%</div>
          )}
          {position.notes && <div><em>{position.notes}</em></div>}
          {position.narrative && <div style={{ marginTop: 4 }}><em>{position.narrative}</em></div>}
          <div style={{ marginTop: 8 }}>
            {isLabeled ? '✓ Labeled' : 'Click to review'}
          </div>
        </div>
      }
      arrow
    >
      <Paper
        sx={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 1,
          p: 1,
          m: 0.5,
          cursor: 'pointer',
          bgcolor: isActive ? 'primary.main' : 'background.paper',
          border: isActive ? '2px solid' : '1px solid',
          borderColor: isActive ? 'primary.dark' : 'divider',
          transition: 'all 0.2s',
          color: isActive ? 'white' : 'inherit',
          '&:hover': {
            bgcolor: isActive ? 'primary.dark' : 'grey.100',
            transform: 'translateY(-2px)',
            boxShadow: 2,
          }
        }}
        onClick={onClick}
      >
        {/* Status Icon */}
        {isLabeled ? (
          <CheckCircleIcon sx={{ color: isActive ? 'success.light' : 'success.main' }} fontSize="small" />
        ) : (
          <RadioButtonUncheckedIcon sx={{ color: isActive ? 'grey.300' : 'disabled' }} fontSize="small" />
        )}

        {/* Time Range */}
        <Chip
          label={`${position.start_time}-${position.end_time}`}
          size="small"
          sx={{
            fontSize: '0.7rem',
            height: 20,
            bgcolor: isActive ? 'rgba(255,255,255,0.2)' : 'default',
            color: isActive ? 'white' : 'inherit',
            borderColor: isActive ? 'rgba(255,255,255,0.3)' : 'default'
          }}
        />

        {/* Position */}
        <Chip
          label={position.position}
          color={isActive ? 'default' : getPositionColor(position.position)}
          size="small"
          sx={{
            bgcolor: isActive ? 'rgba(255,255,255,0.25)' : undefined,
            color: isActive ? 'white' : undefined,
            fontWeight: isActive ? 600 : 400
          }}
        />

        {/* Sub-position if available */}
        {position.sub_position && (
          <Chip
            label={position.sub_position}
            variant="outlined"
            size="small"
            sx={{
              fontSize: '0.7rem',
              borderColor: isActive ? 'rgba(255,255,255,0.5)' : undefined,
              color: isActive ? 'white' : undefined
            }}
          />
        )}

        {/* Confidence indicator */}
        <Box
          sx={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            bgcolor: isActive ? 'rgba(255,255,255,0.7)' : `${confidenceColor}.main`,
          }}
        />

        {/* Action Score indicator (Exp3 & Exp4) */}
        {(position.avg_action !== undefined || position.action_score !== undefined) && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Typography variant="body2">🔥</Typography>
            <Typography variant="caption" sx={{ fontWeight: 600 }}>
              {((position.avg_action || position.action_score || 0) * 100).toFixed(0)}%
            </Typography>
          </Box>
        )}
      </Paper>
    </Tooltip>
  );
};

export default PositionMarker;
