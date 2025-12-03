import React from 'react';
import { Box, Typography, List, ListItem, ListItemButton, ListItemText, Paper, Chip } from '@mui/material';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';

const KeyMomentsPanel = ({ keyMoments, onMomentClick }) => {
  if (!keyMoments || keyMoments.length === 0) {
    return null;
  }

  // Parse timestamp from moment string (e.g., "0:00: Match starts")
  const parseTimestamp = (momentStr) => {
    const match = momentStr.match(/^(\d+:\d+):/);
    return match ? match[1] : null;
  };

  const timeToSeconds = (timeStr) => {
    const parts = timeStr.split(':');
    if (parts.length === 2) {
      return parseInt(parts[0]) * 60 + parseInt(parts[1]);
    }
    return 0;
  };

  return (
    <Paper sx={{ p: 2, mt: 2, bgcolor: 'background.default' }}>
      <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <AccessTimeIcon fontSize="small" /> Key Moments ({keyMoments.length})
      </Typography>
      <List dense sx={{ maxHeight: 200, overflow: 'auto' }}>
        {keyMoments.map((moment, idx) => {
          const timestamp = parseTimestamp(moment);
          const description = moment.replace(/^\d+:\d+:\s*/, '');
          
          return (
            <ListItem key={idx} disablePadding>
              <ListItemButton
                onClick={() => {
                  if (timestamp && onMomentClick) {
                    onMomentClick(timeToSeconds(timestamp));
                  }
                }}
                sx={{
                  '&:hover': {
                    bgcolor: 'action.hover'
                  }
                }}
              >
                <PlayCircleOutlineIcon fontSize="small" sx={{ mr: 1, color: 'primary.main' }} />
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {timestamp && (
                        <Chip
                          label={timestamp}
                          size="small"
                          sx={{ minWidth: 50, fontFamily: 'monospace' }}
                        />
                      )}
                      <Typography variant="body2">{description}</Typography>
                    </Box>
                  }
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>
    </Paper>
  );
};

export default KeyMomentsPanel;
