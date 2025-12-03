import React from 'react';
import { Box, Card, CardContent, Typography, Chip, Grid } from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';

const AthleteProfiles = ({ athleteProfiles }) => {
  if (!athleteProfiles || Object.keys(athleteProfiles).length === 0) {
    return null;
  }

  return (
    <Card sx={{ mb: 2, bgcolor: 'background.paper' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <PersonIcon /> Athlete Profiles
        </Typography>
        <Grid container spacing={2}>
          {Object.entries(athleteProfiles).map(([key, profile]) => (
            <Grid item xs={12} md={6} key={key}>
              <Box
                sx={{
                  p: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                  bgcolor: 'background.default'
                }}
              >
                <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                  {profile.name || key}
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {profile.style}
                </Typography>
                {profile.strengths && profile.strengths.length > 0 && (
                  <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {profile.strengths.map((strength, idx) => (
                      <Chip
                        key={idx}
                        label={strength}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                    ))}
                  </Box>
                )}
              </Box>
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default AthleteProfiles;
