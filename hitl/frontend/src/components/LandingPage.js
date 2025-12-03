import React from 'react';
import { Box, Container, Typography, Button, Grid, Card, CardContent } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import AutorenewIcon from '@mui/icons-material/Autorenew';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import GroupIcon from '@mui/icons-material/Group';
import SportsKabaddiIcon from '@mui/icons-material/SportsKabaddi';
import VisibilityIcon from '@mui/icons-material/Visibility';

const LandingPage = () => {
  const navigate = useNavigate();

  const features = [
    {
      image: '/assets/smart-roll-breakdown.png',
      title: 'Smart Roll Breakdown',
      description: 'Understand every exchange with AI clarity.',
    },
    {
      image: '/assets/progress-insights.png',
      title: 'Progress Insights',
      description: 'Track improvements from your rolls, competitions.',
    },
    {
      image: '/assets/technique-library.png',
      title: 'Technique Library',
      description: 'Train with diverse perspectives from coaches using AI.',
    },
    {
      image: '/assets/coaches-ai.png',
      title: 'Coaches + AI',
      description: 'Train with diverse perspectives from coaches using AI.',
    },
    {
      image: '/assets/competition-prep.png',
      title: 'Competition Prep',
      description: 'Sharpen your strategy for every opponent and ruleset.',
    },
    {
      image: '/assets/spectator-mode.png',
      title: 'Spectator Mode',
      description: 'Study iconic matches through the trained eye of Sensai.',
    },
  ];

  return (
    <Box
      sx={{
        minHeight: '100vh',
        bgcolor: 'background.default',
        py: 4,
      }}
    >
      <Container maxWidth="lg">
        {/* Hero Section */}
        <Box
          sx={{
            textAlign: 'center',
            mb: 5,
          }}
        >
          {/* Logo */}
          <Box
            component="img"
            src="/logo-brown.png"
            alt="Sensai Logo"
            sx={{
              width: 175,
              height: 'auto',
              mb: 2.5,
              mx: 'auto',
            }}
          />

          {/* Headline */}
          <Typography
            variant="h1"
            sx={{
              mb: 1.5,
              fontSize: { xs: '1.75rem', md: '2.25rem' },
            }}
          >
            Train the mind. Refine the art.
          </Typography>

          {/* Subheadline */}
          <Typography
            variant="h6"
            sx={{
              mb: 2.5,
              color: 'text.secondary',
              fontWeight: 400,
              fontSize: { xs: '0.9rem', md: '1rem' },
            }}
          >
            Your AI training partner — learning Jiu-Jitsu with you.
          </Typography>

          {/* CTA Button */}
          <Button
            variant="contained"
            size="large"
            sx={{
              color: 'white',
              px: 4,
              py: 1.25,
              fontSize: '0.95rem',
              mb: 2,
            }}
          >
            Request Early Access
          </Button>

          {/* Coach Login Link */}
          <Box>
            <Typography variant="body1" sx={{ color: 'text.secondary' }}>
              Are you a coach?{' '}
              <Typography
                component="span"
                onClick={() => navigate('/hitl')}
                sx={{
                  color: 'primary.main',
                  cursor: 'pointer',
                  textDecoration: 'underline',
                  fontWeight: 600,
                  '&:hover': {
                    color: 'primary.dark',
                  },
                }}
              >
                Log in to the HITL →
              </Typography>
            </Typography>
          </Box>
        </Box>

        {/* Features Section - All Cards Visible */}
        <Box sx={{ mb: 3 }}>
          <Typography
            variant="h2"
            sx={{
              textAlign: 'center',
              mb: 3,
              fontSize: { xs: '1.4rem', md: '1.75rem' },
            }}
          >
            What's next in training evolution
          </Typography>

          <Grid container spacing={1.5}>
            {features.map((feature, index) => (
              <Grid item xs={2} key={index}>
                <Card
                  sx={{
                    height: '100%',
                    textAlign: 'center',
                    transition: 'transform 0.2s',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                    },
                  }}
                >
                  <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
                    <Box 
                      component="img"
                      src={feature.image}
                      alt={feature.title}
                      sx={{ 
                        width: '100%',
                        height: 100,
                        objectFit: 'contain',
                        borderRadius: 1,
                        mb: 1 
                      }}
                    />
                    <Typography
                      variant="h6"
                      sx={{
                        mb: 0.5,
                        fontWeight: 600,
                        fontSize: '0.75rem',
                        lineHeight: 1.2,
                      }}
                    >
                      {feature.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.65rem', lineHeight: 1.3 }}>
                      {feature.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Footer */}
        <Box sx={{ textAlign: 'center', py: 2.5 }}>
          <Typography
            variant="h3"
            sx={{
              fontSize: '1.25rem',
              color: 'primary.main',
              fontWeight: 400,
              letterSpacing: 2,
            }}
          >
            sensai
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default LandingPage;
