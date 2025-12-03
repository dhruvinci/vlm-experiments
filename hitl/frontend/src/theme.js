import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#5D4037',
      light: '#8B6F47',
      dark: '#3E2723',
    },
    secondary: {
      main: '#D84315',
      light: '#FF6E40',
      dark: '#BF360C',
    },
    background: {
      default: '#F5F1E8',
      paper: '#FFFFFF',
    },
    text: {
      primary: '#3E2723',
      secondary: '#5D4037',
    },
  },
  typography: {
    fontFamily: '"Satoshi Variable", "Aeonik", -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
    h1: {
      fontFamily: '"Satoshi Variable", "Aeonik", sans-serif',
      fontWeight: 700,
      fontSize: '3rem',
      color: '#0A0A0A',
    },
    h2: {
      fontFamily: '"Satoshi Variable", "Aeonik", sans-serif',
      fontWeight: 600,
      fontSize: '2.5rem',
      color: '#0A0A0A',
    },
    h3: {
      fontFamily: '"Satoshi Variable", "Aeonik", sans-serif',
      fontWeight: 600,
      fontSize: '2rem',
      color: '#0A0A0A',
    },
    h4: {
      fontFamily: '"Satoshi Variable", "Aeonik", sans-serif',
      fontWeight: 600,
      fontSize: '1.5rem',
      color: '#0A0A0A',
    },
    h6: {
      fontFamily: '"Satoshi Variable", "Aeonik", sans-serif',
      fontWeight: 600,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '12px 32px',
          fontSize: '1rem',
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 4px 12px rgba(93, 64, 55, 0.3)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
          '&:hover': {
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.12)',
          },
        },
      },
    },
  },
});

export default theme;
