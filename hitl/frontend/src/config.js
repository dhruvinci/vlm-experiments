/**
 * Configuration for the BJJ Video Analyzer frontend
 */

const config = {
  // API configuration
  api: {
    // Local development API endpoint
    local: 'http://localhost:5002',

    // Production API endpoint (update this when deploying to RunPod)
    // Format: https://<pod-id>-8000.proxy.runpod.net
    // Example: https://abcd1234-8000.proxy.runpod.net
    production: 'https://kn40wbnvvdt6jg-8000.proxy.runpod.net',

    // Set to true to use production API endpoint
    // IMPORTANT: Update this to true and set the production URL above when deploying
    useProduction: false,
    
    // Get the current API endpoint based on configuration
    get baseUrl() {
      return this.useProduction ? this.production : this.local;
    },
    
    // API endpoints
    endpoints: {
      upload: '/upload',
      analyze: '/analyze',
      status: (jobId) => `/status/${jobId}`,
      results: (jobId) => `/results/${jobId}`
    }
  },
  
  // Analysis configuration defaults
  analysis: {
    // Adaptive frame sampling based on video length (handled by backend)
    defaultFrameSamplingRate: 8,  // For short videos (<2 min)
    defaultBatchSize: 64,
    // Use the enhanced prompt from the backend by default
    useDefaultPrompt: true,
    // Custom prompt option (only used if useDefaultPrompt is false)
    defaultPrompt: "Analyze this BJJ video with detailed position tracking and transitions."
  },
  
  // UI configuration
  ui: {
    // Timeline configuration
    timeline: {
      positionColors: {
        mount: '#f44336',
        guard: '#4caf50',
        back: '#ff9800',
        side: '#9c27b0',
        halfguard: '#2196f3',
        closed: '#009688',
        open: '#cddc39',
        turtle: '#795548',
        default: '#3f51b5'
      }
    }
  }
};

export default config;
