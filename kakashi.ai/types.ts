export type SegmentType = 'standing' | 'guard' | 'pass' | 'submission' | 'scramble' | 'control';

export interface Segment {
  id: string;
  startTime: number; // in seconds
  endTime: number;
  type: SegmentType;
  title: string;
  confidence: number;
  intensity: number; // 0-100
  tags: string[];
  notes: {
    coach: string;
    strategy: string;
    whatNext: string;
  };
  metrics: {
    action: number;
    control: number;
    threat: number;
  };
}

export interface VideoAnalysis {
  id: string;
  title: string;
  date: string;
  duration: number; // seconds
  thumbnailUrl: string;
  status: 'processing' | 'completed' | 'failed';
  type: 'Gi' | 'No-Gi';
  context: 'Training' | 'Competition';
  segments: Segment[];
  summary: {
    avgControl: number;
    avgAction: number;
    dominantPositions: string[];
    keyWeakness: string;
  };
}

export interface UserProfile {
  id: string;
  handle: string;
  name: string;
  avatarUrl: string;
  belt: string;
  academy: string;
  stats: {
    rollsAnalyzed: number;
    submissions: number;
    sweeps: number;
    hoursLogged: number;
  };
  attributes: {
    aggression: number;
    defense: number;
    technique: number;
    cardio: number;
    scramble: number;
    submission: number;
  };
  archetype: string;
}

export interface SocialPost {
  id: string;
  user: UserProfile;
  analysisId: string;
  caption: string;
  timestamp: string;
  likes: number;
  comments: number;
  highlightSegment?: Segment;
}