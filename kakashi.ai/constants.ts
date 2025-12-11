import { VideoAnalysis, UserProfile, SocialPost } from './types';

export const CURRENT_USER: UserProfile = {
  id: 'u1',
  handle: '@mat_rat_99',
  name: 'Alex Pereira',
  avatarUrl: 'https://picsum.photos/seed/alex/150/150',
  belt: 'Purple Belt',
  academy: 'Renzo Gracie NYC',
  stats: {
    rollsAnalyzed: 142,
    submissions: 34,
    sweeps: 89,
    hoursLogged: 320,
  },
  attributes: {
    aggression: 85,
    defense: 60,
    technique: 75,
    cardio: 90,
    scramble: 80,
    submission: 70,
  },
  archetype: 'The Pressure Passer',
};

export const MOCK_ANALYSIS: VideoAnalysis = {
  id: 'v123',
  title: 'Sparring vs. Mike (Heavyweight)',
  date: '2023-10-24',
  duration: 181, // 3:01
  thumbnailUrl: 'https://picsum.photos/seed/bjj1/600/400',
  status: 'completed',
  type: 'No-Gi',
  context: 'Training',
  summary: {
    avgControl: 31,
    avgAction: 50,
    dominantPositions: ['Side Control', 'North South'],
    keyWeakness: 'Leg Entanglement Entry Defense',
  },
  segments: [
    {
      id: 's1',
      startTime: 0,
      endTime: 15,
      type: 'standing',
      title: 'Hand Fighting / Setup',
      confidence: 98,
      intensity: 20,
      tags: ['feel_out', 'distance_management'],
      notes: {
        coach: 'Good low stance. You are hand fighting well but giving up inside position too easily.',
        strategy: 'Jones\'s strategy is to avoid a standing wrestling exchange and force a guard pull.',
        whatNext: 'Look for the arm drag or snap down to create an angle.',
      },
      metrics: { action: 20, control: 0, threat: 10 }
    },
    {
      id: 's2',
      startTime: 15,
      endTime: 35,
      type: 'scramble',
      title: 'Takedown Defense -> Scramble',
      confidence: 85,
      intensity: 80,
      tags: ['sprawl', 'front_headlock'],
      notes: {
        coach: 'Excellent reaction to the shot. Your hip pressure was heavy.',
        strategy: 'Opponent overcommitted to the double leg. You capitalized with a sprawl.',
        whatNext: 'Transition to the back or guillotine immediately.',
      },
      metrics: { action: 80, control: 40, threat: 60 }
    },
    {
      id: 's3',
      startTime: 35,
      endTime: 87,
      type: 'guard',
      title: 'Butterfly Guard Retention',
      confidence: 92,
      intensity: 40,
      tags: ['seated_guard', 'hook_sweep_attempt'],
      notes: {
        coach: 'You are flat on your back too often. Sit up to generate drive.',
        strategy: 'Using hooks to keep weight off, but failing to off-balance via kuzushi.',
        whatNext: 'Get an underhook or overhook to connect your chest to theirs.',
      },
      metrics: { action: 40, control: 30, threat: 20 }
    },
    {
      id: 's4',
      startTime: 87,
      endTime: 119,
      type: 'pass',
      title: 'Torreando Pass',
      confidence: 88,
      intensity: 75,
      tags: ['passing', 'speed', 'connection'],
      notes: {
        coach: 'Beautiful redirection of the legs. The hip switch was perfectly timed.',
        strategy: 'Explosive lateral movement creates opening in their retention.',
        whatNext: 'Solidify side control before looking for mount.',
      },
      metrics: { action: 75, control: 80, threat: 50 }
    },
    {
      id: 's5',
      startTime: 119,
      endTime: 145,
      type: 'submission',
      title: 'Leg Entanglement Entry',
      confidence: 95,
      intensity: 90,
      tags: ['saddle_entry', 'inside_heel_hook'],
      notes: {
        coach: 'Risky entry but it paid off. Watch your own heel exposure during the roll.',
        strategy: 'Abandoning the pass to attack the legs caught them off guard.',
        whatNext: 'Secure the secondary leg to prevent them from rolling out.',
      },
      metrics: { action: 90, control: 60, threat: 95 }
    },
     {
      id: 's6',
      startTime: 145,
      endTime: 181,
      type: 'control',
      title: 'Top Control & Reset',
      confidence: 99,
      intensity: 30,
      tags: ['mount', 'pressure'],
      notes: {
        coach: 'Solid pressure cooking here. Good patience.',
        strategy: 'Consolidating position after the scramble.',
        whatNext: 'Isolate an arm for the finish.',
      },
      metrics: { action: 30, control: 90, threat: 40 }
    }
  ]
};

export const MOCK_FEED: SocialPost[] = [
  {
    id: 'p1',
    user: { ...CURRENT_USER, handle: '@gordon_wannabe', name: 'John Danaher Clone' },
    analysisId: 'v999',
    caption: 'Finally hit the backside 50/50 entry we drilled last week. Kakashi says my entry speed was 0.4s faster than average. 🥋🚀',
    timestamp: '2 hours ago',
    likes: 42,
    comments: 5,
  },
  {
    id: 'p2',
    user: { ...CURRENT_USER, handle: '@bjj_globetrotter', name: 'Sarah Connor' },
    analysisId: 'v888',
    caption: 'Getting smashed in comp class. Need to fix my frame structure in bottom side control. Thoughts?',
    timestamp: '5 hours ago',
    likes: 12,
    comments: 18,
  }
];
