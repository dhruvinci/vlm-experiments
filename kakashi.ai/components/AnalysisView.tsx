import React, { useState, useEffect, useMemo } from 'react';
import { ChevronLeft, Play, Pause, Share2, Download, Shield, Target, Brain, Zap, Maximize2, Layers, Activity } from 'lucide-react';
import { VideoAnalysis, Segment } from '../types';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface AnalysisViewProps {
  analysis: VideoAnalysis;
  onBack: () => void;
}

export const AnalysisView: React.FC<AnalysisViewProps> = ({ analysis, onBack }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [activeTab, setActiveTab] = useState<'details' | 'summary'>('details');
  const duration = analysis.duration;

  // Simulate video playback
  useEffect(() => {
    let interval: number;
    if (isPlaying) {
      interval = window.setInterval(() => {
        setCurrentTime((prev) => {
          if (prev >= duration) {
            setIsPlaying(false);
            return duration;
          }
          return prev + 1;
        });
      }, 1000); 
    }
    return () => clearInterval(interval);
  }, [isPlaying, duration]);

  const currentSegment = useMemo(() => {
    return analysis.segments.find(s => currentTime >= s.startTime && currentTime <= s.endTime) || analysis.segments[0];
  }, [currentTime, analysis.segments]);

  const togglePlay = () => setIsPlaying(!isPlaying);
  
  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = x / rect.width;
    setCurrentTime(percentage * duration);
  };

  const jumpToSegment = (seg: Segment) => {
    setCurrentTime(seg.startTime);
    setIsPlaying(true);
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const getSegmentColor = (type: string) => {
    switch (type) {
      case 'submission': return 'bg-accent'; // Pink
      case 'pass': return 'bg-primary'; // Violet
      case 'scramble': return 'bg-orange-500';
      case 'control': return 'bg-secondary'; // Cyan
      default: return 'bg-zinc-700';
    }
  };

  return (
    <div className="flex flex-col h-full bg-background relative">
      {/* Mobile Top Header (only back button) or Desktop Header */}
      <div className="h-14 lg:h-16 border-b border-white/5 flex items-center px-4 lg:px-6 justify-between bg-surface/80 backdrop-blur-xl z-20 shrink-0 sticky top-0">
        <div className="flex items-center gap-4">
          <button onClick={onBack} className="p-2 -ml-2 hover:bg-white/5 rounded-lg text-textMuted hover:text-white transition-colors flex items-center gap-2">
            <ChevronLeft className="w-5 h-5" />
            <span className="hidden lg:inline text-sm font-medium">Back to Hub</span>
          </button>
          <div className="h-6 w-px bg-white/10 hidden lg:block" />
          <div className="hidden lg:block">
            <h2 className="font-bold text-sm text-white tracking-wide">{analysis.title}</h2>
          </div>
        </div>
        
        {/* Tab Switcher */}
        <div className="flex bg-white/5 p-1 rounded-lg">
          <button 
             onClick={() => setActiveTab('details')}
             className={`px-3 lg:px-6 py-1.5 rounded-md text-xs lg:text-sm font-medium transition-all ${activeTab === 'details' ? 'bg-primary text-white shadow-lg' : 'text-textMuted hover:text-white'}`}
          >
            Vision
          </button>
          <button 
             onClick={() => setActiveTab('summary')}
             className={`px-3 lg:px-6 py-1.5 rounded-md text-xs lg:text-sm font-medium transition-all ${activeTab === 'summary' ? 'bg-primary text-white shadow-lg' : 'text-textMuted hover:text-white'}`}
          >
            Debrief
          </button>
        </div>

        <div className="hidden lg:flex items-center gap-2">
          <button className="p-2 text-textMuted hover:text-white hover:bg-white/5 rounded-lg"><Share2 className="w-4 h-4" /></button>
          <button className="p-2 text-textMuted hover:text-white hover:bg-white/5 rounded-lg"><Download className="w-4 h-4" /></button>
        </div>
      </div>

      {activeTab === 'summary' ? (
        <SummaryView analysis={analysis} />
      ) : (
        <div className="flex-1 flex flex-col lg:flex-row overflow-hidden relative">
          
          {/* VIDEO CONTAINER - STICKY ON MOBILE */}
          <div className="w-full lg:flex-1 flex flex-col bg-black sticky top-0 lg:relative z-10 shrink-0 border-b lg:border-r border-white/10">
            {/* Video Player */}
            <div className="aspect-video lg:flex-1 relative group bg-[#050505] flex items-center justify-center overflow-hidden">
               <img 
                src={analysis.thumbnailUrl} 
                className="absolute inset-0 w-full h-full object-cover opacity-60" 
                alt="Video Frame"
              />
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-transparent via-black/20 to-black/80" />
              
              {/* "Scanning" Overlay Effect */}
              <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-10 pointer-events-none"></div>
              
              {/* Bounding Box Simulation (Visual Flair) */}
              <div className="absolute top-1/4 left-1/4 w-1/2 h-1/2 border border-primary/30 rounded-lg pointer-events-none hidden lg:block">
                 <div className="absolute -top-2 -left-2 w-4 h-4 border-t-2 border-l-2 border-primary"></div>
                 <div className="absolute -top-2 -right-2 w-4 h-4 border-t-2 border-r-2 border-primary"></div>
                 <div className="absolute -bottom-2 -left-2 w-4 h-4 border-b-2 border-l-2 border-primary"></div>
                 <div className="absolute -bottom-2 -right-2 w-4 h-4 border-b-2 border-r-2 border-primary"></div>
                 <div className="absolute top-2 left-2 bg-primary/20 backdrop-blur px-2 py-0.5 rounded text-[10px] text-primary font-mono">
                    CONF: {Math.round(currentSegment.confidence)}%
                 </div>
              </div>

              {!isPlaying && (
                <button 
                  onClick={togglePlay}
                  className="absolute p-4 lg:p-6 rounded-full bg-white/10 backdrop-blur border border-white/20 text-white hover:scale-110 transition-transform z-20 hover:bg-primary hover:border-primary"
                >
                  <Play className="w-8 h-8 lg:w-10 lg:h-10 fill-current" />
                </button>
              )}

              {/* Controls */}
              <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black to-transparent opacity-100 lg:opacity-0 lg:group-hover:opacity-100 transition-opacity">
                 <div className="flex items-center justify-between">
                     <div className="flex items-center gap-4">
                       <button onClick={togglePlay} className="text-white hover:text-primary transition-colors">
                         {isPlaying ? <Pause className="w-5 h-5 fill-current" /> : <Play className="w-5 h-5 fill-current" />}
                       </button>
                       <span className="text-xs font-mono text-white/80">{formatTime(currentTime)} / {formatTime(duration)}</span>
                     </div>
                     <Maximize2 className="w-4 h-4 text-white/70" />
                 </div>
              </div>
            </div>

            {/* Timeline Area - Compact on Mobile */}
            <div className="bg-[#09090b] border-t border-white/5 p-3 lg:p-4">
               {/* The Track */}
               <div 
                 className="relative h-8 lg:h-12 w-full bg-black rounded cursor-pointer overflow-hidden ring-1 ring-white/10 group"
                 onClick={handleTimelineClick}
               >
                 {/* Grid Lines in Timeline */}
                 <div className="absolute inset-0 w-full h-full opacity-20 bg-[linear-gradient(90deg,transparent_0%,transparent_49%,#fff_50%,transparent_51%)] bg-[length:10%_100%] pointer-events-none"></div>

                 {analysis.segments.map((seg) => (
                   <div
                     key={seg.id}
                     className={`absolute top-0 bottom-0 ${getSegmentColor(seg.type)} opacity-40 hover:opacity-80 transition-opacity border-r border-black/30`}
                     style={{
                       left: `${(seg.startTime / duration) * 100}%`,
                       width: `${((seg.endTime - seg.startTime) / duration) * 100}%`
                     }}
                   />
                 ))}
                 
                 {/* Playhead */}
                 <div 
                   className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_15px_white] z-20 pointer-events-none transition-all duration-75"
                   style={{ left: `${(currentTime / duration) * 100}%` }}
                 >
                    <div className="absolute -top-1 -left-1.5 w-4 h-4 bg-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity"></div>
                 </div>
               </div>
            </div>
          </div>

          {/* SCROLLABLE HUD AREA */}
          <div className="w-full lg:w-[420px] bg-background/50 backdrop-blur-md flex flex-col overflow-y-auto border-l border-white/5">
            
            {/* Live Context Header */}
            <div className="p-5 border-b border-white/5 bg-[#050505] sticky top-0 z-10 shadow-lg">
              <div className="flex items-center justify-between mb-2">
                 <span className="text-[10px] font-mono text-primary uppercase tracking-widest flex items-center gap-1.5">
                    <Activity className="w-3 h-3" /> Live Analysis
                 </span>
                 <div className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[10px] font-mono text-textMuted">
                    ID: {currentSegment.id}
                 </div>
              </div>
              <h2 className="text-xl font-bold text-white mb-3 leading-tight tracking-tight">{currentSegment.title}</h2>
              
              {/* Chips */}
              <div className="flex flex-wrap gap-2">
                 {currentSegment.tags.map(tag => (
                   <span key={tag} className="text-[10px] uppercase font-bold px-2 py-1 rounded bg-white/5 border border-white/10 text-gray-300 hover:border-primary/50 transition-colors">
                     #{tag.replace('_', ' ')}
                   </span>
                 ))}
              </div>
            </div>

            {/* Metrics Row */}
            <div className="grid grid-cols-3 gap-px bg-white/5 border-b border-white/5">
               {[
                 { label: 'Intensity', value: currentSegment.metrics.action, icon: Zap, color: 'text-yellow-400' },
                 { label: 'Control', value: currentSegment.metrics.control, icon: Shield, color: 'text-secondary' },
                 { label: 'Threat', value: currentSegment.metrics.threat, icon: Target, color: 'text-accent' },
               ].map((m, i) => (
                 <div key={i} className="bg-[#0a0a0a] p-4 flex flex-col items-center justify-center hover:bg-[#121212] transition-colors">
                   <m.icon className={`w-4 h-4 ${m.color} mb-2`} />
                   <span className="text-lg font-mono font-bold text-white">{m.value}</span>
                   <span className="text-[10px] uppercase text-textMuted tracking-wider mt-1">{m.label}</span>
                 </div>
               ))}
            </div>

            {/* Scrollable Content */}
            <div className="p-5 space-y-6">
              
              {/* Coach Notes */}
              <div className="space-y-2 relative">
                <div className="absolute left-0 top-0 bottom-0 w-px bg-gradient-to-b from-primary/50 to-transparent"></div>
                <div className="pl-4">
                  <h3 className="text-xs font-bold text-textMuted uppercase tracking-widest mb-2 flex items-center gap-2">
                     Coach's Eye
                  </h3>
                  <p className="text-sm leading-relaxed text-gray-300">
                    {currentSegment.notes.coach}
                  </p>
                </div>
              </div>

              {/* Strategy */}
              <div className="p-4 rounded-xl bg-gradient-to-br from-white/5 to-transparent border border-white/5">
                <h3 className="text-xs font-bold text-secondary uppercase tracking-widest mb-2 flex items-center gap-2">
                  <Brain className="w-3 h-3" /> Strategy
                </h3>
                <p className="text-sm text-textMuted leading-relaxed">
                  {currentSegment.notes.strategy}
                </p>
              </div>

              {/* Next Steps */}
              <div className="space-y-3">
                <h3 className="text-xs font-bold text-textMuted uppercase tracking-widest">Suggested Adjustment</h3>
                <div className="flex items-start gap-3 p-3 rounded-lg border border-dashed border-white/10 bg-black/40">
                   <div className="w-5 h-5 rounded-full bg-primary/20 flex items-center justify-center text-xs font-bold text-primary shrink-0 mt-0.5">1</div>
                   <p className="text-sm text-gray-300">{currentSegment.notes.whatNext}</p>
                </div>
              </div>

            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const SummaryView = ({ analysis }: { analysis: VideoAnalysis }) => {
  const data = analysis.segments.map(s => ({
    name: s.title.split(' ')[0],
    action: s.metrics.action,
    control: s.metrics.control
  }));

  return (
    <div className="flex-1 overflow-y-auto p-4 lg:p-12 max-w-6xl mx-auto w-full">
      <div className="mb-8 lg:mb-12 text-center">
        <h1 className="text-2xl lg:text-3xl font-bold text-white mb-2">Session De-Brief</h1>
        <div className="flex justify-center gap-8 mt-6">
          <div className="text-center p-4 rounded-2xl bg-white/5 border border-white/5">
             <div className="text-3xl lg:text-4xl font-mono font-bold text-primary">{analysis.summary.avgAction}%</div>
             <div className="text-[10px] uppercase tracking-widest text-textMuted mt-1">Avg Action</div>
          </div>
          <div className="text-center p-4 rounded-2xl bg-white/5 border border-white/5">
             <div className="text-3xl lg:text-4xl font-mono font-bold text-secondary">{analysis.summary.avgControl}%</div>
             <div className="text-[10px] uppercase tracking-widest text-textMuted mt-1">Avg Control</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="glass-panel p-6 rounded-2xl">
           <h3 className="font-bold text-white mb-6 text-sm uppercase tracking-widest flex items-center gap-2">
             <Layers className="w-4 h-4 text-textMuted" /> Flow Analysis
           </h3>
           <div className="h-64 w-full">
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={data}>
                 <XAxis dataKey="name" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                 <Tooltip 
                    cursor={{fill: 'rgba(255,255,255,0.05)'}}
                    contentStyle={{ backgroundColor: '#09090b', border: '1px solid #27272a', borderRadius: '8px', fontSize: '12px' }}
                    itemStyle={{ color: '#f4f4f5' }}
                 />
                 <Bar dataKey="action" fill="#8b5cf6" radius={[2, 2, 0, 0]} barSize={30} />
                 <Bar dataKey="control" fill="#06b6d4" radius={[2, 2, 0, 0]} barSize={30} />
               </BarChart>
             </ResponsiveContainer>
           </div>
        </div>

        <div className="space-y-4">
           <div className="glass-panel p-6 rounded-2xl border-l-2 border-l-accent relative overflow-hidden">
             <div className="absolute top-0 right-0 p-3 opacity-10">
               <Brain className="w-16 h-16 text-accent" />
             </div>
             <h3 className="text-xs font-bold text-accent uppercase tracking-widest mb-2">Focus Area</h3>
             <p className="text-lg font-bold text-white">"{analysis.summary.keyWeakness}"</p>
             <p className="text-textMuted text-sm mt-3 leading-relaxed">
               System detected repeated loss of structure during leg entanglement phases. Recommend drilling defensive knee alignment.
             </p>
           </div>
           
           <div className="glass-panel p-6 rounded-2xl border-l-2 border-l-primary">
             <h3 className="text-xs font-bold text-primary uppercase tracking-widest mb-2">Dominant Positions</h3>
             <div className="flex flex-wrap gap-2">
               {analysis.summary.dominantPositions.map(p => (
                 <span key={p} className="px-3 py-1 bg-primary/10 rounded-full text-xs font-bold text-white border border-primary/20">{p}</span>
               ))}
             </div>
           </div>
        </div>
      </div>
    </div>
  );
};