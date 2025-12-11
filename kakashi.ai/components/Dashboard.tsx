import React from 'react';
import { Plus, Search, Filter, PlayCircle, Clock, ChevronRight, Activity, TrendingUp } from 'lucide-react';
import { ViewState } from '../App';
import { MOCK_ANALYSIS } from '../constants';

interface DashboardProps {
  onNavigate: (view: ViewState, id?: string) => void;
  onUpload: () => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onNavigate, onUpload }) => {
  const recentAnalyses = [
    MOCK_ANALYSIS,
    { ...MOCK_ANALYSIS, id: 'v2', title: 'Worlds 2023 - Round 1', date: '2023-09-12', context: 'Competition', type: 'Gi', status: 'completed' },
    { ...MOCK_ANALYSIS, id: 'v3', title: 'Open Mat Rolls', date: '2023-09-10', context: 'Training', type: 'No-Gi', status: 'processing' },
    { ...MOCK_ANALYSIS, id: 'v4', title: 'Drilling Leg Locks', date: '2023-09-08', context: 'Training', type: 'No-Gi', status: 'completed' },
  ];

  return (
    <div className="p-4 md:p-8 lg:p-12 max-w-[1600px] mx-auto space-y-6 md:space-y-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 md:gap-6">
        <div className="space-y-1 md:space-y-2">
          <div className="flex items-center gap-2 text-primary font-mono text-[10px] md:text-xs uppercase tracking-widest">
            <span className="w-2 h-2 bg-primary rounded-full animate-pulse"></span>
            System Online
          </div>
          <h1 className="text-2xl md:text-4xl font-bold tracking-tight text-white">Training Hub</h1>
        </div>
        <button 
          onClick={onUpload}
          className="flex items-center justify-center px-6 py-3 bg-white text-black font-bold rounded-lg hover:bg-gray-200 transition-all active:scale-95 shadow-[0_0_20px_rgba(255,255,255,0.1)] group w-full md:w-auto text-sm md:text-base"
        >
          <Plus className="w-4 h-4 md:w-5 md:h-5 mr-2 transition-transform group-hover:rotate-90" />
          New Analysis
        </button>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-3 gap-2 md:gap-4">
        {[
          { label: 'Weekly Volume', value: '4.2h', sub: 'Mat Time', change: '+12%' },
          { label: 'Pass Rate', value: '68%', sub: 'Technique', change: '+5%' },
          { label: 'Sub Rate', value: '22%', sub: 'Efficiency', change: '-2%' },
        ].map((stat, i) => (
          <div key={i} className="glass-panel p-3 md:p-5 rounded-xl border border-white/5 flex flex-col justify-center">
            <div className="flex items-center justify-between mb-1 md:mb-4">
               <p className="text-[9px] md:text-xs text-textMuted font-mono uppercase tracking-widest truncate max-w-[80px] md:max-w-none" title={stat.label}>{stat.label}</p>
               <span className={`text-[9px] md:text-xs font-bold ${stat.change.startsWith('+') ? 'text-emerald-400' : 'text-red-400'} bg-white/5 px-1.5 py-0.5 rounded ml-1`}>
                 {stat.change}
               </span>
            </div>
            <div className="flex flex-col">
              <span className="text-lg md:text-4xl font-mono font-bold text-white tracking-tighter leading-none">{stat.value}</span>
              <span className="text-[9px] md:text-xs text-textMuted mt-1 truncate">{stat.sub}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="space-y-4">
        {/* Filters */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
           <h2 className="text-lg md:text-xl font-bold text-white flex items-center gap-2">
             Recent Footage <span className="text-xs font-normal text-textMuted bg-white/5 px-2 py-0.5 rounded-full font-mono">{recentAnalyses.length}</span>
           </h2>
           <div className="flex items-center gap-2 bg-surface/50 p-1 rounded-lg border border-white/5 w-full sm:w-auto">
             <div className="relative flex-1 sm:flex-none">
                <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-textMuted" />
                <input 
                  type="text" 
                  placeholder="Search..." 
                  className="bg-transparent border-none outline-none pl-9 pr-4 py-1.5 text-sm text-white placeholder-textMuted w-full sm:w-48 focus:w-64 transition-all"
                />
             </div>
             <button className="p-1.5 hover:bg-white/10 rounded-md text-textMuted hover:text-white transition-colors">
               <Filter className="w-4 h-4" />
             </button>
           </div>
        </div>

        {/* Video Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 md:gap-6">
          {recentAnalyses.map((vid) => (
            <div 
              key={vid.id}
              onClick={() => vid.status === 'completed' && onNavigate('analysis', vid.id)}
              className={`group relative flex flex-col bg-[#0c0c0e] rounded-xl overflow-hidden border border-white/5 hover:border-primary/30 transition-all duration-300 hover:shadow-[0_0_30px_rgba(0,0,0,0.5)] cursor-pointer ${vid.status !== 'completed' ? 'opacity-80' : ''}`}
            >
              {/* Thumbnail */}
              <div className="relative aspect-video overflow-hidden bg-surfaceHighlight">
                <img 
                  src={vid.thumbnailUrl} 
                  alt={vid.title} 
                  className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105 opacity-60 group-hover:opacity-100"
                />
                
                {/* Overlays */}
                <div className="absolute inset-0 bg-gradient-to-t from-[#0c0c0e] via-transparent to-transparent opacity-80" />
                
                <div className="absolute bottom-3 left-3 right-3 flex justify-between items-end">
                   <div className="flex gap-1.5">
                      <span className="text-[10px] font-bold font-mono bg-black/50 backdrop-blur border border-white/10 px-1.5 py-0.5 rounded text-textMuted group-hover:text-white transition-colors">
                        {vid.type}
                      </span>
                   </div>
                   <div className="bg-black/80 backdrop-blur text-[10px] font-mono font-medium px-1.5 py-0.5 rounded text-white flex items-center border border-white/10">
                    <Clock className="w-3 h-3 mr-1 text-primary" />
                    {Math.floor(vid.duration / 60)}:{(vid.duration % 60).toString().padStart(2, '0')}
                  </div>
                </div>

                {/* Status Overlay */}
                {vid.status !== 'completed' && (
                  <div className="absolute inset-0 bg-black/80 backdrop-blur-[2px] flex flex-col items-center justify-center">
                    <Activity className="w-8 h-8 text-primary animate-pulse mb-2" />
                    <span className="text-xs font-mono font-bold tracking-widest text-primary uppercase">Analyzing</span>
                    <span className="text-[10px] text-textMuted mt-1">Computer Vision Active</span>
                  </div>
                )}
                
                {/* Play Button Overlay */}
                {vid.status === 'completed' && (
                  <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    <div className="w-12 h-12 rounded-full bg-white/10 backdrop-blur border border-white/20 flex items-center justify-center">
                      <PlayCircle className="w-6 h-6 text-white" />
                    </div>
                  </div>
                )}
              </div>

              {/* Content */}
              <div className="p-4 flex flex-col flex-1">
                <div className="flex-1">
                  <h3 className="text-sm md:text-base font-bold text-white mb-1 line-clamp-1 group-hover:text-primary transition-colors">{vid.title}</h3>
                  <p className="text-[10px] md:text-xs text-textMuted font-mono mb-3">{vid.date}</p>
                </div>

                <div className="flex items-center justify-between pt-3 border-t border-white/5 mt-2">
                  <div className="flex items-center gap-2">
                     <span className={`w-1.5 h-1.5 rounded-full ${vid.context === 'Competition' ? 'bg-accent' : 'bg-secondary'}`} />
                     <span className="text-[10px] md:text-xs text-textMuted">{vid.context}</span>
                  </div>
                  <ChevronRight className="w-3 h-3 md:w-4 md:h-4 text-white/20 group-hover:text-primary transition-colors" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};