import React from 'react';
import { CURRENT_USER } from '../constants';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import { Medal, Flame, Zap, Award, Target, Share2 } from 'lucide-react';

export const ProfileView: React.FC = () => {
  const user = CURRENT_USER;
  
  const chartData = [
    { subject: 'AGGRO', A: user.attributes.aggression, fullMark: 100 },
    { subject: 'DEF', A: user.attributes.defense, fullMark: 100 },
    { subject: 'TECH', A: user.attributes.technique, fullMark: 100 },
    { subject: 'CARDIO', A: user.attributes.cardio, fullMark: 100 },
    { subject: 'SCRMB', A: user.attributes.scramble, fullMark: 100 },
    { subject: 'SUB', A: user.attributes.submission, fullMark: 100 },
  ];

  return (
    <div className="p-4 md:p-8 lg:p-12 max-w-6xl mx-auto space-y-6 lg:space-y-8">
      {/* Header Profile Card */}
      <div className="glass-panel rounded-3xl p-6 lg:p-10 mb-8 relative overflow-hidden border border-white/5 group">
        <div className="absolute -top-32 -right-32 w-96 h-96 bg-primary/20 blur-[120px] rounded-full pointer-events-none group-hover:bg-primary/30 transition-colors duration-1000" />
        
        <div className="flex flex-col md:flex-row items-center gap-6 lg:gap-10 relative z-10">
          <div className="relative shrink-0">
             <img src={user.avatarUrl} alt={user.name} className="w-24 h-24 lg:w-32 lg:h-32 rounded-full border-4 border-[#09090b] shadow-2xl" />
             <div className="absolute -bottom-2 -right-2 bg-purple-600 text-white text-[10px] lg:text-xs font-bold px-3 py-1 rounded-full border border-[#09090b] uppercase tracking-wider shadow-lg">
               {user.belt}
             </div>
          </div>
          
          <div className="text-center md:text-left flex-1 space-y-2">
            <div>
              <h1 className="text-2xl lg:text-4xl font-bold text-white tracking-tight">{user.name}</h1>
              <p className="text-textMuted font-mono text-xs lg:text-sm">{user.handle} • {user.academy}</p>
            </div>
            
            <div className="flex flex-wrap justify-center md:justify-start gap-3 pt-2">
              <div className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 flex items-center gap-2 hover:bg-white/10 transition-colors cursor-default">
                <Award className="w-3 h-3 lg:w-4 lg:h-4 text-yellow-500" />
                <span className="text-xs lg:text-sm font-bold text-gray-200">{user.archetype}</span>
              </div>
              <button className="px-3 py-1.5 rounded-lg border border-white/10 hover:border-white/30 transition-colors">
                <Share2 className="w-3 h-3 lg:w-4 lg:h-4 text-textMuted" />
              </button>
            </div>
          </div>

          <div className="flex gap-6 lg:gap-10 text-center border-t md:border-t-0 md:border-l border-white/10 pt-6 md:pt-0 pl-0 md:pl-10 w-full md:w-auto justify-center md:justify-start">
             <div>
               <div className="text-2xl lg:text-3xl font-mono font-bold text-white">{user.stats.rollsAnalyzed}</div>
               <div className="text-[10px] text-textMuted uppercase tracking-widest mt-1">Rolls</div>
             </div>
             <div>
               <div className="text-2xl lg:text-3xl font-mono font-bold text-primary">{user.stats.submissions}</div>
               <div className="text-[10px] text-textMuted uppercase tracking-widest mt-1">Subs</div>
             </div>
             <div>
               <div className="text-2xl lg:text-3xl font-mono font-bold text-white">{user.stats.hoursLogged}</div>
               <div className="text-[10px] text-textMuted uppercase tracking-widest mt-1">Hours</div>
             </div>
          </div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Radar Chart */}
        <div className="glass-panel p-4 lg:p-8 rounded-3xl min-h-[400px] flex flex-col relative border border-white/5">
          <div className="flex justify-between items-start mb-4">
            <h3 className="text-sm font-bold text-white uppercase tracking-widest flex items-center gap-2">
               <Zap className="w-4 h-4 text-primary" /> Attribute Matrix
            </h3>
          </div>
          <div className="flex-1 w-full flex items-center justify-center">
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart cx="50%" cy="50%" outerRadius="70%" data={chartData}>
                <PolarGrid stroke="#3f3f46" strokeDasharray="3 3" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#71717a', fontSize: 10, fontWeight: 'bold' }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                <Radar
                  name={user.name}
                  dataKey="A"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  fill="#8b5cf6"
                  fillOpacity={0.2}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Activity & Badges */}
        <div className="space-y-6">
           <div className="glass-panel p-6 rounded-3xl border border-white/5">
              <h3 className="text-sm font-bold text-white uppercase tracking-widest mb-6">Consistency Heatmap</h3>
              <div className="flex gap-1.5 flex-wrap justify-center md:justify-start">
                 {Array.from({ length: 90 }).map((_, i) => {
                   const opacity = Math.random();
                   return (
                     <div 
                        key={i} 
                        className={`w-3 h-3 rounded-[2px] transition-colors duration-500 hover:scale-125 ${opacity > 0.8 ? 'bg-primary shadow-[0_0_8px_rgba(139,92,246,0.5)]' : opacity > 0.4 ? 'bg-primary/30' : 'bg-[#1a1a1c]'}`}
                     />
                   )
                 })}
              </div>
              <p className="mt-6 text-xs text-textMuted font-mono">Current Streak: <span className="text-white font-bold">4 Days</span></p>
           </div>

           <div className="glass-panel p-6 rounded-3xl border border-white/5">
              <h3 className="text-sm font-bold text-white uppercase tracking-widest mb-4">Mastery Badges</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                 <div className="flex items-center gap-4 p-4 rounded-2xl bg-gradient-to-br from-white/5 to-transparent border border-white/5 hover:border-yellow-500/30 transition-colors group cursor-pointer">
                    <div className="w-12 h-12 rounded-full bg-yellow-500/10 flex items-center justify-center text-yellow-500 group-hover:scale-110 transition-transform">
                       <Medal className="w-6 h-6" />
                    </div>
                    <div>
                       <div className="font-bold text-white text-sm">Guard Retention</div>
                       <div className="text-[10px] text-textMuted uppercase tracking-wider">Level 5 Master</div>
                    </div>
                 </div>
                 <div className="flex items-center gap-4 p-4 rounded-2xl bg-gradient-to-br from-white/5 to-transparent border border-white/5 hover:border-red-500/30 transition-colors group cursor-pointer">
                    <div className="w-12 h-12 rounded-full bg-red-500/10 flex items-center justify-center text-red-500 group-hover:scale-110 transition-transform">
                       <Target className="w-6 h-6" />
                    </div>
                    <div>
                       <div className="font-bold text-white text-sm">Leg Locker</div>
                       <div className="text-[10px] text-textMuted uppercase tracking-wider">Level 3 Adept</div>
                    </div>
                 </div>
              </div>
           </div>
        </div>
      </div>
    </div>
  );
};