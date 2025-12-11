import React, { useState } from 'react';
import { MOCK_FEED } from '../constants';
import { ViewState } from '../App';
import { Heart, MessageCircle, Share2, Play, MoreHorizontal, Users, Globe } from 'lucide-react';

interface SocialFeedProps {
  onNavigate: (view: ViewState) => void;
}

export const SocialFeed: React.FC<SocialFeedProps> = ({ onNavigate }) => {
  const [feedScope, setFeedScope] = useState<'gym' | 'global'>('global');

  return (
    <div className="p-4 md:p-8 max-w-2xl mx-auto pb-24">
      <div className="flex items-center justify-between mb-8">
         <h1 className="text-3xl font-bold tracking-tight text-white">Feed</h1>
         
         {/* Feed Scope Selector */}
         <div className="flex bg-white/5 p-1 rounded-lg border border-white/5">
            <button 
              onClick={() => setFeedScope('gym')} 
              className={`px-3 py-1.5 flex items-center gap-2 rounded-md transition-all ${feedScope === 'gym' ? 'bg-primary/20 text-primary shadow-sm' : 'text-textMuted hover:text-white'}`}
            >
              <Users className="w-3 h-3" />
              <span className="text-[10px] font-mono font-bold uppercase tracking-wider">My Gym</span>
            </button>
            <button 
              onClick={() => setFeedScope('global')}
              className={`px-3 py-1.5 flex items-center gap-2 rounded-md transition-all ${feedScope === 'global' ? 'bg-primary/20 text-primary shadow-sm' : 'text-textMuted hover:text-white'}`}
            >
              <Globe className="w-3 h-3" />
              <span className="text-[10px] font-mono font-bold uppercase tracking-wider">Global</span>
            </button>
         </div>
      </div>
      
      <div className="space-y-8">
        {MOCK_FEED.map((post) => (
          <div key={post.id} className="bg-surface/40 backdrop-blur rounded-2xl overflow-hidden border border-white/5 hover:border-white/10 transition-colors">
            {/* Header */}
            <div className="p-4 flex items-center justify-between">
              <div className="flex items-center gap-3 cursor-pointer" onClick={() => onNavigate('profile')}>
                <div className="relative">
                  <img src={post.user.avatarUrl} alt={post.user.handle} className="w-10 h-10 rounded-full border border-white/10" />
                  <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-primary rounded-full border border-black"></div>
                </div>
                <div>
                  <div className="font-bold text-white text-sm hover:text-primary transition-colors">{post.user.name}</div>
                  <div className="text-xs text-textMuted font-mono">{post.user.handle} • {post.timestamp}</div>
                </div>
              </div>
              <button className="text-textMuted hover:text-white p-2">
                <MoreHorizontal className="w-5 h-5" />
              </button>
            </div>

            {/* Visual Content (Analysis Snippet) */}
            <div className="bg-black relative group cursor-pointer aspect-[16/9]">
              <div className="absolute inset-0 flex items-center justify-center z-10">
                 <div className="w-14 h-14 rounded-full bg-white/10 backdrop-blur-md border border-white/20 flex items-center justify-center group-hover:scale-110 transition-transform shadow-[0_0_30px_rgba(139,92,246,0.3)]">
                    <Play className="w-6 h-6 text-white ml-1 fill-white" />
                 </div>
              </div>
              
              {/* Data Overlay */}
              <div className="absolute bottom-4 left-4 right-4 flex justify-between items-end z-10">
                 <div className="bg-black/60 backdrop-blur px-3 py-1.5 rounded-lg text-xs font-mono text-primary border border-primary/30 flex items-center gap-2">
                    <div className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse"></div>
                    SPEED: +0.4s
                 </div>
              </div>

              {/* Fake visuals */}
              <img 
                 src="https://picsum.photos/seed/bjj_action/800/450" 
                 alt="Content"
                 className="w-full h-full object-cover opacity-60 transition-opacity group-hover:opacity-80" 
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent" />
            </div>

            {/* Caption */}
            <div className="px-5 pt-4 pb-2">
              <p className="text-sm text-gray-300 leading-relaxed font-normal">{post.caption}</p>
            </div>

            {/* Actions */}
            <div className="p-3 px-5 flex items-center justify-between border-t border-white/5 mt-2">
              <div className="flex items-center gap-6">
                 <button className="flex items-center gap-2 text-textMuted hover:text-accent transition-colors group">
                   <Heart className="w-5 h-5 group-hover:fill-current group-active:scale-90 transition-transform" />
                   <span className="text-xs font-mono font-bold">{post.likes}</span>
                 </button>
                 <button className="flex items-center gap-2 text-textMuted hover:text-white transition-colors">
                   <MessageCircle className="w-5 h-5" />
                   <span className="text-xs font-mono font-bold">{post.comments}</span>
                 </button>
              </div>
              <button className="text-textMuted hover:text-primary transition-colors">
                 <Share2 className="w-5 h-5" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};