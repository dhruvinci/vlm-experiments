import React from 'react';
import { LayoutDashboard, Globe, User, LogOut, Hexagon, Menu } from 'lucide-react';
import { ViewState } from '../App';

interface NavigationProps {
  currentView: ViewState;
  onNavigate: (view: ViewState) => void;
}

const NAV_ITEMS = [
  { id: 'dashboard', icon: LayoutDashboard, label: 'Hub' },
  { id: 'social', icon: Globe, label: 'Feed' },
  { id: 'profile', icon: User, label: 'Profile' },
];

export const DesktopSidebar: React.FC<NavigationProps> = ({ currentView, onNavigate }) => {
  return (
    <div className="hidden md:flex w-20 lg:w-64 h-full border-r border-white/5 flex-col justify-between bg-black/40 backdrop-blur-xl z-20">
      <div>
        {/* Brand */}
        <div className="h-16 lg:h-20 flex items-center px-0 lg:px-6 justify-center lg:justify-start border-b border-white/5">
          <Hexagon className="w-6 h-6 lg:w-7 lg:h-7 text-primary text-glow" strokeWidth={2} />
          <span className="ml-3 font-bold text-lg tracking-tighter hidden lg:block text-white">
            KAKASHI<span className="text-primary">.AI</span>
          </span>
        </div>

        {/* Navigation */}
        <nav className="p-3 lg:p-4 space-y-2">
          {NAV_ITEMS.map((item) => {
            const isActive = currentView === item.id;
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id as ViewState)}
                className={`w-full flex items-center justify-center lg:justify-start p-3 rounded-xl transition-all duration-300 group relative overflow-hidden ${
                  isActive 
                    ? 'bg-white/5 text-white' 
                    : 'text-textMuted hover:text-white hover:bg-white/5'
                }`}
              >
                {isActive && (
                  <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary shadow-[0_0_10px_rgba(139,92,246,0.8)]" />
                )}
                <Icon className={`w-6 h-6 ${isActive ? 'text-primary' : ''}`} strokeWidth={isActive ? 2.5 : 1.5} />
                <span className={`ml-3 font-medium hidden lg:block ${isActive ? 'text-white' : ''}`}>
                  {item.label}
                </span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer */}
      <div className="p-3 lg:p-4 border-t border-white/5">
        <button className="w-full flex items-center justify-center lg:justify-start p-3 rounded-xl text-textMuted hover:text-white hover:bg-white/5 transition-colors">
          <LogOut className="w-5 h-5" />
          <span className="ml-3 font-medium hidden lg:block text-sm">Logout</span>
        </button>
      </div>
    </div>
  );
};

export const MobileNavigation: React.FC<NavigationProps> = ({ currentView, onNavigate }) => {
  return (
    <div className="md:hidden fixed bottom-0 left-0 right-0 h-16 bg-[#09090b]/90 backdrop-blur-xl border-t border-white/10 z-50 flex items-center justify-around px-2 pb-safe">
      {NAV_ITEMS.map((item) => {
        const isActive = currentView === item.id;
        const Icon = item.icon;
        return (
          <button
            key={item.id}
            onClick={() => onNavigate(item.id as ViewState)}
            className={`flex flex-col items-center justify-center w-16 h-full space-y-1 ${
              isActive ? 'text-primary' : 'text-textMuted'
            }`}
          >
            <Icon className="w-6 h-6" strokeWidth={isActive ? 2.5 : 1.5} />
            <span className="text-[10px] font-medium">{item.label}</span>
          </button>
        );
      })}
    </div>
  );
};