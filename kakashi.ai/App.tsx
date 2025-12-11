import React, { useState } from 'react';
import { DesktopSidebar, MobileNavigation } from './components/Sidebar';
import { Dashboard } from './components/Dashboard';
import { AnalysisView } from './components/AnalysisView';
import { SocialFeed } from './components/SocialFeed';
import { ProfileView } from './components/ProfileView';
import { UploadModal } from './components/UploadModal';
import { MOCK_ANALYSIS } from './constants';

export type ViewState = 'dashboard' | 'analysis' | 'social' | 'profile';

export default function App() {
  const [currentView, setCurrentView] = useState<ViewState>('dashboard');
  const [selectedAnalysisId, setSelectedAnalysisId] = useState<string | null>(null);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

  const handleNavigate = (view: ViewState, id?: string) => {
    setCurrentView(view);
    if (id) setSelectedAnalysisId(id);
  };

  const renderView = () => {
    switch (currentView) {
      case 'dashboard':
        return (
          <Dashboard 
            onNavigate={handleNavigate} 
            onUpload={() => setIsUploadModalOpen(true)} 
          />
        );
      case 'analysis':
        return (
          <AnalysisView 
            analysis={MOCK_ANALYSIS} 
            onBack={() => handleNavigate('dashboard')} 
          />
        );
      case 'social':
        return <SocialFeed onNavigate={handleNavigate} />;
      case 'profile':
        return <ProfileView />;
      default:
        return <Dashboard onNavigate={handleNavigate} onUpload={() => setIsUploadModalOpen(true)} />;
    }
  };

  return (
    <div className="flex h-screen w-full bg-background text-textMain overflow-hidden selection:bg-primary/30">
      {/* Desktop Navigation */}
      <DesktopSidebar currentView={currentView} onNavigate={handleNavigate} />

      {/* Main Content */}
      <main className="flex-1 h-full overflow-hidden relative flex flex-col">
        {/* Mobile Top Brand Bar (only visible on small screens when not in analysis) */}
        {currentView !== 'analysis' && (
          <div className="md:hidden h-14 border-b border-white/5 flex items-center justify-between px-4 bg-background/50 backdrop-blur sticky top-0 z-10">
            <span className="font-bold text-lg tracking-tighter flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
              KAKASHI
            </span>
            <div className="w-8 h-8 rounded-full bg-white/10" />
          </div>
        )}

        <div className="flex-1 overflow-y-auto pb-20 md:pb-0 scroll-smooth">
          {renderView()}
        </div>
      </main>

      {/* Mobile Navigation */}
      <MobileNavigation currentView={currentView} onNavigate={handleNavigate} />

      {/* Upload Modal Overlay */}
      {isUploadModalOpen && (
        <UploadModal onClose={() => setIsUploadModalOpen(false)} />
      )}
    </div>
  );
}