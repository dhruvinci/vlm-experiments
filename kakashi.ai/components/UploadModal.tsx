import React, { useState } from 'react';
import { X, UploadCloud, FileVideo, CheckCircle2, Scan } from 'lucide-react';

interface UploadModalProps {
  onClose: () => void;
}

export const UploadModal: React.FC<UploadModalProps> = ({ onClose }) => {
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [isUploading, setIsUploading] = useState(false);

  // Mock Upload Process
  const handleUpload = () => {
    setIsUploading(true);
    setTimeout(() => {
      setStep(3);
      setIsUploading(false);
    }, 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-end md:items-center justify-center p-0 md:p-4 bg-black/90 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="w-full max-w-lg bg-[#0c0c0e] border-t md:border border-white/10 rounded-t-3xl md:rounded-3xl shadow-[0_0_50px_rgba(0,0,0,0.8)] overflow-hidden relative">
        {/* Progress Bar (Fake) */}
        {isUploading && (
           <div className="absolute top-0 left-0 h-1 bg-primary w-full animate-[scan_2s_ease-in-out_infinite]"></div>
        )}

        {/* Close Button */}
        <button onClick={onClose} className="absolute top-4 right-4 p-2 text-textMuted hover:text-white hover:bg-white/10 rounded-full transition-colors z-10">
          <X className="w-5 h-5" />
        </button>

        <div className="p-8">
          {step === 1 && (
            <div className="text-center">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-6 text-primary shadow-[0_0_20px_rgba(139,92,246,0.2)]">
                <UploadCloud className="w-8 h-8" />
              </div>
              <h2 className="text-2xl font-bold text-white mb-2 tracking-tight">Upload Footage</h2>
              <p className="text-textMuted mb-8 text-sm">Drag and drop your raw video files here. <br/>Supports MP4, MOV up to 4K.</p>
              
              <div className="border-2 border-dashed border-white/10 rounded-2xl p-10 hover:border-primary/50 hover:bg-white/5 transition-all cursor-pointer group mb-6 relative overflow-hidden">
                <div className="absolute inset-0 bg-grid opacity-10 pointer-events-none"></div>
                <FileVideo className="w-10 h-10 text-textMuted mx-auto group-hover:scale-110 transition-transform mb-2 group-hover:text-primary" />
                <span className="text-xs font-mono font-bold text-textMuted group-hover:text-white uppercase tracking-wider">Select File</span>
              </div>

              <button 
                onClick={() => setStep(2)}
                className="w-full py-3.5 bg-white text-black font-bold rounded-xl hover:bg-gray-200 transition-colors shadow-lg active:scale-95 duration-200"
              >
                Continue
              </button>
            </div>
          )}

          {step === 2 && (
            <div>
              <h2 className="text-xl font-bold text-white mb-6">Classification</h2>
              
              <div className="space-y-6">
                <div>
                   <label className="text-[10px] font-mono uppercase tracking-widest text-textMuted mb-3 block">Discipline</label>
                   <div className="grid grid-cols-2 gap-4">
                     {['Gi', 'No-Gi'].map(t => (
                       <button key={t} className="p-4 rounded-xl border border-white/10 hover:border-primary hover:bg-primary/5 text-left transition-all focus:border-primary focus:ring-1 focus:ring-primary group">
                         <span className="font-bold text-white group-hover:text-primary transition-colors">{t}</span>
                       </button>
                     ))}
                   </div>
                </div>

                <div>
                   <label className="text-[10px] font-mono uppercase tracking-widest text-textMuted mb-3 block">Context</label>
                   <div className="grid grid-cols-2 gap-4">
                     {['Training', 'Competition'].map(t => (
                       <button key={t} className="p-4 rounded-xl border border-white/10 hover:border-primary hover:bg-primary/5 text-left transition-all focus:border-primary focus:ring-1 focus:ring-primary group">
                         <span className="font-bold text-white group-hover:text-primary transition-colors">{t}</span>
                       </button>
                     ))}
                   </div>
                </div>
              </div>

              <button 
                onClick={handleUpload}
                disabled={isUploading}
                className="w-full mt-8 py-3.5 bg-primary hover:bg-primaryDark text-white font-bold rounded-xl transition-all flex items-center justify-center shadow-[0_0_20px_rgba(139,92,246,0.4)] disabled:opacity-70 disabled:cursor-not-allowed"
              >
                {isUploading ? (
                  <span className="flex items-center gap-2 font-mono text-sm animate-pulse">
                     <Scan className="w-4 h-4" /> ENCRYPTING...
                  </span>
                ) : (
                  'Initialize Analysis'
                )}
              </button>
            </div>
          )}

          {step === 3 && (
            <div className="text-center py-8">
              <div className="w-20 h-20 bg-secondary/10 rounded-full flex items-center justify-center mx-auto mb-6 text-secondary animate-in zoom-in duration-300 shadow-[0_0_30px_rgba(6,182,212,0.3)]">
                <CheckCircle2 className="w-10 h-10" />
              </div>
              <h2 className="text-2xl font-bold text-white mb-2">Processing Queued</h2>
              <p className="text-textMuted mb-8 text-sm">Your video is being processed by our neural network. <br/>Est. time: <span className="text-secondary font-mono">1m 30s</span></p>
              
              <button 
                onClick={onClose}
                className="px-8 py-3 bg-white/5 hover:bg-white/10 border border-white/10 text-white font-bold rounded-xl transition-colors w-full"
              >
                Return to Hub
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};