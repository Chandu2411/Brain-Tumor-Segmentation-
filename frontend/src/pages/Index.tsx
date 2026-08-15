import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Brain, Upload, BarChart3, Shield } from 'lucide-react';

export default function Index() {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/segment');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950">
      {/* Header */}
      <header className="border-b border-slate-800/50 backdrop-blur-sm bg-slate-950/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold text-white">NeuroSeg AI</span>
          </div>
          <nav className="flex items-center gap-4">
            <Button variant="ghost" className="text-slate-300 hover:text-white" onClick={() => navigate('/segment')}>
              Segmentation
            </Button>
            <Button variant="ghost" className="text-slate-300 hover:text-white" onClick={() => navigate('/history')}>
              History
            </Button>
            <Button variant="ghost" className="text-slate-300 hover:text-white" onClick={() => navigate('/performance')}>
              Model Performance
            </Button>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-6 py-24 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-sm mb-8">
          <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
          Enhanced U-Net Architecture
        </div>
        <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
          Brain Tumor<br />
          <span className="bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
            Segmentation
          </span>
        </h1>
        <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-12">
          Advanced deep learning-powered MRI analysis for precise brain tumor detection and segmentation using our Enhanced U-Net model.
        </p>
        <div className="flex items-center justify-center gap-4">
          <Button
            size="lg"
            className="bg-indigo-600 hover:bg-indigo-700 text-white px-8 py-6 text-lg rounded-xl"
            onClick={handleGetStarted}
          >
            <Upload className="w-5 h-5 mr-2" />
            Start Segmentation
          </Button>
        </div>
      </section>

      {/* Features */}
      <section className="max-w-7xl mx-auto px-6 pb-24">
        <div className="grid md:grid-cols-3 gap-6">
          <Card className="bg-slate-900/50 border-slate-800 hover:border-indigo-500/50 transition-all duration-300">
            <CardContent className="p-8">
              <div className="w-12 h-12 rounded-xl bg-indigo-500/10 flex items-center justify-center mb-4">
                <Brain className="w-6 h-6 text-indigo-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Enhanced U-Net Model</h3>
              <p className="text-slate-400 text-sm">
                Multi-scale feature extraction with encoder-decoder architecture and skip connections for precise tumor boundary detection.
              </p>
            </CardContent>
          </Card>
          <Card className="bg-slate-900/50 border-slate-800 hover:border-purple-500/50 transition-all duration-300">
            <CardContent className="p-8">
              <div className="w-12 h-12 rounded-xl bg-purple-500/10 flex items-center justify-center mb-4">
                <BarChart3 className="w-6 h-6 text-purple-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Inference Analytics</h3>
              <p className="text-slate-400 text-sm">
                Real-time tumor detection, coverage analysis, and probability mapping for each uploaded MRI scan. Model evaluation metrics available on the Performance page.
              </p>
            </CardContent>
          </Card>
          <Card className="bg-slate-900/50 border-slate-800 hover:border-cyan-500/50 transition-all duration-300">
            <CardContent className="p-8">
              <div className="w-12 h-12 rounded-xl bg-cyan-500/10 flex items-center justify-center mb-4">
                <Shield className="w-6 h-6 text-cyan-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Secure & Private</h3>
              <p className="text-slate-400 text-sm">
                All medical images are processed securely with user authentication and encrypted storage for patient data protection.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Sample Images */}
      <section className="max-w-7xl mx-auto px-6 pb-24">
        <h2 className="text-3xl font-bold text-white text-center mb-12">Sample MRI Scans</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="rounded-xl overflow-hidden border border-slate-800 hover:border-indigo-500/50 transition-all duration-300">
              <img
                src={`/assets/samples/image-${i}.png`}
                alt={`MRI Sample ${i}`}
                className="w-full h-48 object-cover"
              />
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}