import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Brain, Upload, ArrowLeft, Loader2, Image as ImageIcon, Target, Activity, Maximize } from 'lucide-react';
import { toast } from 'sonner';
import client from '@/lib/client';

export default function Segment() {
  const navigate = useNavigate();
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<any>(null);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    setUser({
      id: 'mock_user_123',
      email: 'mock_user@example.com',
      name: 'Mock User',
      role: 'admin'
    });
    setLoading(false);
  }, []);

  const handleFile = (f: File) => {
    if (!f.type.startsWith('image/')) {
      toast.error('Please upload an image file');
      return;
    }
    setFile(f);
    setResult(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  }, []);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = () => setDragActive(false);

  const handleAnalyze = async () => {
    if (!preview || !file) return;
    setAnalyzing(true);
    setProgress(0);
    setResult(null);

    // Simulate progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) { clearInterval(interval); return 90; }
        return prev + Math.random() * 15;
      });
    }, 300);

    try {
      const response = await client.apiCall.invoke({
        url: '/api/v1/segmentation/analyze',
        method: 'POST',
        data: {
          image: preview,
          filename: file.name,
        },
      });

      clearInterval(interval);
      setProgress(100);
      setResult(response.data);
      toast.success('Segmentation completed successfully!');
    } catch (err: any) {
      clearInterval(interval);
      setProgress(0);
      toast.error(err?.data?.detail || err?.message || 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const loadSample = (index: number) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx?.drawImage(img, 0, 0);
      const dataUrl = canvas.toDataURL('image/png');
      setPreview(dataUrl);
      setFile(new File([dataUrl], `sample-${index}.png`, { type: 'image/png' }));
      setResult(null);
    };
    img.src = `/assets/samples/image-${index}.png`;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
      </div>
    );
  }

  // Extract inference stats from the API response
  const stats = result?.inference_stats;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950">
      {/* Header */}
      <header className="border-b border-slate-800/50 backdrop-blur-sm bg-slate-950/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" className="text-slate-400 hover:text-white" onClick={() => navigate('/')}>
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold text-white">Tumor Segmentation</span>
          </div>
          <Button variant="ghost" className="text-slate-300 hover:text-white" onClick={() => navigate('/history')}>
            View History
          </Button>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Upload className="w-5 h-5 text-indigo-400" />
                  Upload MRI Scan
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div
                  className={`border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer ${
                    dragActive
                      ? 'border-indigo-500 bg-indigo-500/10'
                      : 'border-slate-700 hover:border-slate-600'
                  }`}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onClick={() => document.getElementById('file-input')?.click()}
                >
                  <input
                    id="file-input"
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                  />
                  {preview ? (
                    <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg" />
                  ) : (
                    <div className="space-y-4">
                      <div className="w-16 h-16 mx-auto rounded-full bg-slate-800 flex items-center justify-center">
                        <ImageIcon className="w-8 h-8 text-slate-500" />
                      </div>
                      <div>
                        <p className="text-slate-300 font-medium">Drop your MRI scan here</p>
                        <p className="text-slate-500 text-sm mt-1">or click to browse files</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Sample images */}
                <div className="mt-6">
                  <p className="text-sm text-slate-400 mb-3">Or try a sample image:</p>
                  <div className="grid grid-cols-4 gap-2">
                    {[1, 2, 3, 4].map((i) => (
                      <button
                        key={i}
                        onClick={() => loadSample(i)}
                        className="rounded-lg overflow-hidden border border-slate-700 hover:border-indigo-500 transition-all"
                      >
                        <img src={`/assets/samples/image-${i}.png`} alt={`Sample ${i}`} className="w-full h-16 object-cover" />
                      </button>
                    ))}
                  </div>
                </div>

                {/* Analyze Button */}
                <Button
                  className="w-full mt-6 bg-indigo-600 hover:bg-indigo-700 text-white py-6 text-lg"
                  disabled={!preview || analyzing}
                  onClick={handleAnalyze}
                >
                  {analyzing ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Brain className="w-5 h-5 mr-2" />
                      Run Segmentation
                    </>
                  )}
                </Button>

                {/* Progress */}
                {analyzing && (
                  <div className="mt-4 space-y-2">
                    <Progress value={progress} className="h-2" />
                    <p className="text-sm text-slate-400 text-center">
                      Processing with Enhanced U-Net model...
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {result ? (
              <>
                {/* Segmented Image */}
                <Card className="bg-slate-900/50 border-slate-800">
                  <CardHeader>
                    <CardTitle className="text-white">Segmentation Result</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-slate-400 mb-2 text-center">Overlay</p>
                        <img
                          src={result.overlay_image}
                          alt="Segmentation overlay"
                          className="w-full rounded-lg border border-slate-700"
                        />
                      </div>
                      <div>
                        <p className="text-sm text-slate-400 mb-2 text-center">Mask</p>
                        <img
                          src={result.mask_image}
                          alt="Segmentation mask"
                          className="w-full rounded-lg border border-slate-700"
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Inference Statistics — honest, prediction-derived */}
                <Card className="bg-slate-900/50 border-slate-800">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-2">
                      Inference Statistics
                      <span className="text-xs font-normal text-slate-500 ml-auto">
                        Model: {result.model_used || 'Unknown'}
                      </span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {/* Tumor Detection Badge */}
                    <div className="mb-5 flex items-center gap-3">
                      {stats?.tumor_detected ? (
                        <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold bg-rose-500/15 text-rose-400 border border-rose-500/20">
                          <Target className="w-4 h-4" />
                          Tumor Detected
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/20">
                          <Target className="w-4 h-4" />
                          No Tumor Detected
                        </span>
                      )}
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <InferenceStatCard
                        label="Tumor Coverage"
                        value={`${(stats?.tumor_coverage_percentage ?? 0).toFixed(2)}%`}
                        sublabel="of total image area"
                        icon={<Activity className="w-4 h-4 text-cyan-400" />}
                        color="cyan"
                      />
                      <InferenceStatCard
                        label="Mean Tumor Probability"
                        value={`${((stats?.mean_tumor_probability ?? 0) * 100).toFixed(1)}%`}
                        sublabel="avg. sigmoid in foreground"
                        icon={<Brain className="w-4 h-4 text-purple-400" />}
                        color="purple"
                      />
                      <InferenceStatCard
                        label="Max Tumor Probability"
                        value={`${((stats?.max_tumor_probability ?? 0) * 100).toFixed(1)}%`}
                        sublabel="peak sigmoid value"
                        icon={<Maximize className="w-4 h-4 text-indigo-400" />}
                        color="indigo"
                      />
                      <InferenceStatCard
                        label="Tumor Pixel Count"
                        value={(stats?.tumor_pixel_count ?? 0).toLocaleString()}
                        sublabel="pixels above threshold"
                        icon={<Target className="w-4 h-4 text-rose-400" />}
                        color="rose"
                      />
                    </div>

                    <p className="text-xs text-slate-600 mt-4 text-center">
                      Note: Dice and IoU require a ground-truth mask and are not shown for normal inference.
                      Use the Model Performance page for test-set evaluation metrics.
                    </p>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card className="bg-slate-900/50 border-slate-800 h-full flex items-center justify-center min-h-[400px]">
                <CardContent className="text-center py-16">
                  <div className="w-20 h-20 mx-auto rounded-full bg-slate-800/50 flex items-center justify-center mb-6">
                    <Brain className="w-10 h-10 text-slate-600" />
                  </div>
                  <h3 className="text-lg font-medium text-slate-400 mb-2">No Results Yet</h3>
                  <p className="text-sm text-slate-500">
                    Upload an MRI scan and run segmentation to see results here.
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function InferenceStatCard({
  label, value, sublabel, icon, color
}: {
  label: string; value: string; sublabel: string; icon: React.ReactNode; color: string;
}) {
  const textColorMap: Record<string, string> = {
    cyan: 'text-cyan-400',
    purple: 'text-purple-400',
    indigo: 'text-indigo-400',
    rose: 'text-rose-400',
  };

  return (
    <div className="bg-slate-800/50 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-xs text-slate-400">{label}</span>
      </div>
      <div className={`text-xl font-bold ${textColorMap[color] || 'text-white'}`}>{value}</div>
      <div className="text-[10px] text-slate-500 mt-0.5">{sublabel}</div>
    </div>
  );
}