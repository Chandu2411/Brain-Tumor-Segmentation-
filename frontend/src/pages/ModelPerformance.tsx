import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Brain, ArrowLeft, Loader2, BarChart3, Activity, Target, Shield, Eye, Crosshair } from 'lucide-react';

interface ModelMetrics {
  trained: boolean;
  model_name: string;
  architecture_changed: boolean;
  test_samples: number;
  positive_masks: number;
  empty_masks: number;
  dice_mean: number;
  dice_std: number;
  iou_mean: number;
  iou_std: number;
  precision: number;
  recall: number;
  specificity: number;
  pixel_accuracy: number;
  prediction_threshold: number;
  training_history?: any;
  message?: string;
}

export default function ModelPerformance() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    try {
      const response = await fetch('/api/model-performance');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setMetrics(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to load model performance data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
      </div>
    );
  }

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
            <span className="text-xl font-bold text-white">Model Performance</span>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" className="text-slate-300 hover:text-white" onClick={() => navigate('/segment')}>
              Segmentation
            </Button>
            <Button variant="ghost" className="text-slate-300 hover:text-white" onClick={() => navigate('/history')}>
              History
            </Button>
          </div>
        </div>
      </header>

      <div className="max-w-5xl mx-auto px-6 py-8">
        {error ? (
          <Card className="bg-slate-900/50 border-slate-800">
            <CardContent className="py-12 text-center">
              <p className="text-rose-400 font-medium mb-2">Failed to Load Metrics</p>
              <p className="text-sm text-slate-500">{error}</p>
            </CardContent>
          </Card>
        ) : metrics && !metrics.trained ? (
          /* Not yet trained */
          <Card className="bg-slate-900/50 border-slate-800">
            <CardContent className="py-16 text-center">
              <div className="w-20 h-20 mx-auto rounded-full bg-amber-500/10 flex items-center justify-center mb-6">
                <BarChart3 className="w-10 h-10 text-amber-400" />
              </div>
              <h3 className="text-lg font-medium text-white mb-2">Training Pending</h3>
              <p className="text-sm text-slate-400 max-w-md mx-auto mb-4">
                {metrics.message || 'Run the training notebook to generate real evaluation metrics on the held-out test set.'}
              </p>
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-800/50 text-slate-400 text-sm">
                <Brain className="w-4 h-4" />
                Model: {metrics.model_name}
              </div>
            </CardContent>
          </Card>
        ) : metrics ? (
          /* Real metrics available */
          <>
            {/* Model Info */}
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                  <Brain className="w-4 h-4 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-white">{metrics.model_name}</h2>
                {!metrics.architecture_changed && (
                  <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/20">
                    Original Architecture
                  </span>
                )}
              </div>
              <p className="text-sm text-slate-400 ml-11">
                Evaluated on {metrics.test_samples} test samples ({metrics.positive_masks} with tumors, {metrics.empty_masks} empty masks)
              </p>
            </div>

            {/* Primary Metrics Grid */}
            <div className="grid md:grid-cols-3 gap-4 mb-6">
              <MetricCard
                label="Test Dice"
                value={`${(metrics.dice_mean * 100).toFixed(2)}%`}
                std={`± ${(metrics.dice_std * 100).toFixed(2)}%`}
                icon={<Activity className="w-5 h-5 text-indigo-400" />}
                color="indigo"
              />
              <MetricCard
                label="Test IoU"
                value={`${(metrics.iou_mean * 100).toFixed(2)}%`}
                std={`± ${(metrics.iou_std * 100).toFixed(2)}%`}
                icon={<Target className="w-5 h-5 text-purple-400" />}
                color="purple"
              />
              <MetricCard
                label="Pixel Accuracy"
                value={`${(metrics.pixel_accuracy * 100).toFixed(2)}%`}
                icon={<Crosshair className="w-5 h-5 text-cyan-400" />}
                color="cyan"
              />
            </div>

            <div className="grid md:grid-cols-3 gap-4 mb-8">
              <MetricCard
                label="Precision"
                value={`${(metrics.precision * 100).toFixed(2)}%`}
                icon={<Shield className="w-5 h-5 text-emerald-400" />}
                color="emerald"
              />
              <MetricCard
                label="Recall / Sensitivity"
                value={`${(metrics.recall * 100).toFixed(2)}%`}
                icon={<Eye className="w-5 h-5 text-amber-400" />}
                color="amber"
              />
              <MetricCard
                label="Specificity"
                value={`${(metrics.specificity * 100).toFixed(2)}%`}
                icon={<Shield className="w-5 h-5 text-rose-400" />}
                color="rose"
              />
            </div>

            {/* Info Card */}
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white text-sm">About These Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-slate-400 leading-relaxed">
                  These metrics were computed by evaluating the {metrics.model_name} on a held-out test set
                  with ground-truth masks. They reflect the model's actual segmentation performance.
                  Prediction threshold: <span className="text-white font-mono">{metrics.prediction_threshold}</span>.
                </p>
                <p className="text-xs text-slate-500 mt-3">
                  Dice and IoU require both a predicted mask and a ground-truth mask. During normal
                  inference (when a user uploads an MRI without a ground-truth mask), only prediction-derived
                  statistics like tumor coverage and probability are shown.
                </p>
              </CardContent>
            </Card>
          </>
        ) : null}
      </div>
    </div>
  );
}

function MetricCard({
  label, value, std, icon, color
}: {
  label: string; value: string; std?: string; icon: React.ReactNode; color: string;
}) {
  const textColorMap: Record<string, string> = {
    indigo: 'text-indigo-400',
    purple: 'text-purple-400',
    cyan: 'text-cyan-400',
    emerald: 'text-emerald-400',
    amber: 'text-amber-400',
    rose: 'text-rose-400',
  };
  const barColorMap: Record<string, string> = {
    indigo: 'from-indigo-500 to-indigo-600',
    purple: 'from-purple-500 to-purple-600',
    cyan: 'from-cyan-500 to-cyan-600',
    emerald: 'from-emerald-500 to-emerald-600',
    amber: 'from-amber-500 to-amber-600',
    rose: 'from-rose-500 to-rose-600',
  };

  const pctNum = parseFloat(value);
  const barWidth = isNaN(pctNum) ? 0 : Math.min(pctNum, 100);

  return (
    <Card className="bg-slate-900/50 border-slate-800">
      <CardContent className="p-5">
        <div className="flex items-center gap-2 mb-3">
          {icon}
          <span className="text-xs text-slate-400">{label}</span>
        </div>
        <div className={`text-2xl font-bold ${textColorMap[color] || 'text-white'}`}>
          {value}
        </div>
        {std && <div className="text-xs text-slate-500 mt-0.5">{std}</div>}
        <div className="mt-3 h-1.5 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${barColorMap[color] || 'from-indigo-500 to-indigo-600'}`}
            style={{ width: `${barWidth}%` }}
          />
        </div>
      </CardContent>
    </Card>
  );
}
