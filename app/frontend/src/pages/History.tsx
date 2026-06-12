import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Brain, ArrowLeft, Loader2, Clock, FileImage } from 'lucide-react';
import client from '@/lib/client';

export default function History() {
  const navigate = useNavigate();
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [results, setResults] = useState<any[]>([]);

  useEffect(() => {
    setUser({
      id: 'mock_user_123',
      email: 'mock_user@example.com',
      name: 'Mock User',
      role: 'admin'
    });
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const response = await client.entities.segmentation_results.query({
        query: {},
        sort: '-created_at',
        limit: 50,
      });
      setResults(response.data?.items || []);
    } catch (err) {
      console.error('Failed to load history:', err);
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
            <span className="text-xl font-bold text-white">Segmentation History</span>
          </div>
          <Button className="bg-indigo-600 hover:bg-indigo-700 text-white" onClick={() => navigate('/segment')}>
            New Analysis
          </Button>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {results.length === 0 ? (
          <Card className="bg-slate-900/50 border-slate-800">
            <CardContent className="py-16 text-center">
              <div className="w-20 h-20 mx-auto rounded-full bg-slate-800/50 flex items-center justify-center mb-6">
                <Clock className="w-10 h-10 text-slate-600" />
              </div>
              <h3 className="text-lg font-medium text-slate-400 mb-2">No History Yet</h3>
              <p className="text-sm text-slate-500 mb-6">
                Run your first segmentation analysis to see results here.
              </p>
              <Button className="bg-indigo-600 hover:bg-indigo-700 text-white" onClick={() => navigate('/segment')}>
                Start Analysis
              </Button>
            </CardContent>
          </Card>
        ) : (
          <>
            {/* Summary Cards */}
            <div className="grid md:grid-cols-4 gap-4 mb-8">
              <SummaryCard
                label="Total Analyses"
                value={results.length.toString()}
                icon={<FileImage className="w-5 h-5 text-indigo-400" />}
              />
              <SummaryCard
                label="Avg Dice Score"
                value={`${(results.reduce((a, r) => a + (r.dice_coefficient || 0), 0) / results.length * 100).toFixed(1)}%`}
                icon={<Brain className="w-5 h-5 text-purple-400" />}
              />
              <SummaryCard
                label="Avg Accuracy"
                value={`${(results.reduce((a, r) => a + (r.accuracy || 0), 0) / results.length * 100).toFixed(1)}%`}
                icon={<Brain className="w-5 h-5 text-cyan-400" />}
              />
              <SummaryCard
                label="Avg IoU"
                value={`${(results.reduce((a, r) => a + (r.iou_score || 0), 0) / results.length * 100).toFixed(1)}%`}
                icon={<Brain className="w-5 h-5 text-green-400" />}
              />
            </div>

            {/* Results Table */}
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white">Analysis Results</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-slate-800">
                      <TableHead className="text-slate-400">Filename</TableHead>
                      <TableHead className="text-slate-400">Dice Coefficient</TableHead>
                      <TableHead className="text-slate-400">Accuracy</TableHead>
                      <TableHead className="text-slate-400">IoU Score</TableHead>
                      <TableHead className="text-slate-400">Date</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {results.map((result, idx) => (
                      <TableRow key={result.id || idx} className="border-slate-800">
                        <TableCell className="text-slate-300 font-medium">
                          {result.filename || 'Unknown'}
                        </TableCell>
                        <TableCell>
                          <span className="text-indigo-400 font-semibold">
                            {((result.dice_coefficient || 0) * 100).toFixed(1)}%
                          </span>
                        </TableCell>
                        <TableCell>
                          <span className="text-purple-400 font-semibold">
                            {((result.accuracy || 0) * 100).toFixed(1)}%
                          </span>
                        </TableCell>
                        <TableCell>
                          <span className="text-cyan-400 font-semibold">
                            {((result.iou_score || 0) * 100).toFixed(1)}%
                          </span>
                        </TableCell>
                        <TableCell className="text-slate-500 text-sm">
                          {result.created_at ? new Date(result.created_at).toLocaleDateString() : '-'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}

function SummaryCard({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) {
  return (
    <Card className="bg-slate-900/50 border-slate-800">
      <CardContent className="p-5">
        <div className="flex items-center justify-between mb-2">
          {icon}
        </div>
        <div className="text-2xl font-bold text-white">{value}</div>
        <div className="text-sm text-slate-400">{label}</div>
      </CardContent>
    </Card>
  );
}