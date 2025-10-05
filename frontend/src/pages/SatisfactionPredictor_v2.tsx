import Navigation from "@/components/Navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Sparkles, TrendingUp, BarChart3, Zap } from "lucide-react";

const SatisfactionPredictorV2 = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="container mx-auto px-6 pt-24 pb-16">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
              <Sparkles className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-5xl font-bold tracking-tight">
              Satisfaction Predictor V2
            </h1>
            <p className="text-xl text-muted-foreground">
              Enhanced prediction model with advanced ML techniques
            </p>
          </div>

          <Card className="border-2 border-primary/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-primary" />
                Model Overview
              </CardTitle>
              <CardDescription>
                Version 2.0 - Released Q3 2024
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="font-semibold mb-2">Improvements</h3>
                <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                  <li>Gradient boosting ensemble model with XGBoost</li>
                  <li>Feature engineering including temporal and geographical data</li>
                  <li>Accuracy: 86% on test dataset (+14% improvement)</li>
                  <li>Processing time: ~35ms per prediction (30% faster)</li>
                  <li>Real-time prediction API with sub-second response</li>
                </ul>
              </div>
            </CardContent>
          </Card>

          <Card className="border-2 border-primary/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-primary" />
                Key Metrics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">86%</div>
                  <div className="text-sm text-muted-foreground mt-1">Accuracy</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">0.83</div>
                  <div className="text-sm text-muted-foreground mt-1">F1 Score</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">35ms</div>
                  <div className="text-sm text-muted-foreground mt-1">Avg Response</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-2 border-primary/30 bg-gradient-to-br from-primary/5 to-transparent">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-primary" />
                What's New in V2
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                <div className="flex gap-3">
                  <div className="w-2 h-2 rounded-full bg-primary mt-2 flex-shrink-0" />
                  <div>
                    <h4 className="font-medium">Advanced Feature Engineering</h4>
                    <p className="text-sm text-muted-foreground">
                      Incorporating delivery time predictions, seasonal patterns, and customer behavior history
                    </p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <div className="w-2 h-2 rounded-full bg-primary mt-2 flex-shrink-0" />
                  <div>
                    <h4 className="font-medium">Ensemble Learning</h4>
                    <p className="text-sm text-muted-foreground">
                      Combining multiple models for more robust predictions
                    </p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <div className="w-2 h-2 rounded-full bg-primary mt-2 flex-shrink-0" />
                  <div>
                    <h4 className="font-medium">Real-time Inference</h4>
                    <p className="text-sm text-muted-foreground">
                      Optimized for production use with low latency predictions
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="p-6 rounded-lg border-2 border-primary/20 bg-primary/5">
            <p className="text-sm text-center text-muted-foreground">
              This is a placeholder page. Add your prediction interface and model integration here.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
};

export default SatisfactionPredictorV2;
