import Navigation from "@/components/Navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, TrendingUp, BarChart3 } from "lucide-react";

const SatisfactionPredictorV1 = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="container mx-auto px-6 pt-24 pb-16">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
              <Brain className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-5xl font-bold tracking-tight">
              Satisfaction Predictor V1
            </h1>
            <p className="text-xl text-muted-foreground">
              First generation customer satisfaction prediction model
            </p>
          </div>

          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-primary" />
                Model Overview
              </CardTitle>
              <CardDescription>
                Version 1.0 - Released Q1 2024
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="font-semibold mb-2">Features</h3>
                <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                  <li>Baseline satisfaction prediction using order data</li>
                  <li>Linear regression model with key order features</li>
                  <li>Accuracy: 72% on test dataset</li>
                  <li>Processing time: ~50ms per prediction</li>
                </ul>
              </div>
            </CardContent>
          </Card>

          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-primary" />
                Key Metrics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">72%</div>
                  <div className="text-sm text-muted-foreground mt-1">Accuracy</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">0.68</div>
                  <div className="text-sm text-muted-foreground mt-1">F1 Score</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">50ms</div>
                  <div className="text-sm text-muted-foreground mt-1">Avg Response</div>
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

export default SatisfactionPredictorV1;
