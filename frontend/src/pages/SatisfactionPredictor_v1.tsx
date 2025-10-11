import { useState } from "react";
import Navigation from "@/components/Navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Brain, TrendingUp, BarChart3, Sparkles, LoaderCircle, AlertTriangle, ThumbsUp, ThumbsDown, ClipboardList, MessageSquare, TrendingDown } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

// --- Type for API response ---
interface PredictionResult {
  is_satisfied_prediction: number;
  satisfaction_probability: number;
}

interface SentimentResult {
  original_text: string;
  cleaned_text: string;
  sentiment: "Positive" | "Negative";
  sentiment_score: number;
  prediction: 0 | 1;
}

// --- Hardcoded classification report data for V1 model ---
const v1Report = {
  "Not Satisfied (0)": { precision: 0.45, recall: 0.40, "f1-score": 0.42, support: 4984 },
  "Satisfied (1)": { precision: 0.82, recall: 0.85, "f1-score": 0.84, support: 16522 },
  "macro avg": { precision: 0.64, recall: 0.63, "f1-score": 0.63, support: 21506 },
  "weighted avg": { precision: 0.74, recall: 0.75, "f1-score": 0.74, support: 21506 },
};

const SatisfactionPredictorV1 = () => {
  const { toast } = useToast();
  
  // --- State management for form inputs ---
  const [price, setPrice] = useState(129.90);
  const [freightValue, setFreightValue] = useState(15.50);
  const [deliveryTime, setDeliveryTime] = useState(8);
  const [deliveryDelta, setDeliveryDelta] = useState(10);

  // --- State management for the API call ---
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  // --- State management for sentiment analysis ---
  const [reviewText, setReviewText] = useState("");
  const [isSentimentLoading, setIsSentimentLoading] = useState(false);
  const [sentimentResult, setSentimentResult] = useState<SentimentResult | null>(null);
  const [sentimentError, setSentimentError] = useState<string | null>(null);

  // --- API call function ---
  const handlePrediction = async () => {
    setResult(null);
    setError(null);
    setIsLoading(true);

    const requestBody = {
      price: price,
      freight_value: freightValue,
      delivery_time_days: deliveryTime,
      estimated_vs_actual_delivery: deliveryDelta,
    };

    try {
      const response = await fetch('http://localhost:8000/satisfaction/v1/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Network response was not ok');
      }
      
      const data: PredictionResult = await response.json();
      setResult(data);

    } catch (error: any) {
      setError(error.message || "An unknown error occurred. Is the backend running?");
    } finally {
      setIsLoading(false);
    }
  };

  // --- Sentiment analysis API call ---
  const handleSentimentAnalysis = async () => {
    if (!reviewText.trim()) {
      toast({
        title: "Input Required",
        description: "Please enter a review to analyze.",
        variant: "destructive"
      });
      return;
    }

    setIsSentimentLoading(true);
    setSentimentError(null);
    setSentimentResult(null);

    try {
      const response = await fetch( "http://127.0.0.1:8000/review/v1/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ review_text: reviewText }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }

      const data: SentimentResult = await response.json();
      setSentimentResult(data);
      toast({
        title: "Analysis Complete",
        description: `Sentiment detected: ${data.sentiment}`,
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "An unexpected error occurred";
      setSentimentError(errorMessage);
      toast({
        title: "Analysis Failed",
        description: "Sorry, an error occurred. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsSentimentLoading(false);
    }
  };

  const exampleReviews = [
    {
      label: "Positive Example",
      text: "O produto é maravilhoso e a entrega foi super rápida! Recomendo muito, qualidade excelente e atendimento perfeito."
    },
    {
      label: "Negative Example",
      text: "Que produto horrível, veio quebrado e demorou muito. Péssima experiência, não compro mais."
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="container mx-auto px-6 pt-24 pb-16">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* --- Hero Section --- */}
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

          {/* --- Model Overview Card --- */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-primary" />
                Model Overview
              </CardTitle>
              <CardDescription>
                Version 1.0.0 
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                <li>Baseline satisfaction prediction using order data</li>
                <li>LightGBM model with 4 key order features and SMOTE</li>
                <li>Accuracy: 74.66% on test dataset</li>
                <li>Processing time: ~50ms per prediction</li>
              </ul>
            </CardContent>
          </Card>

          {/* --- Key Metrics Card --- */}
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
                  <div className="text-3xl font-bold text-primary">74.66%</div>
                  <div className="text-sm text-muted-foreground mt-1">Accuracy</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">0.74</div>
                  <div className="text-sm text-muted-foreground mt-1">Weighted F1 Score</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">~50ms</div>
                  <div className="text-sm text-muted-foreground mt-1">Avg Response</div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* --- Model Profile Card --- */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="w-5 h-5 text-primary" />
                Model Profile
              </CardTitle>
              <CardDescription>Strengths and weaknesses of the V1 model.</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <h3 className="font-semibold flex items-center text-green-500"><ThumbsUp className="w-4 h-4 mr-2" />Advantages</h3>
                <ul className="list-disc list-inside space-y-1 text-muted-foreground text-sm">
                  <li>Fast & Lightweight: Rapid predictions suitable for real-time use.</li>
                  <li>Good Baseline: Establishes a solid performance benchmark.</li>
                  <li>Handles Imbalance: The SMOTE pipeline correctly addresses data imbalance.</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h3 className="font-semibold flex items-center text-red-500"><ThumbsDown className="w-4 h-4 mr-2" />Disadvantages</h3>
                <ul className="list-disc list-inside space-y-1 text-muted-foreground text-sm">
                  <li>Low Recall: Fails to identify the majority (60%) of dissatisfied customers.</li>
                  <li>Limited Features: Model only uses 4 features, likely missing key signals.</li>
                  <li>Not Tuned: Uses default parameters, which are not optimized.</li>
                </ul>
              </div>
            </CardContent>
          </Card>

          {/* --- Classification Report Card --- */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><ClipboardList className="w-5 h-5 text-primary" />Classification Report</CardTitle>
              <CardDescription>A detailed breakdown of the model's performance on the test set.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                  <thead className="text-xs text-muted-foreground uppercase bg-primary/5">
                    <tr>
                      <th scope="col" className="px-4 py-2 rounded-l-lg">Class</th>
                      <th scope="col" className="px-4 py-2 text-center">Precision</th>
                      <th scope="col" className="px-4 py-2 text-center">Recall</th>
                      <th scope="col" className="px-4 py-2 text-center">F1-Score</th>
                      <th scope="col" className="px-4 py-2 rounded-r-lg text-center">Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(v1Report).map(([key, value]) => (
                      <tr key={key} className="border-b border-primary/10">
                        <td className="px-4 py-2 font-semibold capitalize">{key}</td>
                        <td className="px-4 py-2 text-center">{value.precision.toFixed(2)}</td>
                        <td className="px-4 py-2 text-center">{value.recall.toFixed(2)}</td>
                        <td className="px-4 py-2 text-center">{value["f1-score"].toFixed(2)}</td>
                        <td className="px-4 py-2 text-center text-muted-foreground">{value.support.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* --- Live Prediction Interface Card --- */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-primary" />
                Live Prediction Interface
              </CardTitle>
              <CardDescription>
                Input order details below to get a live satisfaction prediction from the V1 model.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-8 items-start">
                {/* Left Column: Form */}
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="price">Price ($)</Label>
                    <p className="text-sm text-muted-foreground mt-1">The total price of the item(s) in the order.</p>
                    <Input id="price" type="number" value={price} onChange={e => setPrice(Number(e.target.value))} placeholder="e.g., 129.90" />
                  </div>
                  <div>
                    <Label htmlFor="freight">Freight Value ($)</Label>
                    <p className="text-sm text-muted-foreground mt-1">The shipping cost for the order.</p>
                    <Input id="freight" type="number" value={freightValue} onChange={e => setFreightValue(Number(e.target.value))} placeholder="e.g., 15.50" />
                  </div>
                  <div>
                    <Label htmlFor="delivery-time">Delivery Time (days)</Label>
                    <p className="text-sm text-muted-foreground mt-1">The actual number of days it took to be delivered.</p>
                    <Input id="delivery-time" type="number" value={deliveryTime} onChange={e => setDeliveryTime(Number(e.target.value))} placeholder="e.g., 8" />
                  </div>
                  <div>
                    <Label htmlFor="delivery-delta">Delivery Delta (days)</Label>
                    <p className="text-sm text-muted-foreground mt-1">Estimated minus actual delivery. Positive is early, negative is late.</p>
                    <Input id="delivery-delta" type="number" value={deliveryDelta} onChange={e => setDeliveryDelta(Number(e.target.value))} placeholder="e.g., 10 (early), -2 (late)" />
                  </div>
                  <Button onClick={handlePrediction} disabled={isLoading} className="w-full">
                    {isLoading ? <LoaderCircle className="animate-spin mr-2" /> : null}
                    {isLoading ? 'Predicting...' : 'Run Prediction'}
                  </Button>
                </div>

                {/* Right Column: Result Display */}
                <div className="min-h-[200px] flex items-center justify-center p-6 rounded-lg bg-primary/5">
                  {isLoading && <LoaderCircle className="w-12 h-12 text-primary animate-spin" />}
                  {error && (
                    <div className="text-center text-destructive">
                      <AlertTriangle className="mx-auto w-10 h-10 mb-2" />
                      <p className="font-semibold">Error</p>
                      <p className="text-sm">{error}</p>
                    </div>
                  )}
                  {result && (
                    <div className="text-center">
                      <p className="text-sm text-muted-foreground">Predicted Outcome</p>
                      <p className={`text-4xl font-bold ${result.is_satisfied_prediction === 1 ? 'text-green-500' : 'text-red-500'}`}>
                        {result.is_satisfied_prediction === 1 ? 'Satisfied' : 'Not Satisfied'}
                      </p>
                      <p className="text-muted-foreground mt-2">
                        Confidence: <span className="font-semibold text-primary">{(result.satisfaction_probability * 100).toFixed(1)}%</span>
                      </p>
                    </div>
                  )}
                  {!isLoading && !error && !result && (
                    <div className="text-center text-muted-foreground">
                      <p>Prediction results will appear here.</p>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* --- Sentiment Analysis Card --- */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="w-5 h-5 text-primary" />
                Sentiment Analysis
              </CardTitle>
              <CardDescription>
                Analyze customer review text to detect emotional sentiment (Portuguese)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <Label htmlFor="review-text">Customer Review</Label>
                  <Textarea
                    id="review-text"
                    placeholder="Digite ou cole a avaliação do cliente aqui..."
                    value={reviewText}
                    onChange={(e) => setReviewText(e.target.value)}
                    className="min-h-[120px] mt-2"
                  />
                  <div className="text-sm text-muted-foreground text-right mt-1">
                    {reviewText.length} characters
                  </div>
                </div>

                {/* Example Buttons */}
                <div className="flex flex-wrap gap-2">
                  <span className="text-sm text-muted-foreground self-center mr-2">
                    Try examples:
                  </span>
                  {exampleReviews.map((example, index) => (
                    <Button
                      key={index}
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setReviewText(example.text);
                        setSentimentResult(null);
                        setSentimentError(null);
                      }}
                      disabled={isSentimentLoading}
                    >
                      {example.label}
                    </Button>
                  ))}
                </div>

                <Button
                  onClick={handleSentimentAnalysis}
                  disabled={isSentimentLoading || !reviewText.trim()}
                  className="w-full"
                >
                  {isSentimentLoading ? (
                    <>
                      <LoaderCircle className="mr-2 h-5 w-5 animate-spin" />
                      Analyzing Sentiment...
                    </>
                  ) : (
                    "Analyze Sentiment"
                  )}
                </Button>

                {/* Sentiment Results */}
                {sentimentResult && (
                  <div className="mt-6 p-4 rounded-lg border-2 bg-primary/5">
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-sm font-medium text-muted-foreground">Detected Sentiment:</span>
                      <div className="flex items-center gap-2">
                        {sentimentResult.sentiment === "Positive" ? (
                          <TrendingUp className="h-5 w-5 text-green-500" />
                        ) : (
                          <TrendingDown className="h-5 w-5 text-red-500" />
                        )}
                        <Badge
                          variant={sentimentResult.sentiment === "Positive" ? "default" : "destructive"}
                          className="text-base px-3 py-1"
                        >
                          {sentimentResult.sentiment}
                        </Badge>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-muted-foreground">Confidence:</span>
                        <span className="text-xl font-bold">
                          {(sentimentResult.sentiment_score * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="w-full bg-secondary rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all ${
                            sentimentResult.sentiment === "Positive" ? "bg-green-500" : "bg-red-500"
                          }`}
                          style={{ width: `${sentimentResult.sentiment_score * 100}%` }}
                        />
                      </div>
                    </div>

                    <div className="mt-4 pt-4 border-t space-y-3">
                      <div>
                        <span className="text-xs font-medium text-muted-foreground block mb-1">
                          Processed Text:
                        </span>
                        <p className="text-sm bg-background p-2 rounded-md font-mono">
                          {sentimentResult.cleaned_text}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Sentiment Error */}
                {sentimentError && (
                  <div className="mt-4 p-4 rounded-lg border-2 border-destructive bg-destructive/5">
                    <div className="flex items-center gap-3 text-destructive">
                      <AlertTriangle className="h-5 w-5" />
                      <div>
                        <p className="font-medium">Sentiment Analysis Failed</p>
                        <p className="text-sm text-muted-foreground mt-1">{sentimentError}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

        </div>
      </main>
    </div>
  );
};

export default SatisfactionPredictorV1;
