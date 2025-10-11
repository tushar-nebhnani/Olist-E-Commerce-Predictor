import { useState } from "react";
import Navigation from "@/components/Navigation"; // Removed this line to resolve compilation error
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Brain, TrendingUp, Sparkles, LoaderCircle, AlertTriangle, ThumbsUp, ThumbsDown, ClipboardList, MessageSquare, Box, Weight, CreditCard } from "lucide-react";
import { useToast } from "@/hooks/use-toast"; // Removed this line to resolve potential compilation errors

// --- Type for API responses ---
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

// --- Hardcoded classification report data for the V2 model ---
const v2Report = {
  "Not Satisfied (0)": { precision: 0.50, recall: 0.55, "f1-score": 0.52, support: 4984 },
  "Satisfied (1)": { precision: 0.86, recall: 0.84, "f1-score": 0.85, support: 17291 },
  "macro avg": { precision: 0.68, recall: 0.70, "f1-score": 0.69, support: 22275 },
  "weighted avg": { precision: 0.78, recall: 0.77, "f1-score": 0.77, support: 22275 },
};

const SatisfactionPredictorV2 = () => {
  // const { toast } = useToast(); // Removed toast hook to resolve dependencies

  // --- State management for form inputs (V2 includes product details) ---
  const [price, setPrice] = useState(110.00);
  const [freightValue, setFreightValue] = useState(20.50);
  const [deliveryTime, setDeliveryTime] = useState(10);
  const [deliveryDelta, setDeliveryDelta] = useState(12);
  // --- NEW V2 FEATURES ---
  const [installments, setInstallments] = useState(3);
  const [photosQty, setPhotosQty] = useState(2);
  const [weightG, setWeightG] = useState(750);

  // --- State for the review text ---
  const [reviewText, setReviewText] = useState("The product arrived on time, but the quality wasn't what I expected for the price.");

  // --- State for API calls ---
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  const [sentimentResult, setSentimentResult] = useState<SentimentResult | null>(null);
  const [sentimentError, setSentimentError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // --- API call functions ---
  const handleSatisfactionPrediction = async () => {
    setIsPredicting(true);
    setPredictionResult(null);
    setPredictionError(null);

    // V2 request body includes the new features
    const requestBody = {
      // Original fields from the form
      price,
      freight_value: freightValue,
      delivery_time_days: deliveryTime,
      estimated_vs_actual_delivery: deliveryDelta,
      payment_installments: installments,
      product_photos_qty: photosQty,
      product_weight_g: weightG,
      
      // Fields required by the 'final' model, with plausible default values
      // You may want to add UI inputs for these later
      payment_value: price + freightValue, // Example calculation
      product_category_name: "cama_mesa_banho", // Example default value
      seller_avg_review_score: 4.1, // Example default value
      seller_order_count: 50, // Example default value
      distance_km: 150.5 // Example default value
    };

    try {
      // NOTE: Replace with your actual V2 satisfaction prediction endpoint
      // FIXED: Corrected the endpoint URL to match the FastAPI router configuration
      const response = await fetch('http://localhost:8000/satisfaction/final/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Prediction failed.');
      setPredictionResult(data);
    } catch (err: any) {
      setPredictionError(err.message);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleSentimentAnalysis = async () => {
    if (!reviewText.trim()) {
      // Replaced toast with a simple alert for compatibility
      alert("Input Required: Please enter some review text to analyze.");
      return;
    }
    setIsAnalyzing(true);
    setSentimentResult(null);
    setSentimentError(null);

    try {
      // NOTE: This now calls the v2 sentiment endpoint
      const response = await fetch('http://localhost:8000/api/v2/sentiment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: reviewText }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Analysis failed.');
      // Adapt the response from the standard sentiment endpoint if needed
      setSentimentResult({
          original_text: data.text,
          cleaned_text: data.cleaned_text || "N/A",
          sentiment: data.sentiment,
          sentiment_score: data.probability,
          prediction: data.sentiment === "Positive" ? 1 : 0
      });
    } catch (err: any) {
      setSentimentError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };


  return (
    <div className="min-h-screen bg-background">
      <Navigation /> 
      
      <main className="container mx-auto px-6 pt-12 pb-16"> {/* Adjusted padding-top */}
        <div className="max-w-4xl mx-auto space-y-8">
          {/* --- Hero Section --- */}
          <div className="text-center space-y-4">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
              <Brain className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-5xl font-bold tracking-tight">Satisfaction Model V2</h1>
            <p className="text-xl text-muted-foreground">
              Enhanced with product details for a more accurate prediction.
            </p>
          </div>

          {/* --- Model Overview & Report --- */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><TrendingUp className="w-5 h-5 text-primary" />Model V2 Overview</CardTitle>
              <CardDescription>LightGBM with Added Product Features</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="list-disc list-inside space-y-1 text-muted-foreground mb-6">
                <li><span className="font-semibold text-primary">More Context:</span> Incorporates payment installments, photo count, and weight.</li>
                <li><span className="font-semibold text-primary">Better Performance:</span> Achieves a higher F1-score for identifying dissatisfied customers.</li>
                <li><span className="font-semibold text-primary">Balanced Approach:</span> Good overall accuracy while improving on the V1 model's weaknesses.</li>
              </ul>
              <h3 className="font-semibold mb-2 flex items-center gap-2"><ClipboardList className="w-4 h-4" />Classification Report (V2)</h3>
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
                    {Object.entries(v2Report).map(([key, value]) => (
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

          {/* --- Live Satisfaction Predictor --- */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Sparkles className="w-5 h-5 text-primary" />Live Satisfaction Predictor (V2)</CardTitle>
              <CardDescription>Enter transaction and product details to predict customer satisfaction.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* --- Input Form --- */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="col-span-1">
                    <Label htmlFor="price">Price ($)</Label>
                    <Input id="price" type="number" value={price} onChange={e => setPrice(Number(e.target.value))} />
                  </div>
                  <div className="col-span-1">
                    <Label htmlFor="freight">Freight ($)</Label>
                    <Input id="freight" type="number" value={freightValue} onChange={e => setFreightValue(Number(e.target.value))} />
                  </div>
                  <div className="col-span-1">
                    <Label htmlFor="delivery-time">Delivery Time (days)</Label>
                    <Input id="delivery-time" type="number" value={deliveryTime} onChange={e => setDeliveryTime(Number(e.target.value))} />
                  </div>
                  <div className="col-span-1">
                    <Label htmlFor="delivery-delta">Est. vs Actual Delta</Label>
                    <Input id="delivery-delta" type="number" value={deliveryDelta} onChange={e => setDeliveryDelta(Number(e.target.value))} />
                  </div>
                  {/* --- NEW V2 INPUTS --- */}
                  <div className="col-span-2 border-t pt-4 grid grid-cols-3 gap-4">
                    <div className="col-span-1">
                        <Label htmlFor="installments"><CreditCard className="w-4 h-4 inline-block mr-1 mb-1"/>Installments</Label>
                        <Input id="installments" type="number" value={installments} onChange={e => setInstallments(Number(e.target.value))} />
                    </div>
                    <div className="col-span-1">
                        <Label htmlFor="photos-qty"><Box className="w-4 h-4 inline-block mr-1 mb-1"/>Photos Qty</Label>
                        <Input id="photos-qty" type="number" value={photosQty} onChange={e => setPhotosQty(Number(e.target.value))} />
                    </div>
                     <div className="col-span-1">
                        <Label htmlFor="weight-g"><Weight className="w-4 h-4 inline-block mr-1 mb-1"/>Weight (g)</Label>
                        <Input id="weight-g" type="number" value={weightG} onChange={e => setWeightG(Number(e.target.value))} />
                    </div>
                  </div>
                  <div className="col-span-2">
                    <Button onClick={handleSatisfactionPrediction} disabled={isPredicting} className="w-full">
                      {isPredicting && <LoaderCircle className="animate-spin mr-2" />}
                      {isPredicting ? 'Calculating...' : 'Predict Satisfaction'}
                    </Button>
                  </div>
                </div>

                {/* --- Prediction Result --- */}
                <div className="min-h-[150px] flex items-center justify-center rounded-lg bg-primary/5 p-6">
                  {isPredicting && <LoaderCircle className="w-10 h-10 text-primary animate-spin" />}
                  
                  {predictionError && (
                     <div className="text-center text-destructive">
                       <AlertTriangle className="mx-auto w-8 h-8 mb-2" />
                       <p className="font-semibold">Prediction Failed</p>
                       <p className="text-sm">{predictionError}</p>
                     </div>
                  )}

                  {predictionResult && (
                    <div className="text-center w-full">
                        <Badge variant={predictionResult.is_satisfied_prediction === 1 ? "default" : "destructive"}>
                            {predictionResult.is_satisfied_prediction === 1 ? 'Predicted: Satisfied' : 'Predicted: Not Satisfied'}
                        </Badge>
                        <div className="text-5xl font-bold my-2">
                            {(predictionResult.satisfaction_probability * 100).toFixed(1)}%
                        </div>
                        <p className="text-sm text-muted-foreground">Confidence Score</p>
                        
                        <div className="w-full bg-background rounded-full h-2.5 mt-4 border">
                            <div 
                                className={`h-2.5 rounded-full ${predictionResult.is_satisfied_prediction === 1 ? 'bg-green-500' : 'bg-red-500'}`}
                                style={{ width: `${predictionResult.satisfaction_probability * 100}%` }}
                            />
                        </div>
                    </div>
                  )}

                  {!isPredicting && !predictionResult && !predictionError && (
                    <div className="text-center text-muted-foreground">
                      <p>Prediction results will appear here.</p>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* --- Live Sentiment Analyzer --- */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><MessageSquare className="w-5 h-5 text-primary" />Review Sentiment Analyzer (V2)</CardTitle>
              <CardDescription>Enter a review comment to analyze its sentiment with the bilingual model.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Textarea
                  placeholder="Enter customer review text here..."
                  value={reviewText}
                  onChange={(e) => setReviewText(e.target.value)}
                  rows={4}
                />
                <Button onClick={handleSentimentAnalysis} disabled={isAnalyzing} className="w-full">
                  {isAnalyzing && <LoaderCircle className="animate-spin mr-2" />}
                  {isAnalyzing ? 'Analyzing...' : 'Analyze Review Text'}
                </Button>

                {/* Sentiment Result */}
                {sentimentResult && (
                  <div className="mt-4 p-4 rounded-lg border-2 bg-primary/5">
                    <div className="flex justify-between items-center">
                      <p className="text-lg font-bold">
                        Sentiment:{" "}
                        <span className={sentimentResult.sentiment === 'Positive' ? 'text-green-600' : 'text-red-600'}>
                          {sentimentResult.sentiment}
                        </span>
                      </p>
                      <div className="flex items-center gap-2">
                        {sentimentResult.sentiment === 'Positive' ? 
                          <ThumbsUp className="h-5 w-5 text-green-600" /> : 
                          <ThumbsDown className="h-5 w-5 text-red-600" />
                        }
                      </div>
                    </div>

                     <div className="mt-2">
                      <Label className="text-xs">Confidence Score</Label>
                      <div className="w-full bg-background rounded-full h-2.5 mt-1 border">
                        <div
                          className={`h-2.5 rounded-full ${sentimentResult.sentiment === 'Positive' ? 'bg-green-500' : 'bg-red-500'}`}
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

export default SatisfactionPredictorV2;

