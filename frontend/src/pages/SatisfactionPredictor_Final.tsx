// src/pages/SatisfactionPredictor_Final.tsx
import { useState } from "react";
import Navigation from "@/components/Navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Brain, TrendingUp, BarChart3, Sparkles, LoaderCircle, AlertTriangle, ThumbsUp, ThumbsDown, ClipboardList, Target } from "lucide-react";

// --- Type for API response ---
interface PredictionResult {
  is_satisfied_prediction: number;
  satisfaction_probability: number;
}

// --- Hardcoded classification report data for the FINAL model ---
const finalReport = {
  "Not Satisfied (0)": { precision: 0.46, recall: 0.60, "f1-score": 0.52, support: 5208 },
  "Satisfied (1)": { precision: 0.87, recall: 0.78, "f1-score": 0.82, support: 17248 },
  "macro avg": { precision: 0.66, recall: 0.69, "f1-score": 0.67, support: 22456 },
  "weighted avg": { precision: 0.77, recall: 0.74, "f1-score": 0.75, support: 22456 },
};

// --- Category translations for the form ---
const categoryTranslations: { [key: string]: string } = {
  cama_mesa_banho: 'Bed, Table & Bath',
  beleza_saude: 'Health & Beauty',
  esporte_lazer: 'Sports & Leisure',
  informatica_acessorios: 'Computer Accessories',
  moveis_decoracao: 'Furniture & Decor',
  utilidades_domesticas: 'Housewares',
  relogios_presentes: 'Watches & Gifts',
  telefonia: 'Telephony',
  automotivo: 'Automotive',
  brinquedos: 'Toys',
  Other: 'Other'
};

const SatisfactionPredictorFinal = () => {
  // --- State management for form inputs (includes ALL advanced features) ---
  const [price, setPrice] = useState(129.90);
  const [freightValue, setFreightValue] = useState(15.50);
  const [deliveryTime, setDeliveryTime] = useState(8);
  const [deliveryDelta, setDeliveryDelta] = useState(10);
  const [installments, setInstallments] = useState(1);
  const [paymentValue, setPaymentValue] = useState(145.40);
  const [photosQty, setPhotosQty] = useState(1);
  const [weightG, setWeightG] = useState(500);
  const [category, setCategory] = useState('cama_mesa_banho');
  const [sellerAvgScore, setSellerAvgScore] = useState(4.1);
  const [sellerOrderCount, setSellerOrderCount] = useState(150);
  const [distance, setDistance] = useState(650);

  // --- State management for the API call ---
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  // --- API call function ---
  const handlePrediction = async () => {
    setResult(null);
    setError(null);
    setIsLoading(true);

    const requestBody = {
      price,
      freight_value: freightValue,
      delivery_time_days: deliveryTime,
      estimated_vs_actual_delivery: deliveryDelta,
      payment_installments: installments,
      payment_value: paymentValue,
      product_photos_qty: photosQty,
      product_weight_g: weightG,
      product_category_name: category,
      seller_avg_review_score: sellerAvgScore,
      seller_order_count: sellerOrderCount,
      distance_km: distance
    };

    try {
      const response = await fetch('http://localhost:8000/satisfaction/final/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();
      if (!response.ok) throw data;
      setResult(data);

    } catch (error: any) {
      console.error("API Error:", error);
      if (error.detail && Array.isArray(error.detail)) {
        const firstError = error.detail[0];
        setError(`Validation Error: Field '${firstError.loc[1]}' - ${firstError.msg}.`);
      } else {
        setError(error.detail || "An unknown error occurred.");
      }
    } finally {
      setIsLoading(false);
    }
  };

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
              Final Satisfaction Model
            </h1>
            <p className="text-xl text-muted-foreground">
              Optimized for detecting dissatisfied customers using XGBoost.
            </p>
          </div>

          {/* --- Model Overview Card --- */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><TrendingUp className="w-5 h-5 text-primary" />Model Overview</CardTitle>
              <CardDescription>Final Version - XGBoost with Advanced Features</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                <li>**Advanced Feature Set:** Uses seller performance metrics and geographic distance for maximum context.</li>
                <li>**Optimized for Recall:** Employs an XGBoost classifier with `scale_pos_weight` to prioritize finding negative reviews.</li>
                <li>**Strategic Trade-off:** Sacrifices some overall accuracy to achieve a massive boost in identifying unhappy customers.</li>
                <li>**Business-Focused:** The best model for proactive customer support and intervention.</li>
              </ul>
            </CardContent>
          </Card>

          {/* --- Key Metrics Card --- */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Target className="w-5 h-5 text-primary" />Key Performance Indicator</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center p-4 rounded-lg bg-red-500/10 border border-red-500/20">
                  <div className="text-3xl font-bold text-red-500">60%</div>
                  <div className="text-sm text-muted-foreground mt-1">Recall (Dissatisfied)</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">74%</div>
                  <div className="text-sm text-muted-foreground mt-1">Overall Accuracy</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">0.75</div>
                  <div className="text-sm text-muted-foreground mt-1">Weighted F1 Score</div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* --- Model Profile Card --- */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Brain className="w-5 h-5 text-primary" />Model Profile</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <h3 className="font-semibold flex items-center text-green-500"><ThumbsUp className="w-4 h-4 mr-2" />Advantages</h3>
                <ul className="list-disc list-inside space-y-1 text-muted-foreground text-sm">
                  <li>**High Recall:** Successfully identifies 60% of all dissatisfied customers, a huge improvement.</li>
                  <li>**Actionable Insights:** Ideal for flagging at-risk orders for proactive intervention.</li>
                  <li>**Powerful Technique:** XGBoost's class weighting is a state-of-the-art method for imbalance.</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h3 className="font-semibold flex items-center text-red-500"><ThumbsDown className="w-4 h-4 mr-2" />Disadvantages</h3>
                <ul className="list-disc list-inside space-y-1 text-muted-foreground text-sm">
                  <li>**Lower Precision:** More likely to incorrectly flag a satisfied customer as "at-risk".</li>
                  <li>**Lower Overall Accuracy:** The strategic focus on recall slightly reduces the overall accuracy metric.</li>
                  <li>**Most Complex:** Requires the most features, making the API and training more complex.</li>
                </ul>
              </div>
            </CardContent>
          </Card>

          {/* --- Classification Report Card --- */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><ClipboardList className="w-5 h-5 text-primary" />Classification Report</CardTitle>
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
                    {Object.entries(finalReport).map(([key, value]) => (
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
              <CardTitle className="flex items-center gap-2"><Sparkles className="w-5 h-5 text-primary" />Live Prediction Interface</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-8 items-start">
                {/* Left Column: Form with ALL final model parameters */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="col-span-1">
                    <Label htmlFor="price">Price ($)</Label>
                    <Input id="price" type="number" value={price} onChange={e => setPrice(Number(e.target.value))} />
                  </div>
                   <div className="col-span-1">
                    <Label htmlFor="freight">Freight Value ($)</Label>
                    <Input id="freight" type="number" value={freightValue} onChange={e => setFreightValue(Number(e.target.value))} />
                  </div>
                  <div className="col-span-1">
                    <Label htmlFor="distance">Distance (km)</Label>
                    <Input id="distance" type="number" value={distance} onChange={e => setDistance(Number(e.target.value))} />
                  </div>
                  <div className="col-span-1">
                    <Label htmlFor="seller-score">Seller Avg. Score (1-5)</Label>
                    <Input id="seller-score" type="number" step="0.1" max="5" min="1" value={sellerAvgScore} onChange={e => setSellerAvgScore(Number(e.target.value))} />
                  </div>
                  <div className="col-span-2 grid grid-cols-2 gap-4">
                    <div className="col-span-1">
                      <Label htmlFor="delivery-time">Delivery Time (days)</Label>
                      <Input id="delivery-time" type="number" value={deliveryTime} onChange={e => setDeliveryTime(Number(e.target.value))} />
                    </div>
                    <div className="col-span-1">
                      <Label htmlFor="delivery-delta">Delivery Delta (days)</Label>
                      <Input id="delivery-delta" type="number" value={deliveryDelta} onChange={e => setDeliveryDelta(Number(e.target.value))} />
                    </div>
                  </div>
                   <div className="col-span-2 grid grid-cols-2 gap-4">
                    <div className="col-span-1">
                      <Label htmlFor="installments">Installments</Label>
                      <Input id="installments" type="number" value={installments} onChange={e => setInstallments(Number(e.target.value))} />
                    </div>
                    <div className="col-span-1">
                      <Label htmlFor="payment-value">Payment Value ($)</Label>
                      <Input id="payment-value" type="number" value={paymentValue} onChange={e => setPaymentValue(Number(e.target.value))} />
                    </div>
                  </div>
                   <div className="col-span-2 grid grid-cols-2 gap-4">
                     <div className="col-span-1">
                      <Label htmlFor="photos-qty">Photos Qty</Label>
                      <Input id="photos-qty" type="number" value={photosQty} onChange={e => setPhotosQty(Number(e.target.value))} />
                    </div>
                    <div className="col-span-1">
                      <Label htmlFor="weight-g">Weight (g)</Label>
                      <Input id="weight-g" type="number" value={weightG} onChange={e => setWeightG(Number(e.target.value))} />
                    </div>
                  </div>
                  <div className="col-span-2">
                     <Label htmlFor="seller-order-count">Seller Order Count</Label>
                    <Input id="seller-order-count" type="number" value={sellerOrderCount} onChange={e => setSellerOrderCount(Number(e.target.value))} />
                  </div>
                  <div className="col-span-2">
                    <Label htmlFor="category">Product Category</Label>
                    <Select value={category} onValueChange={setCategory}>
                        <SelectTrigger><SelectValue placeholder="Select a category" /></SelectTrigger>
                        <SelectContent>
                            {Object.entries(categoryTranslations).map(([key, value]) => (<SelectItem key={key} value={key}>{value}</SelectItem>))}
                        </SelectContent>
                    </Select>
                  </div>
                  <div className="col-span-2">
                    <Button onClick={handlePrediction} disabled={isLoading} className="w-full">
                      {isLoading ? <LoaderCircle className="animate-spin mr-2" /> : null}
                      {isLoading ? 'Predicting...' : 'Run Prediction'}
                    </Button>
                  </div>
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
        </div>
      </main>
    </div>
  );
};

export default SatisfactionPredictorFinal;