// src/pages/PurchasePredictionV2.tsx

import { useState } from "react";
import Navigation from "@/components/Navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Brain, Sparkles, LoaderCircle, AlertTriangle, ShoppingCart, Percent, ThumbsUp, ThumbsDown, ClipboardList, BarChart4 } from "lucide-react"; // Added BarChart4 icon

interface PredictionResult {
  purchase_probability: number;
}

const v2ReportFull = {
  "Not Purchased (0)": { precision: 0.99, recall: 0.90, "f1-score": 0.94, support: 94556 },
  "Purchased (1)": { precision: 0.70, recall: 0.95, "f1-score": 0.81, support: 23910 },
  "accuracy": { precision: null, recall: null, "f1-score": 0.91, support: 118466 }, // Accuracy as f1-score here represents overall accuracy
  "macro avg": { precision: 0.84, recall: 0.92, "f1-score": 0.87, support: 118466 },
  "weighted avg": { precision: 0.93, recall: 0.91, "f1-score": 0.91, support: 118466 }
};

const PurchasePredictorV2 = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const [formData, setFormData] = useState({
    price: 120.0,
    freight_value: 20.0,
    product_photos_qty: 2.0,
    product_weight_g: 500.0,
    product_volume_cm3: 1000.0,
    distance_km: 500.0,
    review_score: 4.0,
    product_category_name_english: "health_beauty",
    customer_state: "SP",
    seller_state: "SP",
    customer_avg_review_score: 4.1,
    customer_order_count: 1.0,
    customer_total_spend: 150.0,
    product_popularity: 50.0,
    product_avg_review_score: 4.2,
    price_vs_category_avg: -10.5,
    category_popularity: 5000.0,
    category_avg_review_score: 4.15,
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: parseFloat(value) || 0 }));
  };

  const handleSelectChange = (name: string, value: string) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('https://olist-e-commerce-predictor-production.up.railway.app/purchase/v2/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "An error occurred");
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const stateOptions = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "ES", "GO", "PE", "CE", "PA", "MT", "MA", "MS", "PB", "PI", "RN", "AL", "SE", "TO", "RO", "AM", "AC", "AP", "RR"].sort().map(s => <SelectItem key={s} value={s}>{s}</SelectItem>);
  
  const productCategories = [
    "health_beauty", "computers_accessories", "auto", "bed_bath_table", "furniture_decor", "sports_leisure", "perfumery", "housewares", "telephony", "watches_gifts", "fashion_bags_accessories", "cool_stuff", "stationery", "toys", "food_drink",
  ].sort();
  const categoryOptions = productCategories.map(c => <SelectItem key={c} value={c}>{c.replace(/_/g, ' ')}</SelectItem>);

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="container mx-auto px-6 pt-24 pb-16">
        <div className="max-w-6xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
              <ShoppingCart className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-5xl font-bold tracking-tight">
              Purchase Predictor (Advanced Model)
            </h1>
            <p className="text-xl text-muted-foreground">
              An advanced model with behavioral and reputational features for superior purchase prediction.
            </p>
          </div>

          <Card className="border-2">
            <CardHeader>
              <CardTitle>Advanced Model Performance</CardTitle>
              <CardDescription>LightGBM model with enhanced features and imbalance handling.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">0.9754</div>
                  <div className="text-sm text-muted-foreground mt-1">AUC-ROC</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">91%</div>
                  <div className="text-sm text-muted-foreground mt-1">Accuracy</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">0.70</div>
                  <div className="text-sm text-muted-foreground mt-1">Precision (Purchased)</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                  <div className="text-3xl font-bold text-green-500">0.95</div>
                  <div className="text-sm text-muted-foreground mt-1">Recall (Purchased)</div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Brain className="w-5 h-5 text-primary" />Model Profile</CardTitle>
              <CardDescription>Strengths and strategic trade-offs of the advanced V2 model.</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <h3 className="font-semibold flex items-center text-green-500"><ThumbsUp className="w-4 h-4 mr-2" />Advantages</h3>
                <ul className="list-disc list-inside space-y-1 text-muted-foreground text-sm">
                  <li>**Exceptional Recall (0.95):** Identifies 95% of all potential purchases, making it highly effective for targeted campaigns.</li>
                  <li>**Outstanding AUC (0.9754):** Demonstrates superior ability to differentiate between buyers and non-buyers.</li>
                  <li>**Rich Feature Set:** Leverages customer history, product popularity, and seller reputation for highly accurate predictions.</li>
                  <li>**High Business Value:** Maximizes opportunity for marketing, sales, and personalized user experiences.</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h3 className="font-semibold flex items-center text-amber-500"><ThumbsDown className="w-4 h-4 mr-2" />Strategic Trade-offs</h3>
                <ul className="list-disc list-inside space-y-1 text-muted-foreground text-sm">
                  <li>**Moderate Precision (0.70):** To achieve high recall, it casts a wider net, meaning some non-buyers might be incorrectly flagged (false positives).</li>
                  <li>**Increased Complexity:** Requires more data processing and a richer feature set compared to the baseline.</li>
                  <li>**Computational Cost:** Training and prediction can be slightly more resource-intensive due to the larger feature space.</li>
                </ul>
              </div>
            </CardContent>
          </Card>
          
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><ClipboardList className="w-5 h-5 text-primary" />Classification Report</CardTitle>
              <CardDescription>A detailed breakdown of the V2 model's performance on the test set.</CardDescription>
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
                    {Object.entries(v2ReportFull).map(([key, value]) => (
                      <tr key={key} className="border-b border-primary/10 last:border-b-0">
                        <td className="px-4 py-2 font-semibold capitalize">{key.replace(/_/g, ' ')}</td>
                        <td className="px-4 py-2 text-center">{value.precision !== null ? value.precision.toFixed(2) : '—'}</td>
                        <td className="px-4 py-2 text-center">{value.recall !== null ? value.recall.toFixed(2) : '—'}</td>
                        <td className="px-4 py-2 text-center">
                          {key === 'accuracy' ? value['f1-score'].toFixed(2) : (value['f1-score'] !== null ? value['f1-score'].toFixed(2) : '—')}
                        </td>
                        <td className="px-4 py-2 text-center text-muted-foreground">{value.support.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                *The high recall of 0.95 for "Purchased (1)" indicates excellent capability to identify potential buyers.
              </p>
            </CardContent>
          </Card>

          {/* --- NEW CARD: Feature Importance --- */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><BarChart4 className="w-5 h-5 text-primary" />Feature Importance</CardTitle>
              <CardDescription>Key features driving the V2 model's purchase predictions.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative w-full h-auto flex justify-center">
                <img
                  src="/feature_importance_v2.png" // Path to the image in your public folder
                  alt="Top 20 Feature Importances for V2 Model"
                  className="max-w-full h-auto rounded-lg shadow-md"
                  style={{ maxWidth: '800px' }} // Adjust max-width as needed
                />
              </div>
              <p className="text-xs text-muted-foreground mt-4 text-center">
                `customer_total_spend`, `price`, `distance_km`, `product_popularity`, and `freight_value` are identified as the most influential factors in predicting purchase probability by the advanced model.
              </p>
            </CardContent>
          </Card>

          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-primary" />
                Interactive Predictor
              </CardTitle>
              <CardDescription>Input features to get a purchase probability prediction</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="md:col-span-2">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(formData).map(([key, value]) => (
                      <div key={key} className="space-y-1">
                        <Label htmlFor={key} className="capitalize text-xs">{key.replace(/_/g, ' ')}</Label>
                        {['customer_state', 'seller_state', 'product_category_name_english'].includes(key) ? (
                          <Select onValueChange={(val) => handleSelectChange(key, val)} defaultValue={value.toString()}>
                            <SelectTrigger id={key}><SelectValue /></SelectTrigger>
                            <SelectContent>{key === 'product_category_name_english' ? categoryOptions : stateOptions}</SelectContent>
                          </Select>
                        ) : (
                          <Input id={key} name={key} type="number" value={value} onChange={handleChange} />
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex flex-col gap-4 items-center justify-center p-6 rounded-lg border-2 border-primary/20 bg-primary/5">
                  <Button onClick={handleSubmit} disabled={isLoading} className="w-full" size="lg">
                    {isLoading ? <LoaderCircle className="animate-spin mr-2" /> : <Brain className="mr-2" />}
                    Predict Purchase
                  </Button>
                  
                  <div className="text-center w-full">
                    {isLoading && <p className="text-muted-foreground animate-pulse">Analyzing...</p>}
                    {error && (
                      <div className="text-center text-destructive p-4 rounded-md bg-destructive/10">
                        <AlertTriangle className="mx-auto w-8 h-8 mb-2" />
                        <p className="font-semibold">Prediction Failed</p>
                        <p className="text-sm">{error}</p>
                      </div>
                    )}
                    {result && (
                      <div className="text-center">
                        <p className="text-sm text-muted-foreground mb-2">Purchase Probability</p>
                        <div className="flex items-center justify-center gap-2">
                          <ShoppingCart className="w-10 h-10 text-primary" />
                          <p className="text-6xl font-bold text-primary">
                            {(result.purchase_probability * 100).toFixed(1)}<span className="text-3xl">%</span>
                          </p>
                        </div>
                        <p className={`mt-2 font-semibold ${result.purchase_probability > 0.5 ? 'text-green-600' : 'text-amber-600'}`}>
                          {result.purchase_probability > 0.7 ? "Very Likely Purchase" : result.purchase_probability > 0.4 ? "Possible Purchase" : "Unlikely Purchase"}
                        </p>
                      </div>
                    )}
                    {!isLoading && !error && !result && (
                      <div className="text-center text-muted-foreground p-4">
                        <Percent className="mx-auto w-10 h-10 mb-2" />
                        <p>Prediction results will appear here.</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default PurchasePredictorV2;