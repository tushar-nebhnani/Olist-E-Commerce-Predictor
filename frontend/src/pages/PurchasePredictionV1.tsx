import { useState } from "react";
import Navigation from "@/components/Navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Brain, ShoppingCart, LoaderCircle, AlertTriangle, Percent, ThumbsUp, ThumbsDown, ClipboardList } from "lucide-react";
import { api } from "@/services/api";

interface PredictionResult {
  purchase_probability: number;
}

// Data for the classification report
const v1ReportFull = {
  "Not Purchased (0)": { precision: 0.83, recall: 0.99, "f1-score": 0.90, support: 95760 },
  "Purchased (1)": { precision: 0.82, recall: 0.20, "f1-score": 0.32, support: 23937 },
  "accuracy": { precision: null, recall: null, "f1-score": 0.83, support: 119697 }, // This is kept for reference but will be filtered out in the UI
  "macro avg": { precision: 0.82, recall: 0.59, "f1-score": 0.61, support: 119697 },
  "weighted avg": { precision: 0.83, recall: 0.83, "f1-score": 0.79, support: 119697 }
};

const PurchasePredictorV1 = () => {
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
    purchase_month: 6,
    purchase_dayofweek: 3,
    product_category_name_english: "health_beauty",
    customer_state: "SP",
    seller_state: "SP",
    review_score: 4.0,
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
      const data = await api.predictPurchaseV1(formData);
      setResult(data);
    } catch (err: any) {
      let errorMessage = 'An unknown error occurred.';
      if (err.message) {
        errorMessage = err.message;
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const stateOptions = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "ES", "GO", "PE", "CE", "PA", "MT", "MA", "MS", "PB", "PI", "RN", "AL", "SE", "TO", "RO", "AM", "AC", "AP", "RR"].sort().map(s => <SelectItem key={s} value={s}>{s}</SelectItem>);
  
  const productCategories = [
    "health_beauty", "computers_accessories", "auto", "bed_bath_table", "furniture_decor", "sports_leisure", "perfumery", "housewares", "telephony", "watches_gifts", "food_drink", "stationery", "toys", "fashion_bags_accessories", "cool_stuff"
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
              Purchase Predictor (Baseline Model)
            </h1>
            <p className="text-xl text-muted-foreground">
              A foundational model predicting purchase probability using core transaction features.
            </p>
          </div>

          <Card className="border-2">
            <CardHeader>
              <CardTitle>Baseline Model Performance</CardTitle>
              <CardDescription>LightGBM model trained on core features.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">0.7575</div>
                  <div className="text-sm text-muted-foreground mt-1">AUC-ROC</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">83%</div>
                  <div className="text-sm text-muted-foreground mt-1">Accuracy</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">0.82</div>
                  <div className="text-sm text-muted-foreground mt-1">Precision (Purchased)</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-red-500/10 border border-red-500/20">
                  <div className="text-3xl font-bold text-red-500">0.20</div>
                  <div className="text-sm text-muted-foreground mt-1">Recall (Purchased)</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Brain className="w-5 h-5 text-primary" />Model Profile</CardTitle>
              <CardDescription>Strengths and weaknesses of the baseline model.</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <h3 className="font-semibold flex items-center text-green-500"><ThumbsUp className="w-4 h-4 mr-2" />Advantages</h3>
                <ul className="list-disc list-inside space-y-1 text-muted-foreground text-sm">
                  <li>High Overall Accuracy: Correctly predicts the outcome in 83% of cases, driven by its strength on the majority class.</li>
                  <li>High Precision: When it predicts a purchase, it is correct 82% of the time, avoiding false alarms.</li>
                  <li>Good Benchmark: Establishes a solid performance baseline to measure improvements against.</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h3 className="font-semibold flex items-center text-red-500"><ThumbsDown className="w-4 h-4 mr-2" />Critical Flaw (The "Fall")</h3>
                <ul className="list-disc list-inside space-y-1 text-muted-foreground text-sm">
                  <li>Extremely Low Recall: Fails to identify 80% of actual buyers, making it unsuitable for marketing or sales targeting.</li>
                  <li>Biased Towards Majority Class: Due to class imbalance, it learns to predict "Not Purchased" by default.</li>
                  <li>Limited Features:** Lacks deeper customer behavior and product popularity signals, limiting its intelligence.</li>
                </ul>
              </div>
            </CardContent>
          </Card>

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
                    {Object.entries(v1ReportFull)
                      .filter(([key]) => key !== 'accuracy') // MODIFIED: Filter out the 'accuracy' row
                      .map(([key, value]) => (
                        <tr key={key} className="border-b border-primary/10 last:border-b-0">
                          <td className="px-4 py-2 font-semibold capitalize">{key}</td>
                          <td className="px-4 py-2 text-center">{value.precision !== null ? value.precision.toFixed(2) : '—'}</td>
                          <td className="px-4 py-2 text-center">{value.recall !== null ? value.recall.toFixed(2) : '—'}</td>
                          <td className="px-4 py-2 text-center">
                            {/* MODIFIED: Simplified the logic, no need for the 'accuracy' special case */}
                            {value['f1-score'] !== null ? value['f1-score'].toFixed(2) : '—'}
                          </td>
                          <td className="px-4 py-2 text-center text-muted-foreground">{value.support.toLocaleString()}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                *The key takeaway is the Recall for "Purchased (1)" is only 0.20, meaning 80% of buyers are missed.
              </p>
            </CardContent>
          </Card>
          
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="w-5 h-5 text-primary" />
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

export default PurchasePredictorV1;