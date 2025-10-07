import { useState } from "react";
import Navigation from "@/components/Navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Brain, ShoppingCart, LoaderCircle, AlertTriangle, Percent, BarChart3 } from "lucide-react";

interface PredictionResult {
  purchase_probability: number;
}

const v1Report = {
  "Not Purchased (0)": { precision: 0.83, recall: 0.99, "f1-score": 0.90 },
  "Purchased (1)": { precision: 0.82, recall: 0.19, "f1-score": 0.31 },
  "accuracy": 0.83,
  "AUC": 0.7564
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
      const response = await fetch('http://127.0.0.1:8000/purchase/v1/predict', {
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
              Purchase Predictor V1
            </h1>
            <p className="text-xl text-muted-foreground">
              Baseline model predicting purchase probability using core transaction features
            </p>
          </div>

          <Card className="border-2">
            <CardHeader>
              <div className="flex items-center gap-4">
                <BarChart3 className="w-8 h-8 text-primary" />
                <div>
                  <CardTitle>Baseline Model Performance</CardTitle>
                  <CardDescription>Version 1.0 - Core Features</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">{v1Report.AUC.toFixed(4)}</div>
                  <div className="text-sm text-muted-foreground mt-1">AUC-ROC</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">{(v1Report.accuracy * 100).toFixed(1)}%</div>
                  <div className="text-sm text-muted-foreground mt-1">Accuracy</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">{v1Report["Purchased (1)"].precision.toFixed(2)}</div>
                  <div className="text-sm text-muted-foreground mt-1">Precision</div>
                </div>
                <div className="text-center p-4 rounded-lg bg-primary/5">
                  <div className="text-3xl font-bold text-primary">{v1Report["Purchased (1)"].recall.toFixed(2)}</div>
                  <div className="text-sm text-muted-foreground mt-1">Recall</div>
                </div>
              </div>
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
