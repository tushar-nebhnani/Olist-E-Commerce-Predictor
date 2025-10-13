import { useState } from "react";
import Navigation from "@/components/Navigation"; // Removed this line to resolve compilation error
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Brain, TrendingUp, Sparkles, LoaderCircle, AlertTriangle, ThumbsUp, ThumbsDown, ClipboardList, MessageSquare, Box, Weight, CreditCard, Clock, MapPin, Store, Users } from "lucide-react";

// NOTE: Since external components like Navigation and useToast cause errors, 
// they are commented out, and placeholder components (like <Navigation />) 
// are left for structure if the user restores them later.

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

// --- Hardcoded classification report data for the FINAL XGBoost model ---
const v2Report = {
    "Not Satisfied (0)": { precision: 0.50, recall: 0.55, "f1-score": 0.52, support: 4984 },
    "Satisfied (1)": { precision: 0.86, recall: 0.84, "f1-score": 0.85, support: 17291 },
    "macro avg": { precision: 0.68, recall: 0.70, "f1-score": 0.69, support: 22275 },
    "weighted avg": { precision: 0.78, recall: 0.77, "f1-score": 0.77, support: 22275 },
};

// --- Dropdown options for Product Category Name (Top 20 used in model) ---
const CATEGORY_OPTIONS = [
    "cama_mesa_banho", "saude_beleza", "esporte_lazer", "moveis_decoracao", "informatica_acessorios",
    "utilidades_domesticas", "relogios_presentes", "telefonia", "ferramentas_jardim", "automotivo",
    "brinquedos", "cool_stuff", "bebes", "perfumaria", "papelaria", 
    "fashion_bolsas_e_acessorios", "eletronicos", "consoles_games", "construcao_ferramentas_construcao",
    "eletrodomesticos", "Other" // 'Other' category used by the ColumnTransformer
];


const SatisfactionPredictorV2 = () => {
    // --- State management for form inputs (V2 requires 12 features) ---
    const [price, setPrice] = useState(110.00);
    const [freightValue, setFreightValue] = useState(20.50);
    const [deliveryTime, setDeliveryTime] = useState(10); // delivery_time_days
    const [deliveryDelta, setDeliveryDelta] = useState(12); // estimated_vs_actual_delivery

    // --- V2/FINAL XGBOOST FEATURES ---
    const [installments, setInstallments] = useState(3); // payment_installments
    const [paymentValueTotal, setPaymentValueTotal] = useState(130.50); // payment_value
    const [photosQty, setPhotosQty] = useState(2); // product_photos_qty
    const [weightG, setWeightG] = useState(750); // product_weight_g
    const [categoryName, setCategoryName] = useState(CATEGORY_OPTIONS[0]); // product_category_name

    // --- GEOSPATIAL & SELLER HISTORY ---
    const [distanceKm, setDistanceKm] = useState(150.5); // distance_km
    const [sellerAvgScore, setSellerAvgScore] = useState(4.1); // seller_avg_review_score
    const [sellerOrderCount, setSellerOrderCount] = useState(50); // seller_order_count

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

        // Request body must include all 12 features required by the ColumnTransformer/XGBoost V2
        const requestBody = {
            price,
            freight_value: freightValue,
            delivery_time_days: deliveryTime,
            estimated_vs_actual_delivery: deliveryDelta,
            payment_installments: installments,
            payment_value: paymentValueTotal,
            product_photos_qty: photosQty,
            product_weight_g: weightG,
            product_category_name: categoryName,
            distance_km: distanceKm,
            seller_avg_review_score: sellerAvgScore,
            seller_order_count: sellerOrderCount,
        };

        try {
            // NOTE: Replace with your actual V2 satisfaction prediction endpoint
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
            alert("Input Required: Please enter some review text to analyze.");
            return;
        }
        setIsAnalyzing(true);
        setSentimentResult(null);
        setSentimentError(null);

        try {
            // NOTE: This calls the v2 sentiment endpoint
            const response = await fetch('http://localhost:8000/api/v2/sentiment', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: reviewText }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Analysis failed.');
            
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
            <div className="pt-24 h-16 w-full bg-primary/5 fixed top-0 left-0 z-10 flex items-center justify-center border-b">
                <span className="text-lg font-semibold text-primary">Olist ML Dashboard</span>
            </div>
            
            <main className="container mx-auto px-6 pt-24 pb-16"> {/* Adjusted padding-top */}
                <div className="max-w-6xl mx-auto space-y-8">
                    {/* --- Hero Section --- */}
                    <div className="text-center space-y-4">
                        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
                            <Brain className="w-8 h-8 text-primary" />
                        </div>
                        <h1 className="text-5xl font-bold tracking-tight">Satisfaction Model V2 (Final)</h1>
                        <p className="text-xl text-muted-foreground">
                            XGBoost Classifier with Geospatial and Historical Context.
                        </p>
                    </div>

                    {/* --- Model Overview & Report --- */}
                    <Card className="border-2 shadow-xl shadow-primary/10">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2"><TrendingUp className="w-5 h-5 text-primary" />Model Architecture & Rigor</CardTitle>
                            <CardDescription>XGBoost with Class Weighting for Proactive Risk Intervention.</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ul className="list-disc list-inside space-y-1 text-muted-foreground mb-6">
                                <li><span className="font-semibold text-primary">Model:</span> XGBoost (Stability & Control)</li>
                                <li><span className="font-semibold text-primary">Imbalance Strategy:</span> Class Weighting (Purity over SMOTE synthesis).</li>
                                <li><span className="font-semibold text-primary">Core Feature:</span> distance_km (Controls for logistical reality).</li>
                            </ul>
                            <h3 className="font-semibold mb-2 flex items-center gap-2"><ClipboardList className="w-4 h-4" />Classification Report (V2)</h3>
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm text-left border rounded-lg">
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
                                            <tr key={key} className="border-b border-primary/10 last:border-b-0">
                                                <td className="px-4 py-2 font-semibold capitalize">
                                                    <Badge variant={key.includes("Not Satisfied") ? "destructive" : key.includes("Satisfied") ? "default" : "secondary"}>
                                                        {key}
                                                    </Badge>
                                                </td>
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
                    <Card className="border-2 shadow-xl shadow-primary/10">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2"><Sparkles className="w-5 h-5 text-primary" />Live Prediction Engine</CardTitle>
                            <CardDescription>Input all 12 V2 features for accurate churn risk prediction.</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                {/* --- Input Form --- */}
                                <div className="grid grid-cols-3 gap-4">
                                    {/* --- Core Logistics Features --- */}
                                    <div className="col-span-1">
                                        <Label htmlFor="price">Price ($)</Label>
                                        <Input id="price" type="number" value={price} onChange={e => setPrice(Number(e.target.value))} />
                                    </div>
                                    <div className="col-span-1">
                                        <Label htmlFor="freight">Freight ($)</Label>
                                        <Input id="freight" type="number" value={freightValue} onChange={e => setFreightValue(Number(e.target.value))} />
                                    </div>
                                    <div className="col-span-1">
                                        <Label htmlFor="total-payment">Total Payment ($)</Label>
                                        <Input id="total-payment" type="number" value={paymentValueTotal} onChange={e => setPaymentValueTotal(Number(e.target.value))} />
                                    </div>

                                    {/* --- Time & Performance Features --- */}
                                    <div className="col-span-1">
                                        <Label htmlFor="delivery-time"><Clock className="w-4 h-4 inline-block mr-1 mb-1"/>Actual Time (days)</Label>
                                        <Input id="delivery-time" type="number" value={deliveryTime} onChange={e => setDeliveryTime(Number(e.target.value))} />
                                    </div>
                                    <div className="col-span-1">
                                        <Label htmlFor="delivery-delta"><TrendingUp className="w-4 h-4 inline-block mr-1 mb-1"/>Est. vs Actual Delta</Label>
                                        <Input id="delivery-delta" type="number" value={deliveryDelta} onChange={e => setDeliveryDelta(Number(e.target.value))} />
                                    </div>
                                    <div className="col-span-1">
                                        <Label htmlFor="installments"><CreditCard className="w-4 h-4 inline-block mr-1 mb-1"/>Installments</Label>
                                        <Input id="installments" type="number" value={installments} onChange={e => setInstallments(Number(e.target.value))} />
                                    </div>

                                    {/* --- Product & Geospatial Context --- */}
                                    <div className="col-span-1">
                                        <Label htmlFor="distance-km"><MapPin className="w-4 h-4 inline-block mr-1 mb-1"/>Distance (km)</Label>
                                        <Input id="distance-km" type="number" value={distanceKm} onChange={e => setDistanceKm(Number(e.target.value))} />
                                    </div>
                                    <div className="col-span-1">
                                        <Label htmlFor="photos-qty"><Box className="w-4 h-4 inline-block mr-1 mb-1"/>Photos Qty</Label>
                                        <Input id="photos-qty" type="number" value={photosQty} onChange={e => setPhotosQty(Number(e.target.value))} />
                                    </div>
                                    <div className="col-span-1">
                                        <Label htmlFor="weight-g"><Weight className="w-4 h-4 inline-block mr-1 mb-1"/>Weight (g)</Label>
                                        <Input id="weight-g" type="number" value={weightG} onChange={e => setWeightG(Number(e.target.value))} />
                                    </div>

                                    {/* --- Seller History --- */}
                                    <div className="col-span-1">
                                        <Label htmlFor="seller-score"><Store className="w-4 h-4 inline-block mr-1 mb-1"/>Seller Avg Score</Label>
                                        <Input id="seller-score" type="number" step="0.01" value={sellerAvgScore} onChange={e => setSellerAvgScore(Number(e.target.value))} />
                                    </div>
                                    <div className="col-span-1">
                                        <Label htmlFor="seller-orders"><Users className="w-4 h-4 inline-block mr-1 mb-1"/>Seller Order Count</Label>
                                        <Input id="seller-orders" type="number" value={sellerOrderCount} onChange={e => setSellerOrderCount(Number(e.target.value))} />
                                    </div>

                                    {/* --- Categorical Input --- */}
                                    <div className="col-span-1">
                                        <Label htmlFor="category-name">Product Category</Label>
                                        <select 
                                            id="category-name" 
                                            value={categoryName} 
                                            onChange={e => setCategoryName(e.target.value)}
                                            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                                        >
                                            {CATEGORY_OPTIONS.map(category => (
                                                <option key={category} value={category}>{category.replace(/_/g, ' ').toUpperCase()}</option>
                                            ))}
                                        </select>
                                    </div>

                                    <div className="col-span-3 mt-2">
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
                    <Card className="border-2 shadow-xl shadow-primary/10">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2"><MessageSquare className="w-5 h-5 text-primary" />Review Sentiment Analyzer (V2)</CardTitle>
                            <CardDescription>Enter a review comment to analyze its sentiment.</CardDescription>
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
