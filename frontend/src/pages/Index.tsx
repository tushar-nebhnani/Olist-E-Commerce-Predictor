import { useState } from "react";
import { Package, Star, ShoppingCart, CreditCard, Users, List, CheckCircle, XCircle, Loader2, Edit } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

type Page = "home" | "v1" | "v2";

interface DatasetCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  details: string;
  customDetails: string;
  onCustomDetailsChange: (details: string) => void;
}

const DatasetCard = ({ icon, title, description, details, customDetails, onCustomDetailsChange }: DatasetCardProps) => {
  const [isHovered, setIsHovered] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [tempDetails, setTempDetails] = useState(customDetails);

  const handleSave = () => {
    onCustomDetailsChange(tempDetails);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setTempDetails(customDetails);
    setIsEditing(false);
  };

  return (
    <div
      className="relative bg-card border-2 border-border rounded-xl p-6 transition-all duration-300 hover:scale-[1.03] hover:border-primary overflow-hidden shadow-lg hover:shadow-2xl hover:shadow-primary/20"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => !isEditing && setIsHovered(false)}
    >
      <div className="relative z-10">
        <div className="text-primary mb-4 bg-primary/10 w-fit p-3 rounded-lg">{icon}</div>
        <h3 className="text-xl font-bold text-foreground mb-2">{title}</h3>
        <p className="text-muted-foreground text-sm">{description}</p>
      </div>
      
      {isHovered && !isEditing && (
        <div className="absolute inset-0 bg-gradient-to-br from-primary/15 to-primary/5 backdrop-blur-sm flex flex-col items-center justify-center p-6 animate-fade-in z-20">
          <div className="mb-4 text-center">
            <p className="text-foreground text-sm font-semibold mb-2">{details}</p>
            {customDetails && (
              <div className="mt-3 pt-3 border-t border-primary/20">
                <p className="text-xs text-muted-foreground font-medium mb-1">Additional Details:</p>
                <p className="text-foreground text-xs whitespace-pre-wrap">{customDetails}</p>
              </div>
            )}
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={(e) => {
              e.stopPropagation();
              setIsEditing(true);
            }}
            className="gap-2"
          >
            <Edit size={14} />
            Add Details
          </Button>
        </div>
      )}

      {isEditing && (
        <div className="absolute inset-0 bg-card/95 backdrop-blur-sm flex flex-col p-4 animate-fade-in z-30">
          <h4 className="text-sm font-bold text-foreground mb-2">Add details for {title}</h4>
          <p className="text-xs text-muted-foreground mb-3">Add information about outliers, data quality, etc.</p>
          <Textarea
            value={tempDetails}
            onChange={(e) => setTempDetails(e.target.value)}
            placeholder="e.g., Contains 2% outliers in price column, Missing values handled..."
            className="flex-1 mb-3 text-xs"
            autoFocus
          />
          <div className="flex gap-2">
            <Button size="sm" onClick={handleSave} className="flex-1">Save</Button>
            <Button size="sm" variant="outline" onClick={handleCancel} className="flex-1">Cancel</Button>
          </div>
        </div>
      )}
    </div>
  );
};

const HomePage = () => {
  const [customDetails, setCustomDetails] = useState<Record<string, string>>({});
  
  const datasets = [
    {
      icon: <Package size={32} />,
      title: "Orders",
      description: "Core dataset with all order information.",
      details: "Rows: 99,441, Key Columns: order_id, customer_id, order_status, purchase_timestamp"
    },
    {
      icon: <Star size={32} />,
      title: "Reviews",
      description: "Contains customer review scores from 1 to 5.",
      details: "Rows: 99,224, Key Columns: review_id, order_id, review_score"
    },
    {
      icon: <ShoppingCart size={32} />,
      title: "Products",
      description: "Information about the products sold.",
      details: "Rows: 32,951, Key Columns: product_id, product_category_name, product_weight_g"
    },
    {
      icon: <CreditCard size={32} />,
      title: "Payments",
      description: "Details about order payment methods.",
      details: "Rows: 103,886, Key Columns: order_id, payment_type, payment_installments"
    },
    {
      icon: <Users size={32} />,
      title: "Customers",
      description: "Customer location and unique ID information.",
      details: "Rows: 99,441, Key Columns: customer_id, customer_unique_id, customer_city"
    },
    {
      icon: <List size={32} />,
      title: "Order Items",
      description: "Links orders with products and sellers.",
      details: "Rows: 112,650, Key Columns: order_id, product_id, seller_id, price"
    }
  ];

  const handleCustomDetailsChange = (title: string, details: string) => {
    setCustomDetails(prev => ({ ...prev, [title]: details }));
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-foreground mb-4">Olist E-commerce Datasets</h1>
        <p className="text-muted-foreground text-lg">
          Explore the key datasets powering our machine learning models
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {datasets.map((dataset, index) => (
          <DatasetCard 
            key={index} 
            {...dataset}
            customDetails={customDetails[dataset.title] || ""}
            onCustomDetailsChange={(details) => handleCustomDetailsChange(dataset.title, details)}
          />
        ))}
      </div>
    </div>
  );
};

interface V2FormData {
  price: number;
  freight_value: number;
  delivery_time_days: number;
  estimated_vs_actual_delivery: number;
  payment_installments: number;
  payment_value: number;
  product_photos_qty: number;
  product_weight_g: number;
  product_category_name: string;
}

const SatisfactionV2Page = () => {
  const [formData, setFormData] = useState<V2FormData>({
    price: 100,
    freight_value: 15,
    delivery_time_days: 10,
    estimated_vs_actual_delivery: 0,
    payment_installments: 1,
    payment_value: 100,
    product_photos_qty: 1,
    product_weight_g: 1000,
    product_category_name: "cama_mesa_banho"
  });
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ satisfied: boolean; probability: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const response = await fetch("http://localhost:8000/satisfaction/v2/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) {
        throw new Error("Failed to get prediction");
      }
      
      const data = await response.json();
      setResult({
        satisfied: data.satisfied,
        probability: data.satisfaction_probability
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      <h1 className="text-3xl font-bold text-foreground mb-8">
        Predict Customer Satisfaction (V2 - Improved Model)
      </h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="bg-card border border-border rounded-lg p-6">
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Price: ${formData.price}
              </label>
              <input
                type="range"
                min="10"
                max="500"
                value={formData.price}
                onChange={(e) => setFormData({ ...formData, price: Number(e.target.value) })}
                className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Freight Value: ${formData.freight_value}
              </label>
              <input
                type="range"
                min="5"
                max="100"
                value={formData.freight_value}
                onChange={(e) => setFormData({ ...formData, freight_value: Number(e.target.value) })}
                className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Delivery Time (days)
              </label>
              <input
                type="number"
                value={formData.delivery_time_days}
                onChange={(e) => setFormData({ ...formData, delivery_time_days: Number(e.target.value) })}
                className="w-full bg-input border border-border rounded-md px-4 py-2 text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Estimated vs Actual Delivery (days)
              </label>
              <input
                type="number"
                value={formData.estimated_vs_actual_delivery}
                onChange={(e) => setFormData({ ...formData, estimated_vs_actual_delivery: Number(e.target.value) })}
                className="w-full bg-input border border-border rounded-md px-4 py-2 text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Payment Installments: {formData.payment_installments}
              </label>
              <input
                type="range"
                min="1"
                max="24"
                value={formData.payment_installments}
                onChange={(e) => setFormData({ ...formData, payment_installments: Number(e.target.value) })}
                className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Payment Value
              </label>
              <input
                type="number"
                value={formData.payment_value}
                onChange={(e) => setFormData({ ...formData, payment_value: Number(e.target.value) })}
                className="w-full bg-input border border-border rounded-md px-4 py-2 text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Product Photos: {formData.product_photos_qty}
              </label>
              <input
                type="range"
                min="1"
                max="20"
                value={formData.product_photos_qty}
                onChange={(e) => setFormData({ ...formData, product_photos_qty: Number(e.target.value) })}
                className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Product Weight (g)
              </label>
              <input
                type="number"
                value={formData.product_weight_g}
                onChange={(e) => setFormData({ ...formData, product_weight_g: Number(e.target.value) })}
                className="w-full bg-input border border-border rounded-md px-4 py-2 text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Product Category
              </label>
              <select
                value={formData.product_category_name}
                onChange={(e) => setFormData({ ...formData, product_category_name: e.target.value })}
                className="w-full bg-input border border-border rounded-md px-4 py-2 text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="cama_mesa_banho">Cama Mesa Banho</option>
                <option value="beleza_saude">Beleza Saúde</option>
                <option value="esporte_lazer">Esporte Lazer</option>
                <option value="informatica_acessorios">Informática Acessórios</option>
                <option value="moveis_decoracao">Móveis Decoração</option>
              </select>
            </div>

            <button
              onClick={handlePredict}
              disabled={loading}
              className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-3 px-6 rounded-md transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <Loader2 className="animate-spin mr-2" size={20} />
                  Predicting...
                </span>
              ) : (
                "Predict Satisfaction"
              )}
            </button>
          </div>
        </div>

        {/* Result Display */}
        <div className="bg-card border border-border rounded-lg p-6 flex items-center justify-center">
          {!result && !error && !loading && (
            <p className="text-muted-foreground text-center">
              Enter order details and click predict to see the result.
            </p>
          )}

          {loading && (
            <div className="text-center">
              <Loader2 className="animate-spin mx-auto mb-4 text-primary" size={48} />
              <p className="text-muted-foreground">Analyzing order data...</p>
            </div>
          )}

          {error && (
            <div className="text-center">
              <XCircle className="mx-auto mb-4 text-destructive" size={48} />
              <p className="text-destructive font-semibold mb-2">Prediction Failed</p>
              <p className="text-muted-foreground text-sm">{error}</p>
            </div>
          )}

          {result && (
            <div className="text-center w-full">
              {result.satisfied ? (
                <CheckCircle className="mx-auto mb-4 text-success" size={64} />
              ) : (
                <XCircle className="mx-auto mb-4 text-destructive" size={64} />
              )}
              
              <h2 className={`text-3xl font-bold mb-6 ${result.satisfied ? "text-success" : "text-destructive"}`}>
                {result.satisfied ? "Prediction: Satisfied" : "Prediction: Not Satisfied"}
              </h2>
              
              <div className="mb-4">
                <p className="text-muted-foreground text-sm mb-2">Confidence Score</p>
                <div className="w-full bg-muted rounded-full h-4 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      result.satisfied ? "bg-success" : "bg-destructive"
                    }`}
                    style={{ width: `${result.probability * 100}%` }}
                  />
                </div>
                <p className="text-foreground font-bold text-2xl mt-2">
                  {(result.probability * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

interface V1FormData {
  price: number;
  freight_value: number;
  delivery_time_days: number;
  estimated_vs_actual_delivery: number;
}

const SatisfactionV1Page = () => {
  const [formData, setFormData] = useState<V1FormData>({
    price: 100,
    freight_value: 15,
    delivery_time_days: 10,
    estimated_vs_actual_delivery: 0
  });
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ satisfied: boolean; probability: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const response = await fetch("http://localhost:8000/satisfaction/v1/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) {
        throw new Error("Failed to get prediction");
      }
      
      const data = await response.json();
      setResult({
        satisfied: data.satisfied,
        probability: data.satisfaction_probability
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      <h1 className="text-3xl font-bold text-foreground mb-8">
        Predict Customer Satisfaction (V1 - Base Model)
      </h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="bg-card border border-border rounded-lg p-6">
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Price: ${formData.price}
              </label>
              <input
                type="range"
                min="10"
                max="500"
                value={formData.price}
                onChange={(e) => setFormData({ ...formData, price: Number(e.target.value) })}
                className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Freight Value: ${formData.freight_value}
              </label>
              <input
                type="range"
                min="5"
                max="100"
                value={formData.freight_value}
                onChange={(e) => setFormData({ ...formData, freight_value: Number(e.target.value) })}
                className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Delivery Time (days)
              </label>
              <input
                type="number"
                value={formData.delivery_time_days}
                onChange={(e) => setFormData({ ...formData, delivery_time_days: Number(e.target.value) })}
                className="w-full bg-input border border-border rounded-md px-4 py-2 text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Estimated vs Actual Delivery (days)
              </label>
              <input
                type="number"
                value={formData.estimated_vs_actual_delivery}
                onChange={(e) => setFormData({ ...formData, estimated_vs_actual_delivery: Number(e.target.value) })}
                className="w-full bg-input border border-border rounded-md px-4 py-2 text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>

            <button
              onClick={handlePredict}
              disabled={loading}
              className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-3 px-6 rounded-md transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <Loader2 className="animate-spin mr-2" size={20} />
                  Predicting...
                </span>
              ) : (
                "Predict Satisfaction"
              )}
            </button>
          </div>
        </div>

        {/* Result Display */}
        <div className="bg-card border border-border rounded-lg p-6 flex items-center justify-center">
          {!result && !error && !loading && (
            <p className="text-muted-foreground text-center">
              Enter order details and click predict to see the result.
            </p>
          )}

          {loading && (
            <div className="text-center">
              <Loader2 className="animate-spin mx-auto mb-4 text-primary" size={48} />
              <p className="text-muted-foreground">Analyzing order data...</p>
            </div>
          )}

          {error && (
            <div className="text-center">
              <XCircle className="mx-auto mb-4 text-destructive" size={48} />
              <p className="text-destructive font-semibold mb-2">Prediction Failed</p>
              <p className="text-muted-foreground text-sm">{error}</p>
            </div>
          )}

          {result && (
            <div className="text-center w-full">
              {result.satisfied ? (
                <CheckCircle className="mx-auto mb-4 text-success" size={64} />
              ) : (
                <XCircle className="mx-auto mb-4 text-destructive" size={64} />
              )}
              
              <h2 className={`text-3xl font-bold mb-6 ${result.satisfied ? "text-success" : "text-destructive"}`}>
                {result.satisfied ? "Prediction: Satisfied" : "Prediction: Not Satisfied"}
              </h2>
              
              <div className="mb-4">
                <p className="text-muted-foreground text-sm mb-2">Confidence Score</p>
                <div className="w-full bg-muted rounded-full h-4 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      result.satisfied ? "bg-success" : "bg-destructive"
                    }`}
                    style={{ width: `${result.probability * 100}%` }}
                  />
                </div>
                <p className="text-foreground font-bold text-2xl mt-2">
                  {(result.probability * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const Index = () => {
  const [currentPage, setCurrentPage] = useState<Page>("home");

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation Bar */}
      <nav className="sticky top-0 z-50 bg-card border-b border-border backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-bold text-foreground">Olist ML Insights Dashboard</h1>
            
            <div className="flex gap-6">
              <button
                onClick={() => setCurrentPage("home")}
                className={`text-sm font-medium transition-colors duration-200 ${
                  currentPage === "home"
                    ? "text-primary border-b-2 border-primary"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Home
              </button>
              <button
                onClick={() => setCurrentPage("v1")}
                className={`text-sm font-medium transition-colors duration-200 ${
                  currentPage === "v1"
                    ? "text-primary border-b-2 border-primary"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Satisfaction Predictor (V1)
              </button>
              <button
                onClick={() => setCurrentPage("v2")}
                className={`text-sm font-medium transition-colors duration-200 ${
                  currentPage === "v2"
                    ? "text-primary border-b-2 border-primary"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Satisfaction Predictor (V2)
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Page Content */}
      <main>
        {currentPage === "home" && <HomePage />}
        {currentPage === "v1" && <SatisfactionV1Page />}
        {currentPage === "v2" && <SatisfactionV2Page />}
      </main>
    </div>
  );
};

export default Index;
