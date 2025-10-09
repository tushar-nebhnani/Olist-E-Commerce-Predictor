import Navigation from "@/components/Navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Sparkles, ShoppingBag, DollarSign, Package, Loader2, AlertCircle } from "lucide-react";
import { useState } from "react";
import { useToast } from "@/hooks/use-toast";

interface RecommendedProduct {
  product_id: string;
  product_category_name: string;
  price: number;
}

const ProductRecommendations = () => {
  const [customerId, setCustomerId] = useState("");
  const [recommendations, setRecommendations] = useState<RecommendedProduct[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  const handleGetRecommendations = async () => {
    if (!customerId.trim()) {
      toast({
        title: "Customer ID Required",
        description: "Please enter a customer ID to get recommendations.",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    setError(null);
    setRecommendations([]);

    try {
      const response = await fetch(`http://127.0.0.1:8000/recommend/${customerId.trim()}`);
      
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error("Customer not found. Please check the customer ID and try again.");
        }
        throw new Error(`Failed to fetch recommendations: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.recommended_products && data.recommended_products.length > 0) {
        setRecommendations(data.recommended_products);
        toast({
          title: "Success!",
          description: `Found ${data.recommended_products.length} product recommendations.`,
        });
      } else {
        setError("No recommendations found for this customer.");
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "An unexpected error occurred.";
      setError(errorMessage);
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleGetRecommendations();
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navigation />
      
      <main className="container mx-auto px-4 pt-24 pb-16">
        <div className="max-w-7xl mx-auto space-y-12 w-full">
          {/* Hero Section */}
          <div className="relative">
            {/* Decorative background elements */}
            <div className="absolute inset-0 -z-10 overflow-hidden">
              <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl animate-pulse"></div>
              <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-accent/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
            </div>

            <div className="text-center space-y-4">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
                <Sparkles className="w-8 h-8 text-primary" />
              </div>
              <h1 className="text-5xl font-bold tracking-tight bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                Product Recommendations
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
                AI-powered personalized product suggestions based on customer purchase history and behavior patterns.
              </p>
            </div>
          </div>

          {/* Input Section */}
          <Card className="max-w-2xl mx-auto bg-gradient-to-br from-card to-card/50 backdrop-blur border-border/50 hover:shadow-2xl hover:border-primary/50 transition-all">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Package className="w-5 h-5 text-primary" />
                Get Customer Recommendations
              </CardTitle>
              <CardDescription>
                Enter a customer ID to discover personalized product recommendations
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-3">
                <Input
                  type="text"
                  placeholder="Enter customer ID (e.g., abc123...)"
                  value={customerId}
                  onChange={(e) => setCustomerId(e.target.value)}
                  onKeyPress={handleKeyPress}
                  disabled={loading}
                  className="flex-1"
                />
                <Button 
                  onClick={handleGetRecommendations} 
                  disabled={loading}
                  className="min-w-[180px]"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Loading...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4 mr-2" />
                      Get Recommendations
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Error Display */}
          {error && (
            <Card className="max-w-2xl mx-auto border-destructive/50 bg-destructive/5">
              <CardContent className="pt-6">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-destructive mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-destructive mb-1">Error</h3>
                    <p className="text-sm text-muted-foreground">{error}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Recommendations Display */}
          {recommendations.length > 0 && (
            <div className="space-y-6">
              <div className="text-center">
                <h2 className="text-3xl font-bold mb-2">
                  Recommended Products
                </h2>
                <p className="text-muted-foreground">
                  {recommendations.length} personalized recommendations for this customer
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {recommendations.map((product, index) => (
                  <Card 
                    key={`${product.product_id}-${index}`}
                    className="bg-gradient-to-br from-card to-card/50 backdrop-blur border-border/50 hover:shadow-2xl hover:border-primary/50 hover:scale-[1.02] transition-all group"
                  >
                    <CardHeader>
                      <div className="flex items-start justify-between mb-2">
                        <div className="p-2 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
                          <ShoppingBag className="w-5 h-5 text-primary" />
                        </div>
                        <Badge variant="secondary" className="text-xs">
                          #{index + 1}
                        </Badge>
                      </div>
                      <CardTitle className="text-lg capitalize">
                        {product.product_category_name.replace(/_/g, ' ')}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex items-center justify-between p-3 rounded-lg bg-gradient-to-br from-muted to-muted/50 border border-border/50">
                        <div className="flex items-center gap-2">
                          <DollarSign className="w-4 h-4 text-primary" />
                          <span className="text-sm text-muted-foreground">Price</span>
                        </div>
                        <span className="font-bold text-primary text-lg">
                          ${product.price.toFixed(2)}
                        </span>
                      </div>
                      <div className="p-2 rounded-md bg-muted/50 border border-border/30">
                        <p className="text-xs text-muted-foreground mb-1">Product ID</p>
                        <p className="text-xs font-mono break-all">{product.product_id}</p>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* Empty State */}
          {!loading && !error && recommendations.length === 0 && (
            <Card className="max-w-2xl mx-auto bg-gradient-to-br from-muted/30 to-muted/10 border-dashed border-2">
              <CardContent className="pt-12 pb-12 text-center">
                <ShoppingBag className="w-16 h-16 mx-auto mb-4 text-muted-foreground/50" />
                <h3 className="text-xl font-semibold mb-2">No Recommendations Yet</h3>
                <p className="text-muted-foreground">
                  Enter a customer ID above to get personalized product recommendations
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
};

export default ProductRecommendations;
