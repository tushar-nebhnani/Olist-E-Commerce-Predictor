// File: frontend/src/.../ProductRecommendation.tsx

import { useState } from "react";
import Navigation from "@/components/Navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";

// Import Popover components and new icons
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Sparkles, ShoppingBag, DollarSign, Package, Loader2, AlertCircle, Weight, Image as ImageIcon, Ruler } from "lucide-react";

// Updated interface to include all details from the dataset
interface RecommendedProduct {
  product_id: string;
  product_category_name_english: string;
  price: number;
  product_name_length?: number;
  product_description_length?: number;
  product_photos_qty?: number;
  product_weight_g?: number;
  product_length_cm?: number;
  product_height_cm?: number;
  product_width_cm?: number;
}

const sampleCustomerIds = [
  "00012a2ce6f8dcda20d059ce98491703",
  "0a3637b5a15322b77d6205b382e8d38b",
  "0a7e868a245f0858e7232c3f8e56b461",
];

const ProductRecommendations = () => {
  const [customerId, setCustomerId] = useState("");
  const [recommendations, setRecommendations] = useState<RecommendedProduct[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  const handleGetRecommendations = async (idToFetch?: string) => {
    const finalCustomerId = (idToFetch || customerId).trim();
    if (!finalCustomerId) {
      toast({ title: "Customer ID Required", description: "Please enter a customer ID.", variant: "destructive" });
      return;
    }
    setLoading(true);
    setError(null);
    setRecommendations([]);
    try {
      const response = await fetch(`http://127.0.0.1:8000/recommendation/${finalCustomerId}`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to fetch: ${response.statusText}`);
      }
      const data = await response.json();
      if (data.recommended_products && data.recommended_products.length > 0) {
        setRecommendations(data.recommended_products);
        toast({ title: "Success!", description: `Found ${data.recommended_products.length} recommendations.` });
      } else {
        setError("No recommendations found for this customer.");
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "An unexpected error occurred.";
      setError(msg);
      toast({ title: "Error", description: msg, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') handleGetRecommendations();
  };

  const handleSampleClick = (sampleId: string) => {
    setCustomerId(sampleId);
    handleGetRecommendations(sampleId);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navigation />
      <main className="container mx-auto px-4 pt-24 pb-16">
        <div className="max-w-7xl mx-auto space-y-12 w-full">
          {/* Hero Section */}
          <div className="text-center space-y-4">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
              <Sparkles className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-5xl font-bold tracking-tight">Product Recommendations</h1>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              AI-powered personalized product suggestions for customers.
            </p>
          </div>

          {/* Input Section */}
          <Card className="max-w-2xl mx-auto">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">Get Customer Recommendations</CardTitle>
              <CardDescription>Enter a customer ID or choose an example to discover personalized products.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-3">
                <Input type="text" placeholder="Enter customer ID..." value={customerId} onChange={(e) => setCustomerId(e.target.value)} onKeyPress={handleKeyPress} disabled={loading} className="flex-1" />
                <Button onClick={() => handleGetRecommendations()} disabled={loading} className="min-w-[180px]">
                  {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Loading...</> : <><Sparkles className="w-4 h-4 mr-2" />Get Recommendations</>}
                </Button>
              </div>
              <div className="flex items-center gap-2 pt-2">
                <p className="text-sm text-muted-foreground">Try an example:</p>
                {sampleCustomerIds.map(id => (
                  <Badge key={id} variant="outline" onClick={() => handleSampleClick(id)} className="cursor-pointer hover:bg-accent">{id.substring(0, 8)}...</Badge>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Error Display */}
          {error && (
            <Card className="max-w-2xl mx-auto border-destructive/50 bg-destructive/5">
              <CardContent className="pt-6">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-destructive mt-0.5" />
                  <div><h3 className="font-semibold text-destructive mb-1">An Error Occurred</h3><p className="text-sm text-muted-foreground">{error}</p></div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Recommendations Display */}
          {recommendations.length > 0 && (
            <div className="space-y-6">
              <div className="text-center"><h2 className="text-3xl font-bold mb-2">Recommended For You</h2></div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {recommendations.map((product, index) => (
                  <Popover key={`${product.product_id}-${index}`}>
                    <PopoverTrigger asChild>
                      <Card className="cursor-pointer hover:shadow-2xl hover:border-primary/50 hover:scale-[1.02] transition-all group">
                        <CardHeader>
                          <div className="flex items-start justify-between mb-2">
                            <ShoppingBag className="w-5 h-5 text-primary" />
                            <Badge variant="secondary">#{index + 1}</Badge>
                          </div>
                          <CardTitle className="text-lg capitalize truncate">
                            {product.product_category_name_english?.replace(/_/g, ' ') ?? 'Unknown Category'}
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2"><DollarSign className="w-4 h-4" /><span className="text-sm">Price</span></div>
                            <span className="font-bold text-lg">${product.price?.toFixed(2) ?? '0.00'}</span>
                          </div>
                        </CardContent>
                      </Card>
                    </PopoverTrigger>
                    <PopoverContent className="w-80" side="top" align="center">
                      <div className="space-y-4">
                        <h4 className="font-semibold leading-none">Product Details</h4>
                        <div className="text-sm text-muted-foreground space-y-2">
                          <p className="font-mono text-xs bg-muted p-2 rounded-md break-all">ID: {product.product_id}</p>
                          <div className="flex items-center justify-between">
                            <span className="flex items-center gap-2"><Weight className="w-4 h-4" /> Weight</span>
                            <span>{product.product_weight_g ?? 0} g</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="flex items-center gap-2"><ImageIcon className="w-4 h-4" /> Photos</span>
                            <span>{product.product_photos_qty ?? 0}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="flex items-center gap-2"><Ruler className="w-4 h-4" /> Dimensions</span>
                            <span>{product.product_length_cm ?? 0}x{product.product_height_cm ?? 0}x{product.product_width_cm ?? 0} cm</span>
                          </div>
                        </div>
                      </div>
                    </PopoverContent>
                  </Popover>
                ))}
              </div>
            </div>
          )}
          
          {/* Empty State */}
          {!loading && !error && recommendations.length === 0 && (
             <Card className="max-w-2xl mx-auto bg-muted/20 border-dashed border-2">
               <CardContent className="pt-12 pb-12 text-center">
                 <ShoppingBag className="w-16 h-16 mx-auto mb-4 text-muted-foreground/50" />
                 <h3 className="text-xl font-semibold mb-2">No Recommendations Yet</h3>
                 <p className="text-muted-foreground">Enter a customer ID above to see the magic happen!</p>
               </CardContent>
             </Card>
           )}
        </div>
      </main>
    </div>
  );
};

export default ProductRecommendations;