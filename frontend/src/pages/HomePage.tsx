import { useState } from "react";
import { Link } from "react-router-dom";
import { Package, Star, ShoppingCart, CreditCard, Users, List } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const datasets = [
  {
    id: "orders",
    title: "Orders",
    description: "Core dataset with all order information.",
    fullDescription: "This comprehensive dataset contains all order-related information including order IDs, timestamps, customer references, delivery status, and order metadata. It serves as the backbone for analyzing order patterns, delivery performance, and customer purchasing behavior.",
    icon: Package,
  },
  {
    id: "reviews",
    title: "Reviews",
    description: "Contains customer review scores from 1 to 5.",
    fullDescription: "Customer feedback dataset containing review scores ranging from 1 to 5 stars, along with review comments, timestamps, and associated order references. This data enables sentiment analysis, product quality assessment, and customer satisfaction tracking.",
    icon: Star,
  },
  {
    id: "products",
    title: "Products",
    description: "Information about the products sold.",
    fullDescription: "Complete product catalog information including product names, categories, descriptions, dimensions, weights, and photos. This dataset is essential for inventory management, product performance analysis, and category-wise sales insights.",
    icon: ShoppingCart,
  },
  {
    id: "payments",
    title: "Payments",
    description: "Details about order payment methods.",
    fullDescription: "Payment transaction data including payment types (credit card, debit, boleto, voucher), payment values, installment information, and transaction status. Critical for financial analysis, payment method preferences, and revenue tracking.",
    icon: CreditCard,
  },
  {
    id: "customers",
    title: "Customers",
    description: "Customer location and unique ID information.",
    fullDescription: "Customer profile data containing unique customer IDs, geographical location (city, state, zip code), and regional information. Enables customer segmentation, geographical analysis, and targeted marketing strategies.",
    icon: Users,
  },
  {
    id: "order-items",
    title: "Order Items",
    description: "Links orders with products and sellers.",
    fullDescription: "Junction dataset that connects orders with specific products and sellers, including item quantities, prices, freight values, and seller information. Essential for marketplace analytics, seller performance tracking, and detailed order composition analysis.",
    icon: List,
  },
];

const HomePage = () => {
  const [selectedDataset, setSelectedDataset] = useState<typeof datasets[0] | null>(null);

  return (
    <div className="min-h-screen bg-[#0f172a]">
      {/* Header */}
      <header className="border-b border-primary/20 bg-gradient-to-r from-background via-background to-primary/5 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
              Olist ML Insights Dashboard
            </h1>
            <nav className="flex gap-6">
              <Link to="/" className="text-primary font-semibold hover:text-primary/80 transition-all duration-300 hover:scale-110">
                Home
              </Link>
              <Link to="/satisfaction-v1" className="text-muted-foreground hover:text-primary transition-all duration-300 hover:scale-110">
                Satisfaction Predictor (V1)
              </Link>
              <Link to="/satisfaction-v2" className="text-muted-foreground hover:text-primary transition-all duration-300 hover:scale-110">
                Satisfaction Predictor (V2)
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-16">
        <div className="text-center mb-12">
          <h2 className="text-5xl font-bold text-foreground mb-4">Olist E-commerce Datasets</h2>
          <p className="text-xl text-muted-foreground">
            Explore the key datasets powering our machine learning models
          </p>
        </div>

        {/* Dataset Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
          {datasets.map((dataset) => {
            const Icon = dataset.icon;
            return (
              <Card
                key={dataset.id}
                className="group cursor-pointer transition-all duration-500 hover:scale-110 hover:shadow-2xl hover:shadow-primary/20 bg-gradient-to-br from-card/80 to-card/40 backdrop-blur-sm border-border/50 hover:border-primary relative overflow-hidden animate-fade-in"
                onMouseEnter={() => setSelectedDataset(dataset)}
              >
                {/* Animated background gradient */}
                <div className="absolute inset-0 bg-gradient-to-br from-primary/0 via-primary/0 to-primary/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                
                <CardHeader className="relative z-10">
                  <div className="mb-4 transform transition-all duration-500 group-hover:scale-110 group-hover:rotate-12">
                    <Icon className="w-12 h-12 text-primary group-hover:text-primary/80 drop-shadow-lg" strokeWidth={1.5} />
                  </div>
                  <CardTitle className="text-2xl group-hover:text-primary transition-colors duration-300">
                    {dataset.title}
                  </CardTitle>
                  <CardDescription className="text-base group-hover:text-foreground/80 transition-colors duration-300">
                    {dataset.description}
                  </CardDescription>
                </CardHeader>
                
                {/* Hover indicator */}
                <div className="absolute bottom-2 right-2 text-xs text-primary/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                  Hover to explore
                </div>
              </Card>
            );
          })}
        </div>
      </main>

      {/* Modal Dialog */}
      <Dialog open={!!selectedDataset} onOpenChange={() => setSelectedDataset(null)}>
        <DialogContent className="max-w-2xl bg-gradient-to-br from-card via-card to-card/80 border-primary/30 shadow-2xl shadow-primary/10 animate-scale-in">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-3 text-3xl bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
              {selectedDataset && <selectedDataset.icon className="w-10 h-10 text-primary animate-pulse" strokeWidth={1.5} />}
              {selectedDataset?.title}
            </DialogTitle>
            <DialogDescription className="text-base pt-6 leading-relaxed text-foreground/90">
              {selectedDataset?.fullDescription}
            </DialogDescription>
          </DialogHeader>
          
          {/* Decorative elements */}
          <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-full blur-3xl -z-10" />
          <div className="absolute bottom-0 left-0 w-32 h-32 bg-primary/5 rounded-full blur-3xl -z-10" />
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default HomePage;
