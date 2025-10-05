import Navigation from "@/components/Navigation";
import DatasetCard from "@/components/DatasetCard";
import { Package, Star, ShoppingCart, CreditCard, Users, List } from "lucide-react";

const Index = () => {
  const datasets = [
    {
      icon: Package,
      title: "Orders",
      description: "Core dataset with all order information.",
      details: "Comprehensive order data including timestamps, status, delivery information, and customer associations. Essential for understanding purchase patterns and order flow.",
    },
    {
      icon: Star,
      title: "Reviews",
      description: "Contains customer review scores from 1 to 5.",
      details: "Customer satisfaction ratings and review text for each order. Crucial for sentiment analysis and customer satisfaction prediction models.",
    },
    {
      icon: ShoppingCart,
      title: "Products",
      description: "Information about the products sold.",
      details: "Product catalog including categories, dimensions, weights, and descriptions. Used for product performance analysis and recommendation systems.",
    },
    {
      icon: CreditCard,
      title: "Payments",
      description: "Details about order payment methods.",
      details: "Payment transaction data including methods, installments, and amounts. Important for financial analysis and fraud detection.",
    },
    {
      icon: Users,
      title: "Customers",
      description: "Customer location and unique ID information.",
      details: "Customer demographic data including geographic location, zip codes, and unique identifiers. Essential for customer segmentation and regional analysis.",
    },
    {
      icon: List,
      title: "Order Items",
      description: "Links orders with products and sellers.",
      details: "Detailed line items for each order including product quantities, prices, freight costs, and seller information. Critical for revenue analysis.",
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="container mx-auto px-6 pt-24 pb-16">
        <div className="text-center mb-16 space-y-4">
          <h2 className="text-5xl font-bold tracking-tight">
            Olist E-commerce Datasets
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Explore the key datasets powering our machine learning models
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {datasets.map((dataset) => (
            <DatasetCard key={dataset.title} {...dataset} />
          ))}
        </div>
      </main>
    </div>
  );
};

export default Index;
