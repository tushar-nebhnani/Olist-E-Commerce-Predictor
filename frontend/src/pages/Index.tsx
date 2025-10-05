import Navigation from "@/components/Navigation";
import DatasetCard from "@/components/DatasetCard";
import { Package, Star, ShoppingCart, CreditCard, Users, List, Store, MapPin, Languages } from "lucide-react";

const Index = () => {
  const datasets = [
    {
      icon: Package,
      title: "Orders",
      description: "This is the transactional backbone of the entire e-commerce platform. Each row represents a single customer order, tracking its journey from initial purchase to final delivery.",
      details: "The dataset contains crucial timestamps for every stage of the order lifecycle: purchase, approval, carrier handoff, and customer delivery. It also includes the order_status (e.g., 'delivered', 'shipped'), delivery estimates, and the customer_id that links it to the customer table. Analyzing this data is essential for evaluating logistics performance, understanding the customer purchase journey, and tracking the overall operational flow of the business.",
    },
    {
      icon: Star,
      title: "Reviews",
      description: "Capturing the voice of the customer, this dataset contains all satisfaction feedback submitted after an order is completed. It is the primary source for measuring customer sentiment.",
      details: "Each review is linked to a unique order and includes a numerical review_score from 1 (very dissatisfied) to 5 (very satisfied), along with optional text comments. This dataset is the target for our primary prediction model, as it allows us to train an algorithm to predict customer satisfaction. Analyzing the scores and comments is vital for identifying service weaknesses, evaluating product quality, and understanding the key drivers of customer happiness or frustration.",
    },
    {
      icon: ShoppingCart,
      title: "Products",
      description: "This dataset serves as the complete inventory catalog, containing detailed information about every unique item available for sale.",
      details: "It provides high-level information like product_category_name and physical attributes such as product_weight_g and product dimensions (length, width, height). This data is fundamental for logistics, as shipping costs and delivery methods often depend on package size and weight. Furthermore, analyzing sales trends by category is a core business activity for managing stock and identifying popular product lines.",
    },
    {
      icon: CreditCard,
      title: "Payments",
      description: "This dataset provides a granular view of the financial aspect of each order, detailing how customers pay for their purchases.",
      details: "It breaks down each transaction, specifying the payment_type (e.g., 'credit_card', 'boleto', 'voucher'), the number of payment_installments, and the total payment_value. Since a single order can have multiple payment methods (e.g., part voucher, part credit card), this dataset is crucial for accurate revenue reporting, analyzing customer payment behavior, and understanding the popularity of different payment options",
    },
    {
      icon: Users,
      title: "Customers",
      description: "This dataset provides the 'who' and 'where' for every order, containing essential geographical and identity information about the customer base.",
      details: "It includes the customer_city, customer_state, and customer_zip_code_prefix. A key feature is the distinction between customer_id (anonymized ID for a single order) and customer_unique_id (anonymized ID for an individual person). Using customer_unique_id allows us to track repeat buyers, which is fundamental for analyzing customer loyalty, retention rates, and the lifetime value of a customer.",
    },
    {
      icon: List,
      title: "Order Items",
      description: "This is the central nervous system of the database. It's a junction table that connects all the other major datasets into a cohesive whole.",
      details: "Each row maps a specific product_id to an order_id, indicating which items were included in which order. It also contains the seller_id who fulfilled that part of the order, along with the price and freight_value for that specific item. Because of its central role, this dataset is the linchpin for nearly any deep analysis, such as calculating total order value, determining top-selling products, or evaluating seller performance.",
    },
    {
      icon: Store,
    title: "Sellers",
    description: "The network of merchants who sell their products on the Olist platform, representing the supply side of the marketplace.",
    details: "This dataset contains the unique `seller_id` for each merchant, along with their geographical location (`seller_city`, `seller_state`). Analyzing this data is essential for evaluating seller performance, tracking sales volume per seller, understanding the geographic distribution of the supply chain, and building any models related to seller quality or delivery efficiency."
  },
  {
    icon: MapPin,
    title: "Geolocation",
    description: "A massive lookup table that maps Brazilian zip codes to precise geographic latitude and longitude coordinates.",
    details: "With over 1 million rows, this is a crucial utility dataset for any spatial analysis. Its primary purpose is to enrich other datasets by providing the `geolocation_lat` and `geolocation_lng` for customer and seller zip codes. This enables powerful features like calculating the physical distance between a seller and a customer, creating heatmaps of sales density, and optimizing logistics."
  },
  {
    icon: Languages,
    title: "Category Name Translation",
    description: "A simple but essential dictionary file that translates the product category names from Portuguese to English.",
    details: "This is a straightforward mapping file that provides the `product_category_name_english` for each Portuguese `product_category_name`. It's a helper dataset used to make reports, visualizations, and the frontend of this very website more accessible and understandable to a non-Portuguese-speaking audience. It's a key part of the data internationalization process."
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
