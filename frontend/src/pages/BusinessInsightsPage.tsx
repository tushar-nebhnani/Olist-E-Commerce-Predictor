import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { 
  DollarSign, MapPin, Star, Truck, ShoppingBag, CreditCard, TrendingUp, Users, Store, 
  BarChartHorizontal, Clock, CheckCircle, Package, CalendarDays, Hourglass, AlertCircle, Lightbulb, Repeat, Sparkles, UserX, UserCheck
} from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, BarChart, CartesianGrid, XAxis, YAxis, Tooltip as RechartsTooltip, Bar as RechartsBar } from 'recharts';
import Navigation from "@/components/Navigation";

const rawSegmentData = {
    "0": { "persona": "Needs Attention", "size": 10828, "percentage": "11.6%", "avg_recency_days": 126.1, "avg_frequency": 1.0, "avg_monetary": 307.23, "avg_tenure_days": 126.1, "avg_interpurchase_time_days": 0.0 },
    "1": { "persona": "Needs Attention", "size": 31856, "percentage": "34.1%", "avg_recency_days": 376.1, "avg_frequency": 1.0, "avg_monetary": 84.73, "avg_tenure_days": 376.1, "avg_interpurchase_time_days": 0.0 },
    "2": { "persona": "Needs Attention", "size": 1792, "percentage": "1.9%", "avg_recency_days": 191.4, "avg_frequency": 2.0, "avg_monetary": 295.75, "avg_tenure_days": 309.0, "avg_interpurchase_time_days": 117.1 },
    "3": { "persona": "Needs Attention", "size": 780, "percentage": "0.8%", "avg_recency_days": 291.8, "avg_frequency": 2.0, "avg_monetary": 276.08, "avg_tenure_days": 291.9, "avg_interpurchase_time_days": 0.0 },
    "4": { "persona": "Potential Loyalists", "size": 15301, "percentage": "16.4%", "avg_recency_days": 321.0, "avg_frequency": 1.0, "avg_monetary": 333.55, "avg_tenure_days": 321.0, "avg_interpurchase_time_days": 0.0 },
    "5": { "persona": "At-Risk", "size": 13121, "percentage": "14.1%", "avg_recency_days": 33.0, "avg_frequency": 1.0, "avg_monetary": 149.32, "avg_tenure_days": 33.0, "avg_interpurchase_time_days": 0.0 },
    "6": { "persona": "Needs Attention", "size": 19429, "percentage": "20.8%", "avg_recency_days": 149.1, "avg_frequency": 1.0, "avg_monetary": 70.63, "avg_tenure_days": 149.1, "avg_interpurchase_time_days": 0.0 },
    "7": { "persona": "Champions", "size": 228, "percentage": "0.2%", "avg_recency_days": 202.3, "avg_frequency": 3.4, "avg_monetary": 506.79, "avg_tenure_days": 359.9, "avg_interpurchase_time_days": 68.7 }
};

// --- Segment Processing and Color Mapping ---
const SEGMENT_COLORS_RAW = [
  'hsl(142 71% 45%)',    // 0: Green (Champions)
  'hsl(48 96% 50%)',     // 1: Yellow (Potential Loyalists)
  'hsl(0 84% 60%)',      // 2: Red (At-Risk)
  'hsl(210 40% 70%)',    // 3: Light Blue (High Value/Lapsed)
  'hsl(210 60% 50%)',    // 4: Mid Blue (Low Value/New Opportunity - Focus)
  'hsl(210 80% 30%)',    // 5: Dark Blue (Dormant/Churn Risk - Largest Segment)
  'hsl(240 40% 60%)',    // 6: Purple/Indigo (Multi-Purchase/Dormant)
  'hsl(210 40% 10%)',    // 7: Catch-All
];

const segmentData = Object.keys(rawSegmentData).map((key, index) => {
    const segment = rawSegmentData[key];
    
    // --- Initial Color Mapping (Priority based) ---
    let colorIndex = index; // Default to index for unique colors
    // Assign primary colors based on recognized personas from backend scoring logic
    if (segment.persona === 'Champions') colorIndex = 0;
    else if (segment.persona === 'Potential Loyalists') colorIndex = 1;
    else if (segment.persona === 'At-Risk') colorIndex = 2;
    
    // Assign a clearer name for the display
    let displayName = segment.persona;
    
    // --- REFINED LOGIC FOR 'NEEDS ATTENTION' CLUSTERS (Actionable Grouping) ---
    if (segment.persona === 'Needs Attention') {
        // Cluster 1 (34.1%): Dormant (R>300) and Lowest Monetary (<100). --> True Dormant Risk.
        if (key === "1") {
            displayName = "Dormant/Churn Risk";
            colorIndex = 5; // Dark Blue - High Risk/Low Priority for spend
        } 
        // Cluster 0 (11.6%) and Cluster 4 (16.4%) (High Monetary, Moderate/Old Recency)
        // C4 is handled by the backend as 'Potential Loyalists'. We refine C0 here.
        else if (key === "0" || key === "4") {
            // C4 is already 'Potential Loyalists' (R=321, M=333) due to backend logic, so it gets Yellow (index 1).
            // We refine C0 (R=126, M=307) which is High Value but Lapsed (not repeat).
            displayName = key === "0" ? "High Value/Lapsed" : segment.persona;
            colorIndex = key === "0" ? 3 : 1; // Light Blue - High Re-engagement Opportunity
        }
        // Cluster 6 (20.8%): Low Monetary (M=70), Recent Recency (R=149 days). --> Immediate Low-Cost Upsell Opportunity.
        else if (key === "6") {
            displayName = "Low Value/New Opportunity";
            colorIndex = 4; // Mid Blue - High Volume, High Opportunity
        }
        // Cluster 2, 3 (Multi-purchase but Dormant)
        else {
             displayName = "Multi-Purchase/Dormant";
             colorIndex = 6; // Purple/Indigo - Complex Re-engagement
        }
    }
    // --- END REFINED LOGIC ---

    return {
        id: key,
        name: displayName,
        count: segment.size,
        percentage: segment.percentage,
        color: SEGMENT_COLORS_RAW[colorIndex],
        recency: segment.avg_recency_days,
        frequency: segment.avg_frequency,
        monetary: segment.avg_monetary,
        tenure: segment.avg_tenure_days,
        interpurchase: segment.avg_interpurchase_time_days,
        persona: segment.persona,
        icon: segment.persona === 'Champions' ? UserCheck : segment.persona === 'At-Risk' ? UserX : Users,
        // Used to order the segments consistently
        score_rank: segment.persona === 'Champions' ? 1 : segment.persona === 'Potential Loyalists' ? 2 : segment.persona === 'At-Risk' ? 8 : 4
    };
}).sort((a, b) => a.score_rank - b.score_rank);

// --- Utility Components (Unchanged) ---
const Bar = ({ label, value, maxValue, unit = "", colorClass = "bg-primary" }: { label: string, value: number | string, maxValue: number, unit?: string, colorClass?: string }) => {
  const displayValue = typeof value === 'number' ? value.toLocaleString() : value;
  const widthValue = typeof value === 'number' ? value : 0;
  
  return (
    <div className="w-full flex items-center text-xs my-1 group">
      <span className="w-2/5 pr-2 text-right text-muted-foreground group-hover:text-primary transition-colors">{label}</span>
      <div className="w-3/5 bg-muted rounded-full h-5 relative overflow-hidden">
        <div 
          className={`${colorClass} h-5 rounded-full text-primary-foreground pl-2 text-xs flex items-center transition-all duration-500`} 
          style={{ width: `${(widthValue / maxValue) * 100}%` }}
        >
          <span className="font-bold">{displayValue}{unit}</span>
        </div>
      </div>
    </div>
  );
};

// --- SVG Brazil Map Component (Unchanged) ---
const BrazilMap = () => (
    <svg viewBox="0 0 500 500" className="w-full h-auto drop-shadow-lg">
        <path d="M250 20 L150 50 L100 150 L120 250 L100 350 L150 450 L250 480 L350 450 L400 350 L380 250 L400 150 L350 50 Z" fill="hsl(var(--card))" className="opacity-80 stroke-primary/20 stroke-2" />
        <path d="M300 300 L280 350 L350 380 L370 320 Z" fill="hsl(var(--primary))" className="opacity-90 hover:opacity-100 transition-opacity" />
        <path d="M370 320 L350 380 L390 360 L380 310 Z" fill="hsl(var(--primary))" className="opacity-70 hover:opacity-90 transition-opacity" />
        <path d="M280 350 L250 390 L300 410 L350 380 Z" fill="hsl(var(--primary))" className="opacity-60 hover:opacity-80 transition-opacity" />
        <path d="M250 480 L150 450 L200 420 L280 460 Z" fill="hsl(var(--primary))" className="opacity-40 hover:opacity-60 transition-opacity" />
        <path d="M400 150 L380 250 L420 230 L430 160 Z" fill="hsl(var(--primary))" className="opacity-30 hover:opacity-50 transition-opacity" />
        <path d="M200 200 L120 250 L250 300 L280 220 Z" fill="hsl(var(--primary))" className="opacity-30 hover:opacity-50 transition-opacity" />
        <path d="M150 50 L100 150 L250 200 L250 50 Z" fill="hsl(var(--primary))" className="opacity-20 hover:opacity-40 transition-opacity" />
        
        <text x="345" y="345" fontSize="12" fill="hsl(var(--primary-foreground))" fontWeight="bold">SP</text>
        <text x="375" y="335" fontSize="10" fill="hsl(var(--primary-foreground))" fontWeight="bold">RJ</text>
    </svg>
);

const TopStateItem = ({ state, orders, percentage }: { state: string, orders: string, percentage: string }) => (
  <TooltipProvider delayDuration={100}>
    <Tooltip>
      <TooltipTrigger asChild>
        <li className="flex justify-between items-center py-2 px-3 rounded-md hover:bg-primary/10 cursor-pointer transition-colors">
          <span className="font-medium text-sm">{state}</span>
          <span className="text-xs text-muted-foreground">{orders} orders</span>
        </li>
      </TooltipTrigger>
      <TooltipContent>
        <p>{percentage} of total orders</p>
      </TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

const MetricCard = ({ 
  title, 
  icon: Icon, 
  value, 
  subtitle, 
  hoverContent 
}: { 
  title: string, 
  icon: any, 
  value: string | React.ReactNode, 
  subtitle: string, 
  hoverContent: React.ReactNode 
}) => (
  <HoverCard openDelay={200} closeDelay={50}>
    <HoverCardTrigger asChild>
      <Card className="cursor-pointer transition-all duration-300 hover:scale-[1.02] hover:shadow-2xl hover:shadow-primary/10 hover:border-primary/50 group overflow-hidden relative">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
          <CardTitle className="text-sm font-medium">{title}</CardTitle>
          <div className="p-2 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
            <Icon className="h-5 w-5 text-primary" />
          </div>
        </CardHeader>
        <CardContent className="relative">
          <div className="text-3xl font-bold text-primary">{value}</div>
          <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
        </CardContent>
      </Card>
    </HoverCardTrigger>
    {hoverContent}
  </HoverCard>
);

const InsightCard = ({ children }: { children: React.ReactNode }) => (
  <Card className="bg-gradient-to-br from-muted/50 to-muted/30 border-dashed border-2 border-primary/20 shadow-lg">
    <CardHeader>
      <CardTitle className="flex items-center gap-3 text-lg">
        <div className="p-2 rounded-lg bg-primary/10">
          <Lightbulb className="w-5 h-5 text-primary"/>
        </div>
        Strategic Insights & Recommendations
      </CardTitle>
    </CardHeader>
    <CardContent className="text-sm text-muted-foreground space-y-3">
      {children}
    </CardContent>
  </Card>
);


const BusinessInsights = () => {
    // --- Chart Helpers ---
    const maxRecency = Math.max(...segmentData.map(d => d.recency));
    const maxFrequency = Math.max(...segmentData.map(d => d.frequency));
    const maxMonetary = Math.max(...segmentData.map(d => d.monetary));
    
    // Data for bar chart comparison (Recency, Frequency, Monetary)
    const comparisonData = segmentData.map(s => ({
        name: s.name,
        Recency: s.recency,
        Frequency: s.frequency,
        Monetary: s.monetary,
        color: s.color, // Carry color information for bar fill
    }));


  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="container mx-auto px-4 pt-24 pb-16">
        <div className="max-w-7xl mx-auto space-y-12 w-full">
          {/* Hero Section */}
          <div className="text-center space-y-6 relative">
            {/* Decorative elements */}
            <div className="absolute inset-0 -z-10 overflow-hidden">
              <div className="absolute top-0 left-1/4 w-72 h-72 bg-primary/5 rounded-full blur-3xl animate-pulse"></div>
              <div className="absolute top-0 right-1/4 w-72 h-72 bg-accent/5 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}}></div>
            </div>
            
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 mb-4 shadow-lg">
              <TrendingUp className="w-10 h-10 text-primary" />
            </div>
            
            <div className="space-y-3">

              <h1 className="text-5xl md:text-6xl font-bold tracking-tight bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                Olist Business Dashboard
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
                A comprehensive interactive overview of the Olist marketplace, powered by deep Exploratory Data Analysis
              </p>
            </div>
          </div>

          <Tabs defaultValue="overview" className="space-y-8">
            <TabsList className="grid w-full grid-cols-2 md:grid-cols-5 h-auto p-1 bg-muted/50 backdrop-blur">
              <TabsTrigger value="overview" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                Overview
              </TabsTrigger>
              <TabsTrigger value="commerce" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                Commerce
              </TabsTrigger>
              <TabsTrigger value="logistics" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                Logistics
              </TabsTrigger>
              <TabsTrigger value="geography" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                Geography
              </TabsTrigger>
              <TabsTrigger value="segmentation" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                Segmentation
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="overview" className="space-y-8">
              <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  title="Total Revenue"
                  icon={DollarSign}
                  value="$13.5M"
                  subtitle="Based on 99,441 lifetime orders"
                  hoverContent={
                    <HoverCardContent className="w-80">
                      <h4 className="font-semibold mb-2">Revenue by Year (in Millions)</h4>
                      <div className="space-y-2">
                        <Bar label="2017" value={6.2} maxValue={6.5} unit="M" colorClass="bg-green-500"/>
                        <Bar label="2018" value={5.8} maxValue={6.5} unit="M" colorClass="bg-primary"/>
                        <Bar label="2016" value={1.5} maxValue={6.5} unit="M" colorClass="bg-orange-500"/>
                      </div>
                      <p className="text-xs text-muted-foreground mt-3">Peak revenue in 2017 shows strong early growth.</p>
                    </HoverCardContent>
                  }
                />
                
                <MetricCard
                  title="Customer Satisfaction"
                  icon={Star}
                  value="4.1 / 5.0"
                  subtitle="Average of 99,224 reviews"
                  hoverContent={
                    <HoverCardContent className="w-80">
                      <h4 className="font-semibold mb-2">Review Score Distribution (%)</h4>
                      <div className="space-y-2">
                        <Bar label="5 Stars" value={57.8} maxValue={60} colorClass="bg-green-500"/>
                        <Bar label="4 Stars" value={19.3} maxValue={60} colorClass="bg-primary"/>
                        <Bar label="1 Star" value={11.5} maxValue={60} colorClass="bg-red-500"/>
                        <Bar label="3 Stars" value={8.2} maxValue={60} colorClass="bg-yellow-500"/>
                        <Bar label="2 Stars" value={3.2} maxValue={60} colorClass="bg-orange-500"/>
                      </div>
                      <p className="text-xs text-muted-foreground mt-3">Over 77% of reviews are positive (4 or 5 stars).</p>
                    </HoverCardContent>
                  }
                />
                
                <MetricCard
                  title="Customer Retention"
                  icon={Repeat}
                  value="3.1%"
                  subtitle="Customers with >1 purchase"
                  hoverContent={
                    <HoverCardContent className="w-80">
                      <h4 className="font-semibold">Loyalty Opportunity</h4>
                      <p className="text-sm text-muted-foreground">The very low repeat-purchase rate highlights a significant opportunity to grow lifetime value through loyalty programs, email marketing, and personalized re-engagement campaigns.</p>
                    </HoverCardContent>
                  }
                />
                
                <MetricCard
                  title="Total Sellers"
                  icon={Store}
                  value="3,095"
                  subtitle="Active sellers on the platform"
                  hoverContent={
                    <HoverCardContent className="w-80">
                      <h4 className="font-semibold">Seller Distribution</h4>
                      <p className="text-sm text-muted-foreground">A diverse base of sellers is concentrated in the Southeast, aligning with customer density and economic activity.</p>
                    </HoverCardContent>
                  }
                />
              </div>
              
              <InsightCard>
                <div className="space-y-1">
                  <p className="font-semibold text-foreground">🔍 Observation:</p>
                  <p>The marketplace demonstrates strong revenue and a large, diverse customer base but struggles significantly with customer retention. While overall satisfaction is high, the substantial volume of 1-star reviews (over 11%) represents a key risk area that directly impacts loyalty.</p>
                </div>
                <div className="space-y-1">
                  <p className="font-semibold text-foreground">💡 Suggestion:</p>
                  <p>Prioritize a customer retention strategy. Implement a loyalty program and launch targeted re-engagement email campaigns for one-time buyers. Critically, perform a root-cause analysis on 1-star reviews by correlating them with delivery delays and product categories to address the primary sources of dissatisfaction.</p>
                </div>
              </InsightCard>
            </TabsContent>

            <TabsContent value="commerce" className="space-y-8">
              <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  title="Average Order Value"
                  icon={BarChartHorizontal}
                  value="$136.60"
                  subtitle="Average spend per order"
                  hoverContent={
                    <HoverCardContent>
                      <h4 className="font-semibold">Order Value Insights</h4>
                      <p className="text-xs text-muted-foreground mt-2">Higher-priced categories like 'Computers' and 'Small Appliances' significantly lift this average, indicating a market for premium goods.</p>
                    </HoverCardContent>
                  }
                />
                
                <MetricCard
                  title="Bestselling Category"
                  icon={ShoppingBag}
                  value="Bed, Bath & Table"
                  subtitle="By number of orders"
                  hoverContent={
                    <HoverCardContent>
                      <h4 className="font-semibold">Top 5 Categories by Orders</h4>
                      <ol className="list-decimal list-inside text-xs mt-2 text-muted-foreground">
                        <li>Bed, Bath & Table</li>
                        <li>Health & Beauty</li>
                        <li>Sports & Leisure</li>
                        <li>Furniture & Decor</li>
                        <li>Computers & Accessories</li>
                      </ol>
                    </HoverCardContent>
                  }
                />
                
                <MetricCard
                  title="Primary Payment"
                  icon={CreditCard}
                  value="Credit Card"
                  subtitle="Used in 75.6% of payments"
                  hoverContent={
                    <HoverCardContent className="w-80">
                      <h4 className="font-semibold mb-2">Payment Type Distribution (%)</h4>
                      <div className="space-y-2">
                        <Bar label="Credit Card" value={75.6} maxValue={80} colorClass="bg-primary"/>
                        <Bar label="Boleto" value={19.4} maxValue={80} colorClass="bg-green-500"/>
                        <Bar label="Voucher" value={3.8} maxValue={80} colorClass="bg-yellow-500"/>
                        <Bar label="Debit Card" value={1.2} maxValue={80} colorClass="bg-orange-500"/>
                      </div>
                      <p className="text-xs text-muted-foreground mt-3">Avg. Credit Card Installments: <strong>2.9</strong></p>
                    </HoverCardContent>
                  }
                />
                
                <MetricCard
                  title="Top Revenue Month"
                  icon={CalendarDays}
                  value="November"
                  subtitle="Likely driven by Black Friday sales"
                  hoverContent={
                    <HoverCardContent>
                      <h4 className="font-semibold">Seasonality</h4>
                      <p className="text-xs text-muted-foreground mt-2">Sales show significant peaks around major commercial dates, indicating a responsive customer base for promotional events.</p>
                    </HoverCardContent>
                  }
                />
              </div>
              
              <InsightCard>
                <div className="space-y-1">
                  <p className="font-semibold text-foreground">🔍 Observation:</p>
                  <p>The "Bed, Bath & Table" category dominates in volume, while credit cards are the overwhelmingly preferred payment method, with customers frequently using installments.</p>
                </div>
                <div className="space-y-1">
                  <p className="font-semibold text-foreground">💡 Suggestion:</p>
                  <p>Create targeted cross-selling campaigns to customers of "Bed, Bath & Table" for related items in "Furniture & Decor." Highlight installment payment options ("Pay in 3x of $XX") on product pages to encourage conversion on higher-ticket items. Double down on marketing efforts during the October-November period to maximize Black Friday revenue.</p>
                </div>
              </InsightCard>
            </TabsContent>
            
            <TabsContent value="logistics" className="space-y-8">
              <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  title="On-Time Delivery Rate"
                  icon={CheckCircle}
                  value="92.3%"
                  subtitle="Delivered on or before estimate"
                  hoverContent={
                    <HoverCardContent className="w-96">
                      <h4 className="font-semibold">Delivery Performance</h4>
                      <p className="text-xs text-muted-foreground mt-2">While the on-time rate is high, this is largely due to conservative estimates. The actual time to delivery is a key factor in customer satisfaction.</p>
                    </HoverCardContent>
                  }
                />
                
                <MetricCard
                  title="Average Delivery Time"
                  icon={Clock}
                  value="~12.5 days"
                  subtitle="From purchase to customer delivery"
                  hoverContent={
                    <HoverCardContent className="w-96">
                      <h4 className="font-semibold">Average Order Timeline Breakdown</h4>
                      <ul className="list-disc list-inside text-xs mt-2 text-muted-foreground">
                        <li><strong>Payment Approval:</strong> ~1.1 days</li>
                        <li><strong>Seller Handling:</strong> ~2.8 days (seller prepares & ships)</li>
                        <li><strong>Carrier Shipping:</strong> ~8.6 days (in transit)</li>
                      </ul>
                    </HoverCardContent>
                  }
                />
                
                <MetricCard
                  title="Longest Phase"
                  icon={Truck}
                  value="Carrier Shipping"
                  subtitle="~8.6 days on average"
                  hoverContent={
                    <HoverCardContent>
                      <p className="text-sm text-muted-foreground">The "last mile" transit is the biggest portion of the delivery timeline and varies significantly by region.</p>
                    </HoverCardContent>
                  }
                />
                
                <MetricCard
                  title="Ahead of Schedule"
                  icon={CalendarDays}
                  value="11.2 days"
                  subtitle="Avg. days delivered before estimate"
                  hoverContent={
                    <HoverCardContent>
                      <p className="text-sm text-muted-foreground">Delivery estimates are generally conservative, which helps achieve high on-time rates but may not meet modern e-commerce speed expectations.</p>
                    </HoverCardContent>
                  }
                />
              </div>
              
              <InsightCard>
                <div className="space-y-1">
                  <p className="font-semibold text-foreground">🔍 Observation:</p>
                  <p>The total delivery timeline averages nearly two weeks, with the majority of that time spent in the carrier network. While most deliveries arrive before the estimated date, the absolute time-to-door is long by modern standards.</p>
                </div>
                <div className="space-y-1">
                  <p className="font-semibold text-foreground">💡 Suggestion:</p>
                  <p>Focus on optimizing the carrier network. Explore partnerships with faster regional logistics providers, especially for routes outside the dense Southeast region. For sellers with high ratings and fast handling times, consider offering an "expedited shipping" option at checkout to meet customer demand for speed.</p>
                </div>
              </InsightCard>
            </TabsContent>

            <TabsContent value="geography" className="space-y-8">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-1">
                  <Card className="h-full">
                    <CardHeader>
                      <CardTitle>Top States by Order Volume</CardTitle>
                      <CardDescription>The economic powerhouses of Brazil</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ul className="space-y-1">
                        <TopStateItem state="1. São Paulo (SP)" orders="41,746" percentage="42.4%" />
                        <TopStateItem state="2. Rio de Janeiro (RJ)" orders="12,852" percentage="13.0%" />
                        <TopStateItem state="3. Minas Gerais (MG)" orders="11,635" percentage="11.8%" />
                        <TopStateItem state="4. Rio Grande do Sul (RS)" orders="5,466" percentage="5.5%" />
                        <TopStateItem state="5. Paraná (PR)" orders="5,045" percentage="5.1%" />
                      </ul>
                    </CardContent>
                  </Card>
                </div>
                <div className="lg:col-span-2">
                  <Card className="h-full">
                    <CardHeader>
                      <CardTitle>Geographic Sales Distribution</CardTitle>
                      <CardDescription>Sales are heavily concentrated in the Southeast.</CardDescription>
                    </CardHeader>
                    <CardContent className="flex flex-col items-center justify-center p-4">
                        <BrazilMap />
                    </CardContent>
                  </Card>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card className="transition-all duration-300 hover:scale-[1.02] hover:shadow-xl hover:border-primary/50">
                  <CardHeader>
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <MapPin className="w-4 h-4 text-primary" />
                      Top Customer City
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold text-primary">São Paulo</p>
                  </CardContent>
                </Card>
                <Card className="transition-all duration-300 hover:scale-[1.02] hover:shadow-xl hover:border-primary/50">
                  <CardHeader>
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <Store className="w-4 h-4 text-primary" />
                      Top Seller City
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold text-primary">São Paulo</p>
                  </CardContent>
                </Card>
              </div>
              
              <InsightCard>
                <div className="space-y-1">
                  <p className="font-semibold text-foreground">🔍 Observation:</p>
                  <p>The market is overwhelmingly concentrated in the Southeast, particularly in São Paulo. This creates a highly efficient "local" market with lower freight costs and faster delivery times for orders within this region.</p>
                </div>
                <div className="space-y-1">
                  <p className="font-semibold text-foreground">💡 Suggestion:</p>
                  <p>Launch geo-targeted marketing campaigns in the top 5 states to maximize wallet share in established markets. To unlock new growth, explore establishing logistics hubs or partnering with sellers in the Northeast (e.g., Bahia - BA) and Central-West to reduce shipping costs and delivery times to those regions, making the platform more attractive to a wider national audience.</p>
                </div>
              </InsightCard>
            </TabsContent>

            <TabsContent value="segmentation" className="space-y-8">
              {/* Segment Distribution Chart */}
              <Card className="bg-gradient-to-br from-card to-card/50 backdrop-blur hover:shadow-2xl hover:border-primary/50 transition-all duration-300 hover:scale-[1.01]">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <TrendingUp className="w-6 h-6 text-primary" />
                    <CardTitle>Customer Segment Distribution (Optimal K=8)</CardTitle>
                  </div>
                  <CardDescription>GMM results showing the market share and key metrics for all 8 segments.</CardDescription>
                </CardHeader>
                <CardContent className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-1 h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                              <Pie
                                data={segmentData}
                                cx="50%"
                                cy="50%"
                                labelLine={false}
                                outerRadius={100}
                                fill="hsl(var(--primary))"
                                dataKey="count"
                                nameKey="name"
                              >
                                {segmentData.map((entry) => (
                                  <Cell key={`cell-${entry.id}`} fill={entry.color} />
                                ))}
                              </Pie>
                              <RechartsTooltip 
                                formatter={(value, name, props) => [`${(value as number).toLocaleString()} customers`, props.payload.name]}
                                contentStyle={{ 
                                  backgroundColor: 'hsl(var(--card))', 
                                  border: '1px solid hsl(var(--border))',
                                  borderRadius: '8px'
                                }}
                                labelFormatter={(label, props) => `Segment: ${props[0].payload.id}`}
                              />
                              <Legend layout="vertical" align="right" verticalAlign="middle" wrapperStyle={{ fontSize: '12px' }} />
                            </PieChart>
                          </ResponsiveContainer>
                    </div>
                    <div className="lg:col-span-2 space-y-3">
                        <div className="bg-muted/30 p-4 rounded-lg border border-border/50">
                            <h4 className="font-semibold text-sm mb-2 text-primary">ML Rationale (BIC)</h4>
                            <p className="text-xs text-muted-foreground">The model used the **Bayesian Information Criterion (BIC)** to statistically justify $K=8$ as the optimal number of segments, balancing model fit against complexity. This ensures the segmentation is robust and non-overfit.</p>
                        </div>
                         {/* Displaying Key Segments separately */}
                        {segmentData.filter(s => ['Champions', 'Potential Loyalists', 'At-Risk'].includes(s.persona)).map((segment) => (
                            <div key={segment.id} className={`p-3 rounded-lg border flex items-center justify-between transition-shadow duration-300`} 
                                style={{ 
                                    backgroundColor: segment.persona === 'Champions' ? 'hsl(142 71% 45% / 0.1)' : segment.persona === 'At-Risk' ? 'hsl(0 84% 60% / 0.1)' : 'hsl(48 96% 50% / 0.1)',
                                    borderColor: segment.color,
                                }}
                            >
                                <div className="flex items-center gap-3">
                                    <segment.icon className="w-5 h-5" style={{ color: segment.color }} />
                                    <div>
                                        <p className="font-bold text-sm" style={{ color: segment.color }}>{segment.persona} (Cluster {segment.id})</p>
                                        <p className="text-xs text-muted-foreground">{segment.percentage} of customers</p>
                                    </div>
                                </div>
                                <div className="text-right text-xs">
                                    <p className="font-medium">R: {segment.recency.toFixed(0)} days</p>
                                    <p className="font-medium">M: ${segment.monetary.toFixed(0)}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </CardContent>
              </Card>

              {/* Segment Profiles - Render All Segments */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {segmentData.map((segment) => (
                  <Card key={segment.id} className="flex flex-col hover:shadow-xl transition-all duration-300 hover:scale-[1.02]">
                    <CardHeader className="pb-3">
                      <div className="flex items-center gap-3 mb-1">
                        <div 
                          className="p-2 rounded-lg" 
                          style={{ backgroundColor: `${segment.color}15` }} // Light background from segment color
                        >
                          <segment.icon className="w-5 h-5" style={{ color: segment.color }} />
                        </div>
                        <CardTitle className="text-lg" style={{ color: segment.color }}>{segment.name}</CardTitle>
                      </div>
                      <CardDescription className="text-xs">Cluster {segment.id} | {segment.percentage} of base</CardDescription>
                    </CardHeader>
                    <CardContent className="flex-grow space-y-2 text-sm">
                        <div className="flex justify-between items-center text-muted-foreground">
                            <span className="flex items-center gap-1"><Clock className="w-3 h-3"/> Recency</span>
                            <span className="font-bold text-foreground">{segment.recency.toFixed(0)} days</span>
                        </div>
                        <div className="flex justify-between items-center text-muted-foreground">
                            <span className="flex items-center gap-1"><Repeat className="w-3 h-3"/> Frequency</span>
                            <span className="font-bold text-foreground">{segment.frequency.toFixed(1)} purchases</span>
                        </div>
                        <div className="flex justify-between items-center text-muted-foreground">
                            <span className="flex items-center gap-1"><DollarSign className="w-3 h-3"/> Monetary</span>
                            <span className="font-bold text-foreground">${segment.monetary.toFixed(0)}</span>
                        </div>
                        <div className="flex justify-between items-center text-muted-foreground">
                            <span className="flex items-center gap-1"><Hourglass className="w-3 h-3"/> Interpurchase</span>
                            <span className="font-bold text-foreground">{segment.interpurchase.toFixed(1)} days</span>
                        </div>
                        <div className="flex justify-between items-center text-muted-foreground">
                            <span className="flex items-center gap-1"><CalendarDays className="w-3 h-3"/> Tenure</span>
                            <span className="font-bold text-foreground">{segment.tenure.toFixed(0)} days</span>
                        </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
              
              {/* Segment Comparison Chart */}
              <Card className="bg-gradient-to-br from-card to-card/50 backdrop-blur hover:shadow-2xl hover:border-primary/50 transition-all duration-300 hover:scale-[1.01]">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <BarChartHorizontal className="w-5 h-5 text-primary" />
                    <CardTitle>Segment Comparison (RFM Averages)</CardTitle>
                  </div>
                  <CardDescription>Comparing the key RFM features across the 8 statistically derived segments.</CardDescription>
                </CardHeader>
                <CardContent className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparisonData} margin={{ top: 5, right: 20, left: 10, bottom: 90 }} barGap={5}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                      <XAxis 
                        dataKey="name" 
                        stroke="hsl(var(--muted-foreground))" 
                        fontSize={12} 
                        tickLine={false}
                        axisLine={false}
                        angle={-45}
                        textAnchor="end"
                        interval={0}
                      />
                      <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                      <RechartsTooltip 
                        formatter={(value, name) => [name === 'Monetary' ? `$${(value as number).toFixed(2)}` : (value as number).toFixed(1), name]}
                        contentStyle={{ 
                          backgroundColor: 'hsl(var(--card))', 
                          border: '1px solid hsl(var(--border))',
                          borderRadius: '8px'
                        }}
                      />
                      <Legend verticalAlign="top" height={36}/>
                      <RechartsBar dataKey="Recency" fill={SEGMENT_COLORS_RAW[4]} name="Recency (days)" radius={[4, 4, 0, 0]} />
                      <RechartsBar dataKey="Frequency" fill={SEGMENT_COLORS_RAW[1]} name="Frequency (units)" radius={[4, 4, 0, 0]} />
                      <RechartsBar dataKey="Monetary" fill={SEGMENT_COLORS_RAW[0]} name="Monetary ($)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <InsightCard>
                <div className="space-y-1">
                  <p className="font-semibold text-foreground">🔍 Observation (GMM $K=8$):</p>
                  <p>The segmentation confirms that ~98% of the customer base are single-purchase buyers (Clusters 0, 1, 4, 5, 6), which highlights an overwhelming need for retention. The **Champions (Cluster 7)**, though tiny ($0.2\%$), are responsible for the highest Monetary value and Frequency, representing the peak of the loyalty funnel.</p>
                </div>
                <div className="space-y-1">
                  <p className="font-semibold text-foreground">💡 Strategy (Targeted Intervention):</p>
                  <p>Intervention must be prioritized by segment value. The At-Risk segment (Cluster 5) is critical: they were recently active ($\text{}=33$ days), but are slipping away. Launch immediate, exclusive, personalized win-back campaigns to prevent churn of this crucial segment before they become lost.</p>
                </div>
              </InsightCard>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
};

export default BusinessInsights;
