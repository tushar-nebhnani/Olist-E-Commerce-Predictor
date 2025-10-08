import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import { 
  DollarSign, MapPin, Star, Truck, ShoppingBag, CreditCard, TrendingUp, Users, Store, 
  BarChartHorizontal, Clock, CheckCircle, Package, CalendarDays, Hourglass, AlertCircle, Lightbulb, Repeat, Sparkles
} from "lucide-react";
import Navigation from "@/components/Navigation";

// --- Reusable Bar Component for Hover Cards ---
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

// --- SVG Brazil Map Component ---
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
              <Badge variant="secondary" className="mb-2 px-4 py-1">
                <Sparkles className="w-3 h-3 mr-1 inline" />
                MasterJi by ChaiCode
              </Badge>
              <h1 className="text-5xl md:text-6xl font-bold tracking-tight bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                Olist Business Dashboard
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
                A comprehensive interactive overview of the Olist marketplace, powered by deep Exploratory Data Analysis
              </p>
            </div>
          </div>

          <Tabs defaultValue="overview" className="space-y-8">
            <TabsList className="grid w-full grid-cols-2 md:grid-cols-4 h-auto p-1 bg-muted/50 backdrop-blur">
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
          </Tabs>
        </div>
      </main>
    </div>
  );
};

export default BusinessInsights;
