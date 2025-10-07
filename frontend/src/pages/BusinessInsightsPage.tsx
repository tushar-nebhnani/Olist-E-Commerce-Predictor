import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { 
  DollarSign, MapPin, Star, Truck, ShoppingBag, CreditCard, TrendingUp, Users, Store, 
  BarChartHorizontal, Clock, CheckCircle, Package, CalendarDays, Hourglass, AlertCircle, Lightbulb, Repeat
} from "lucide-react";
import Navigation from "@/components/Navigation";

// --- Reusable Bar Component for Hover Cards ---
const Bar = ({ label, value, maxValue, unit = "", colorClass = "bg-primary" }: { label: string, value: number | string, maxValue: number, unit?: string, colorClass?: string }) => {
  const displayValue = typeof value === 'number' ? value.toLocaleString() : value;
  const widthValue = typeof value === 'number' ? value : 0;
  
  return (
    <div className="w-full flex items-center text-xs my-1 group">
      <span className="w-2/5 pr-2 text-right text-muted-foreground group-hover:text-primary transition-colors">{label}</span>
      <div className="w-3/5 bg-muted rounded-full h-5 relative">
        <div 
          className={`${colorClass} h-5 rounded-full text-primary-foreground pl-2 text-xs flex items-center transition-all duration-300`} 
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
        <path d="M250 20 L150 50 L100 150 L120 250 L100 350 L150 450 L250 480 L350 450 L400 350 L380 250 L400 150 L350 50 Z" fill="#27272A" className="opacity-80 stroke-primary/20 stroke-1" />
        <path d="M300 300 L280 350 L350 380 L370 320 Z" fill="#A855F7" className="opacity-90 hover:opacity-100 transition-opacity" />
        <path d="M370 320 L350 380 L390 360 L380 310 Z" fill="#9333EA" className="opacity-80 hover:opacity-90 transition-opacity" />
        <path d="M280 350 L250 390 L300 410 L350 380 Z" fill="#9333EA" className="opacity-80 hover:opacity-90 transition-opacity" />
        <path d="M250 480 L150 450 L200 420 L280 460 Z" fill="#60A5FA" className="opacity-70 hover:opacity-80 transition-opacity" />
        <path d="M400 150 L380 250 L420 230 L430 160 Z" fill="#3B82F6" className="opacity-60 hover:opacity-70 transition-opacity" />
        <path d="M200 200 L120 250 L250 300 L280 220 Z" fill="#3B82F6" className="opacity-60 hover:opacity-70 transition-opacity" />
        <path d="M150 50 L100 150 L250 200 L250 50 Z" fill="#3B82F6" className="opacity-40 hover:opacity-50 transition-opacity" />
        
        <text x="345" y="345" fontSize="12" fill="white" fontWeight="bold">SP</text>
        <text x="375" y="335" fontSize="10" fill="white" fontWeight="bold">RJ</text>
    </svg>
);

const TopStateItem = ({ state, orders, percentage }: { state: string, orders: string, percentage: string }) => (
  <TooltipProvider delayDuration={100}>
    <Tooltip>
      <TooltipTrigger asChild>
        <li className="flex justify-between items-center py-2 px-3 rounded-md hover:bg-primary/10 cursor-pointer">
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

const BusinessInsights = () => {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navigation />
      
      <main className="container mx-auto px-4 pt-24 pb-16">
        <div className="max-w-7xl mx-auto space-y-12 w-full">
          <div className="text-center space-y-4">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
              <TrendingUp className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-5xl font-bold tracking-tight">
              Olist Business Dashboard
            </h1>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              A god-level interactive overview of the Olist marketplace, powered by deep Exploratory Data Analysis.
            </p>
          </div>

          <Tabs defaultValue="overview" className="space-y-6">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="commerce">Commerce</TabsTrigger>
              <TabsTrigger value="logistics">Logistics & Timelines</TabsTrigger>
              <TabsTrigger value="geography">Geography</TabsTrigger>
            </TabsList>
            
            <TabsContent value="overview" className="space-y-8">
              <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
                <HoverCard openDelay={200} closeDelay={50}>
                  <HoverCardTrigger asChild><Card className="cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-xl hover:border-primary"><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Total Revenue</CardTitle><DollarSign className="h-5 w-5 text-muted-foreground" /></CardHeader><CardContent><div className="text-3xl font-bold text-primary">$13.5M</div><p className="text-xs text-muted-foreground mt-1">Based on 99,441 lifetime orders</p></CardContent></Card></HoverCardTrigger>
                  <HoverCardContent className="w-80"><h4 className="font-semibold mb-2">Revenue by Year (in Millions)</h4><div className="space-y-2"><Bar label="2017" value={6.2} maxValue={6.5} unit="M" colorClass="bg-green-500"/><Bar label="2018" value={5.8} maxValue={6.5} unit="M" colorClass="bg-blue-500"/><Bar label="2016" value={1.5} maxValue={6.5} unit="M" colorClass="bg-orange-500"/></div><p className="text-xs text-muted-foreground mt-3">Peak revenue in 2017 shows strong early growth.</p></HoverCardContent>
                </HoverCard>
                <HoverCard openDelay={200} closeDelay={50}>
                  <HoverCardTrigger asChild><Card className="cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-xl hover:border-primary"><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Customer Satisfaction</CardTitle><Star className="h-5 w-5 text-muted-foreground" /></CardHeader><CardContent><div className="text-3xl font-bold text-primary">4.1 / 5.0</div><p className="text-xs text-muted-foreground mt-1">Average of 99,224 reviews</p></CardContent></Card></HoverCardTrigger>
                   <HoverCardContent className="w-80"><h4 className="font-semibold mb-2">Review Score Distribution (%)</h4><div className="space-y-2"><Bar label="5 Stars" value={57.8} maxValue={60} colorClass="bg-green-500"/><Bar label="4 Stars" value={19.3} maxValue={60} colorClass="bg-blue-400"/><Bar label="1 Star" value={11.5} maxValue={60} colorClass="bg-red-500"/><Bar label="3 Stars" value={8.2} maxValue={60} colorClass="bg-yellow-500"/><Bar label="2 Stars" value={3.2} maxValue={60} colorClass="bg-orange-500"/></div><p className="text-xs text-muted-foreground mt-3">Over 77% of reviews are positive (4 or 5 stars).</p></HoverCardContent>
                </HoverCard>
                <HoverCard openDelay={200} closeDelay={50}>
                  <HoverCardTrigger asChild><Card className="cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-xl hover:border-primary"><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Customer Retention</CardTitle><Repeat className="h-5 w-5 text-muted-foreground" /></CardHeader><CardContent><div className="text-3xl font-bold text-primary">3.1%</div><p className="text-xs text-muted-foreground mt-1">Customers with {'>'}1 purchase</p></CardContent></Card></HoverCardTrigger>
                  <HoverCardContent className="w-80"><h4 className="font-semibold">Loyalty Opportunity</h4><p className="text-sm text-muted-foreground">The very low repeat-purchase rate highlights a significant opportunity to grow lifetime value through loyalty programs, email marketing, and personalized re-engagement campaigns.</p></HoverCardContent>
                </HoverCard>
                 <HoverCard openDelay={200} closeDelay={50}>
                  <HoverCardTrigger asChild><Card className="cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-xl hover:border-primary"><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Total Sellers</CardTitle><Store className="h-5 w-5 text-muted-foreground" /></CardHeader><CardContent><div className="text-3xl font-bold text-primary">3,095</div><p className="text-xs text-muted-foreground mt-1">Active sellers on the platform</p></CardContent></Card></HoverCardTrigger>
                  <HoverCardContent className="w-80"><h4 className="font-semibold">Seller Distribution</h4><p className="text-sm text-muted-foreground">A diverse base of sellers is concentrated in the Southeast, aligning with customer density and economic activity.</p></HoverCardContent>
                </HoverCard>
              </div>
              <Card className="bg-muted/50 border-dashed">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg"><Lightbulb className="w-5 h-5 text-primary"/>High-Level Insights & Strategic Suggestions</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground space-y-2">
                  <p><strong>Observation:</strong> The marketplace demonstrates strong revenue and a large, diverse customer base but struggles significantly with customer retention. While overall satisfaction is high, the substantial volume of 1-star reviews (over 11%) represents a key risk area that directly impacts loyalty.</p>
                  <p><strong>Suggestion:</strong> Prioritize a customer retention strategy. Implement a loyalty program and launch targeted re-engagement email campaigns for one-time buyers. Critically, perform a root-cause analysis on 1-star reviews by correlating them with delivery delays and product categories to address the primary sources of dissatisfaction.</p>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="commerce" className="space-y-8">
              <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
                 <HoverCard openDelay={200} closeDelay={50}>
                  <HoverCardTrigger asChild><Card className="cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-xl hover:border-primary"><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Average Order Value</CardTitle><BarChartHorizontal className="h-5 w-5 text-muted-foreground" /></CardHeader><CardContent><div className="text-3xl font-bold text-primary">$136.60</div><p className="text-xs text-muted-foreground mt-1">Average spend per order</p></CardContent></Card></HoverCardTrigger>
                  <HoverCardContent><h4 className="font-semibold">Order Value Insights</h4><p className="text-xs text-muted-foreground mt-2">Higher-priced categories like 'Computers' and 'Small Appliances' significantly lift this average, indicating a market for premium goods.</p></HoverCardContent>
                 </HoverCard>
                <HoverCard openDelay={200} closeDelay={50}>
                  <HoverCardTrigger asChild><Card className="cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-xl hover:border-primary"><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Bestselling Category</CardTitle><ShoppingBag className="h-5 w-5 text-muted-foreground" /></CardHeader><CardContent><div className="text-3xl font-bold text-primary">Bed, Bath & Table</div><p className="text-xs text-muted-foreground mt-1">By number of orders</p></CardContent></Card></HoverCardTrigger>
                  <HoverCardContent><h4 className="font-semibold">Top 5 Categories by Orders</h4><ol className="list-decimal list-inside text-xs mt-2 text-muted-foreground"><li>Bed, Bath & Table</li><li>Health & Beauty</li><li>Sports & Leisure</li><li>Furniture & Decor</li><li>Computers & Accessories</li></ol></HoverCardContent>
                 </HoverCard>
                <HoverCard openDelay={200} closeDelay={50}>
                  <HoverCardTrigger asChild><Card className="cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-xl hover:border-primary"><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Primary Payment</CardTitle><CreditCard className="h-5 w-5 text-muted-foreground" /></CardHeader><CardContent><div className="text-3xl font-bold text-primary">Credit Card</div><p className="text-xs text-muted-foreground mt-1">Used in 75.6% of payments</p></CardContent></Card></HoverCardTrigger>
                  <HoverCardContent className="w-80"><h4 className="font-semibold mb-2">Payment Type Distribution (%)</h4><div className="space-y-2"><Bar label="Credit Card" value={75.6} maxValue={80} colorClass="bg-blue-500"/><Bar label="Boleto" value={19.4} maxValue={80} colorClass="bg-green-500"/><Bar label="Voucher" value={3.8} maxValue={80} colorClass="bg-yellow-500"/><Bar label="Debit Card" value={1.2} maxValue={80} colorClass="bg-orange-500"/></div><p className="text-xs text-muted-foreground mt-3">Avg. Credit Card Installments: <strong>2.9</strong></p></HoverCardContent>
                </HoverCard>
                <HoverCard openDelay={200} closeDelay={50}>
                  <HoverCardTrigger asChild><Card className="cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-xl hover:border-primary"><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Top Revenue Month</CardTitle><CalendarDays className="h-5 w-5 text-muted-foreground" /></CardHeader><CardContent><div className="text-3xl font-bold text-primary">November</div><p className="text-xs text-muted-foreground mt-1">Likely driven by Black Friday sales</p></CardContent></Card></HoverCardTrigger>
                  <HoverCardContent><h4 className="font-semibold">Seasonality</h4><p className="text-xs text-muted-foreground mt-2">Sales show significant peaks around major commercial dates, indicating a responsive customer base for promotional events.</p></HoverCardContent>
                </HoverCard>
              </div>
              <Card className="bg-muted/50 border-dashed">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg"><Lightbulb className="w-5 h-5 text-primary"/>Commerce Strategy & Opportunities</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground space-y-2">
                  <p><strong>Observation:</strong> The "Bed, Bath & Table" category dominates in volume, while credit cards are the overwhelmingly preferred payment method, with customers frequently using installments.</p>
                  <p><strong>Suggestion:</strong> Create targeted cross-selling campaigns to customers of "Bed, Bath & Table" for related items in "Furniture & Decor." Highlight installment payment options ("Pay in 3x of $XX") on product pages to encourage conversion on higher-ticket items. Double down on marketing efforts during the October-November period to maximize Black Friday revenue.</p>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="logistics" className="space-y-8">
              <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
                <HoverCard openDelay={200} closeDelay={50}>
                  <HoverCardTrigger asChild><Card className="cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-xl hover:border-primary"><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">On-Time Delivery Rate</CardTitle><CheckCircle className="h-5 w-5 text-green-500" /></CardHeader><CardContent><div className="text-3xl font-bold text-primary">92.3%</div><p className="text-xs text-muted-foreground mt-1">Delivered on or before estimate</p></CardContent></Card></HoverCardTrigger>
                  <HoverCardContent className="w-96"><h4 className="font-semibold">Delivery Performance</h4><p className="text-xs text-muted-foreground mt-2">While the on-time rate is high, this is largely due to conservative estimates. The actual time to delivery is a key factor in customer satisfaction.</p></HoverCardContent>
                </HoverCard>
                 <HoverCard openDelay={200} closeDelay={50}>
                  <HoverCardTrigger asChild><Card className="cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-xl hover:border-primary"><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Average Delivery Time</CardTitle><Clock className="h-5 w-5 text-muted-foreground" /></CardHeader><CardContent><div className="text-3xl font-bold text-primary">~12.5 days</div><p className="text-xs text-muted-foreground mt-1">From purchase to customer delivery</p></CardContent></Card></HoverCardTrigger>
                  <HoverCardContent className="w-96"><h4 className="font-semibold">Average Order Timeline Breakdown</h4><ul className="list-disc list-inside text-xs mt-2 text-muted-foreground"><li><strong>Payment Approval:</strong> ~1.1 days</li><li><strong>Seller Handling:</strong> ~2.8 days (seller prepares & ships)</li><li><strong>Carrier Shipping:</strong> ~8.6 days (in transit)</li></ul></HoverCardContent>
                </HoverCard>
                <HoverCard openDelay={200} closeDelay={50}>
                  <HoverCardTrigger asChild><Card className="cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-xl hover:border-primary"><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Longest Phase</CardTitle><Truck className="h-5 w-5 text-muted-foreground" /></CardHeader><CardContent><div className="text-3xl font-bold text-primary">Carrier Shipping</div><p className="text-xs text-muted-foreground mt-1">~8.6 days on average</p></CardContent></Card></HoverCardTrigger>
                  <HoverCardContent><p className="text-sm text-muted-foreground">The "last mile" transit is the biggest portion of the delivery timeline and varies significantly by region.</p></HoverCardContent>
                </HoverCard>
                <HoverCard openDelay={200} closeDelay={50}>
                  <HoverCardTrigger asChild><Card className="cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-xl hover:border-primary"><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Ahead of Schedule</CardTitle><CalendarDays className="h-5 w-5 text-muted-foreground" /></CardHeader><CardContent><div className="text-3xl font-bold text-primary">11.2 days</div><p className="text-xs text-muted-foreground mt-1">Avg. days delivered before estimate</p></CardContent></Card></HoverCardTrigger>
                  <HoverCardContent><p className="text-sm text-muted-foreground">Delivery estimates are generally conservative, which helps achieve high on-time rates but may not meet modern e-commerce speed expectations.</p></HoverCardContent>
                </HoverCard>
              </div>
              <Card className="bg-muted/50 border-dashed">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg"><Lightbulb className="w-5 h-5 text-primary"/>Logistics Analysis & Recommendations</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground space-y-2">
                  <p><strong>Observation:</strong> The total delivery timeline averages nearly two weeks, with the majority of that time spent in the carrier network. While most deliveries arrive before the estimated date, the absolute time-to-door is long by modern standards.</p>
                  <p><strong>Suggestion:</strong> Focus on optimizing the carrier network. Explore partnerships with faster regional logistics providers, especially for routes outside the dense Southeast region. For sellers with high ratings and fast handling times, consider offering an "expedited shipping" option at checkout to meet customer demand for speed.</p>
                </CardContent>
              </Card>
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
                <Card className="transition-all duration-300 hover:scale-105 hover:shadow-xl hover:border-primary">
                    <CardHeader><CardTitle className="text-sm font-medium">Top Customer City</CardTitle></CardHeader>
                    <CardContent><p className="text-2xl font-bold">sao paulo</p></CardContent>
                </Card>
                <Card className="transition-all duration-300 hover:scale-105 hover:shadow-xl hover:border-primary">
                    <CardHeader><CardTitle className="text-sm font-medium">Top Seller City</CardTitle></CardHeader>
                    <CardContent><p className="text-2xl font-bold">sao paulo</p></CardContent>
                </Card>
              </div>
              <Card className="bg-muted/50 border-dashed">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg"><Lightbulb className="w-5 h-5 text-primary"/>Geographic Strategy Insights</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground space-y-2">
                  <p><strong>Observation:</strong> The market is overwhelmingly concentrated in the Southeast, particularly in São Paulo. This creates a highly efficient "local" market with lower freight costs and faster delivery times for orders within this region.</p>
                  <p><strong>Suggestion:</strong> Launch geo-targeted marketing campaigns in the top 5 states to maximize wallet share in established markets. To unlock new growth, explore establishing logistics hubs or partnering with sellers in the Northeast (e.g., Bahia - BA) and Central-West to reduce shipping costs and delivery times to those regions, making the platform more attractive to a wider national audience.</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
};

export default BusinessInsights;

