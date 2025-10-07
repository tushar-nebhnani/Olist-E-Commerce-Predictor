// src/pages/BusinessInsights.tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { DollarSign, MapPin, Star, Truck, ShoppingBag, CreditCard, TrendingUp } from "lucide-react";
import Navigation from "@/components/Navigation"; // Assuming you have a Navigation component

const BusinessInsights = () => {
  return (
    <div className="flex min-h-screen w-full flex-col bg-muted/40">
      { <Navigation /> }
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-8">
        <div className="max-w-6xl mx-auto space-y-8 w-full">
          <div className="text-center space-y-4">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
              <TrendingUp className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-5xl font-bold tracking-tight">
              Business Insights
            </h1>
            <p className="text-xl text-muted-foreground">
              Comprehensive analytics and metrics for the Olist marketplace
            </p>
          </div>

          <Tabs defaultValue="overview" className="space-y-6">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="commerce">Commerce</TabsTrigger>
              <TabsTrigger value="geography">Geography</TabsTrigger>
            </TabsList>

            {/* Overview Tab Content */}
            <TabsContent value="overview" className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                <HoverCard>
                  <HoverCardTrigger asChild>
                    <Card className="border-2 cursor-pointer hover:border-primary transition-colors">
                      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">
                          Total Revenue
                        </CardTitle>
                        <DollarSign className="h-5 w-5 text-primary" />
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-primary">$13.5M</div>
                        <p className="text-xs text-muted-foreground mt-1">
                          Based on 99,441 lifetime orders
                        </p>
                      </CardContent>
                    </Card>
                  </HoverCardTrigger>
                  <HoverCardContent className="w-80">
                    <h4 className="font-semibold">Revenue Details</h4>
                    <p className="text-sm text-muted-foreground">Total sales value across all completed orders. This excludes shipping fees.</p>
                    <div className="mt-2 text-xs">
                        <p><strong>2017:</strong> $6.2M (Peak Year)</p>
                        <p><strong>2018:</strong> $5.8M</p>
                        <p><strong>2016:</strong> $1.5M</p>
                    </div>
                  </HoverCardContent>
                </HoverCard>
                
                <HoverCard>
                  <HoverCardTrigger asChild>
                    <Card className="border-2 cursor-pointer hover:border-primary transition-colors">
                      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">
                          Customer Satisfaction
                        </CardTitle>
                        <Star className="h-5 w-5 text-primary" />
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-primary">4.1 / 5.0</div>
                        <p className="text-xs text-muted-foreground mt-1">
                          Average of all customer reviews
                        </p>
                      </CardContent>
                    </Card>
                  </HoverCardTrigger>
                   <HoverCardContent className="w-80">
                    <h4 className="font-semibold">Review Score Distribution</h4>
                    <p className="text-sm text-muted-foreground">Breakdown of review scores from 1 (worst) to 5 (best).</p>
                    <div className="mt-2 text-xs">
                        <p><strong>5 Stars:</strong> 57.8%</p>
                        <p><strong>4 Stars:</strong> 19.3%</p>
                        <p><strong>3 Stars:</strong> 8.2%</p>
                        <p><strong>2 Stars:</strong> 3.2%</p>
                        <p><strong>1 Star:</strong> 11.5%</p>
                    </div>
                  </HoverCardContent>
                </HoverCard>
                
                <HoverCard>
                  <HoverCardTrigger asChild>
                    <Card className="border-2 cursor-pointer hover:border-primary transition-colors">
                      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">
                          On-Time Delivery
                        </CardTitle>
                        <Truck className="h-5 w-5 text-primary" />
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-primary">92.1%</div>
                        <p className="text-xs text-muted-foreground mt-1">
                          Orders delivered on or before estimate
                        </p>
                      </CardContent>
                    </Card>
                  </HoverCardTrigger>
                  <HoverCardContent className="w-80">
                    <h4 className="font-semibold">Delivery Performance</h4>
                    <p className="text-sm text-muted-foreground">The majority of orders arrive early, with an average early arrival of 11.2 days.</p>
                     <div className="mt-2 text-xs">
                        <p><strong>Average Delivery Time:</strong> 12.5 days</p>
                        <p><strong>Late Deliveries:</strong> 7.9%</p>
                    </div>
                  </HoverCardContent>
                </HoverCard>
              </div>
            </TabsContent>

            {/* Commerce Tab Content */}
            <TabsContent value="commerce" className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                <HoverCard>
                  <HoverCardTrigger asChild>
                    <Card className="border-2 cursor-pointer hover:border-primary transition-colors">
                      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Bestselling Category</CardTitle>
                        <ShoppingBag className="h-5 w-5 text-primary" />
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-primary">Bed, Bath & Table</div>
                        <p className="text-xs text-muted-foreground mt-1">
                          Top category by total orders
                        </p>
                      </CardContent>
                    </Card>
                  </HoverCardTrigger>
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
                 </HoverCard>
                
                <HoverCard>
                  <HoverCardTrigger asChild>
                    <Card className="border-2 cursor-pointer hover:border-primary transition-colors">
                      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Primary Payment Method</CardTitle>
                        <CreditCard className="h-5 w-5 text-primary" />
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-primary">Credit Card</div>
                        <p className="text-xs text-muted-foreground mt-1">
                          Used in 75.6% of all payments
                        </p>
                      </CardContent>
                    </Card>
                  </HoverCardTrigger>
                  <HoverCardContent>
                     <h4 className="font-semibold">Payment Method Breakdown</h4>
                      <div className="mt-2 text-xs">
                        <p><strong>Credit Card:</strong> 75.6%</p>
                        <p><strong>Boleto:</strong> 19.4%</p>
                        <p><strong>Voucher:</strong> 3.8%</p>
                        <p><strong>Debit Card:</strong> 1.2%</p>
                    </div>
                  </HoverCardContent>
                </HoverCard>
              </div>
            </TabsContent>
            
            {/* Geography Tab Content */}
            <TabsContent value="geography" className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                <HoverCard>
                  <HoverCardTrigger asChild>
                    <Card className="border-2 cursor-pointer hover:border-primary transition-colors">
                      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Top Sales Region</CardTitle>
                        <MapPin className="h-5 w-5 text-primary" />
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-primary">São Paulo</div>
                        <p className="text-xs text-muted-foreground mt-1">
                          State with 42.4% of total orders
                        </p>
                      </CardContent>
                    </Card>
                  </HoverCardTrigger>
                  <HoverCardContent>
                     <h4 className="font-semibold">Top 5 States by Orders</h4>
                      <ol className="list-decimal list-inside text-xs mt-2 text-muted-foreground">
                        <li>São Paulo (SP)</li>
                        <li>Rio de Janeiro (RJ)</li>
                        <li>Minas Gerais (MG)</li>
                        <li>Rio Grande do Sul (RS)</li>
                        <li>Paraná (PR)</li>
                      </ol>
                  </HoverCardContent>
                </HoverCard>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
};

export default BusinessInsights;
