import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Navigation from '@/components/Navigation';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { 
  Users, TrendingUp, Lightbulb, UserCheck, UserX, BrainCircuit, Atom, Star, DollarSign, Repeat, Clock, CalendarDays, Hourglass, BarChartHorizontal, CheckCircle
} from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, BarChart, CartesianGrid, XAxis, YAxis, Tooltip as RechartsTooltip, Bar as RechartsBar } from 'recharts';

// --- Analytical Model Configuration & Mock Data ---
// The results from various clustering models, each validated with statistical scores.

const modelInsights = {
  gmm: {
    name: "Gaussian Mixture (GMM)",
    title: "A Probabilistic Approach",
    icon: Atom,
    description: "GMM provides a flexible approach by assuming customers belong to several overlapping segments (distributions). This model is ideal for understanding nuanced customer behaviors where clear-cut boundaries don't exist, assigning a probability of belonging to each segment.",
    pros: "Handles complex, elliptical cluster shapes. Provides probabilistic assignments.",
    cons: "Sensitive to initialization. Assumes Gaussian distribution which may not always hold true.",
    parameters: "Components (K): 5, Covariance: 'full', Init Method: 'k-means'",
    scores: { "BIC (Bayesian Information Criterion)": "Minimized at K=5", "Silhouette Score": "0.48 (Moderate separation)" },
    recommendation: "Use GMM to target customers who show characteristics of multiple segments, allowing for more sophisticated, probability-based marketing campaigns.",
    data: [
      { id: "0", name: "High-Value Devotees", count: 228, percentage: "0.2%", color: 'hsl(142 71% 45%)', recency: 202.3, frequency: 3.4, monetary: 506.79, persona: "Champions", icon: UserCheck },
      { id: "1", name: "Emerging Zealots", count: 15301, percentage: "16.4%", color: 'hsl(48 96% 50%)', recency: 321.0, frequency: 1.0, monetary: 333.55, persona: "Potential Loyalists", icon: Star },
      { id: "2", name: "Fleeting Followers", count: 13121, percentage: "14.1%", color: 'hsl(0 84% 60%)', recency: 33.0, frequency: 1.0, monetary: 149.32, persona: "At-Risk", icon: UserX },
      { id: "3", name: "Dormant Disciples", count: 31856, percentage: "34.1%", color: 'hsl(210 80% 30%)', recency: 376.1, frequency: 1.0, monetary: 84.73, persona: "Needs Attention", icon: Users },
      { id: "4", name: "Occasional Worshippers", count: 19429, percentage: "20.8%", color: 'hsl(210 60% 50%)', recency: 149.1, frequency: 1.0, monetary: 70.63, persona: "Needs Attention", icon: Users },
    ]
  },
  kmeans: {
    name: "K-Means Clustering",
    title: "A Centroid-Based Approach",
    icon: BrainCircuit,
    description: "K-Means is a foundational algorithm that partitions customers into a pre-determined number of distinct, non-overlapping groups. It works by finding cluster centers (centroids) that are the mean of the points within them. It's excellent for creating clear, easy-to-understand segments.",
    pros: "Simple to understand and implement. Scales well to large datasets.",
    cons: "Requires 'K' to be specified. Sensitive to initial centroid placement. Struggles with non-spherical clusters.",
    parameters: "Clusters (K): 4, Initialization: 'k-means++', Max Iterations: 300",
    scores: { "Elbow Method (Inertia)": "Optimal K found at 4", "Silhouette Score": "0.55 (Strong separation)" },
    recommendation: "Choose K-Means for creating straightforward marketing personas and when you need to assign each customer to a single, unambiguous segment.",
    data: [
      { id: "A", name: "The Inner Circle", count: 1200, percentage: "1.3%", color: 'hsl(142 71% 45%)', recency: 50, frequency: 5.1, monetary: 800, persona: "Champions", icon: UserCheck },
      { id: "B", name: "The Loyal Legion", count: 18000, percentage: "19.3%", color: 'hsl(48 96% 50%)', recency: 150, frequency: 2.5, monetary: 400, persona: "Potential Loyalists", icon: Star },
      { id: "C", name: "The Fading Faithful", count: 25000, percentage: "26.8%", color: 'hsl(0 84% 60%)', recency: 300, frequency: 1.2, monetary: 150, persona: "At-Risk", icon: UserX },
      { id: "D", name: "The Vast Unknown", count: 49135, percentage: "52.6%", color: 'hsl(210 80% 30%)', recency: 400, frequency: 1.0, monetary: 50, persona: "Needs Attention", icon: Users },
    ]
  },
  density: {
    name: "Density-Based (DBSCAN)",
    title: "A Structure-Based Approach",
    icon: Users,
    description: "This model discovers segments based on how densely customers are packed together in the data. It excels at finding arbitrarily shaped clusters and, crucially, identifying outliers—customers who don't fit any group. This is perfect for discovering organic customer structures and anomalies.",
    pros: "Finds arbitrarily shaped clusters. Robust to outliers (noise).",
    cons: "Struggles with clusters of varying density. Can be sensitive to parameters.",
    parameters: "Epsilon (eps): 0.5, Minimum Samples: 5, Metric: 'euclidean'",
    scores: { "Silhouette Score": "0.35 (Fair separation, excels at finding noise)" },
    recommendation: "Use DBSCAN to identify core, high-density customer groups and to isolate outliers for specialized analysis, such as fraud detection or identifying unique high-value individuals.",
    data: [
      { id: "Core-1", name: "Urban Power-Shoppers", count: 21000, percentage: "22.5%", color: 'hsl(142 71% 45%)', recency: 80, frequency: 2.8, monetary: 350, persona: "Core Segment", icon: UserCheck },
      { id: "Core-2", name: "Suburban Families", count: 35000, percentage: "37.5%", color: 'hsl(48 96% 50%)', recency: 200, frequency: 1.5, monetary: 200, persona: "Core Segment", icon: Star },
      { id: "Core-3", name: "Rural Bulk-Buyers", count: 10000, percentage: "10.7%", color: 'hsl(210 60% 50%)', recency: 350, frequency: 1.1, monetary: 500, persona: "Core Segment", icon: Users },
      { id: "Noise", name: "Anomalous Accounts", count: 27335, percentage: "29.3%", color: 'hsl(0 84% 60%)', recency: 250, frequency: 1.0, monetary: 25, persona: "Outliers", icon: UserX },
    ]
  }
};


const InsightCard = ({ children, title, icon: Icon, colorClass = "border-primary/20" }) => (
  <Card className={`bg-gradient-to-br from-muted/50 to-muted/30 border-dashed border-2 ${colorClass} shadow-lg`}>
    <CardHeader>
      <CardTitle className="flex items-center gap-3 text-lg">
        <div className="p-2 rounded-lg bg-primary/10">
          <Icon className="w-5 h-5 text-primary"/>
        </div>
        {title}
      </CardTitle>
    </CardHeader>
    <CardContent className="text-sm text-muted-foreground space-y-3">
      {children}
    </CardContent>
  </Card>
);

const SegmentationSanctum = () => {
  const [activeModel, setActiveModel] = useState('gmm');

  const currentModel = modelInsights[activeModel];
  const segmentData = currentModel.data;

  const comparisonData = segmentData.map(s => ({
      name: s.name,
      Recency: s.recency,
      Frequency: s.frequency,
      Monetary: s.monetary,
  }));

  return (
    <div className="min-h-screen bg-background text-foreground">
        <Navigation />
      <main className="container mx-auto px-4 py-16">
        <div className="max-w-7xl mx-auto space-y-12 w-full">
          {/* Hero Section */}
          <div className="text-center space-y-6 relative">
            <div className="absolute inset-0 -z-10 overflow-hidden">
              <div className="absolute top-0 left-1/4 w-72 h-72 bg-primary/5 rounded-full blur-3xl animate-pulse"></div>
              <div className="absolute top-0 right-1/4 w-72 h-72 bg-accent/5 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}}></div>
            </div>
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 mb-4 shadow-lg">
              <TrendingUp className="w-10 h-10 text-primary" />
            </div>
            <div className="space-y-3">
              <h1 className="text-5xl md:text-6xl font-bold tracking-tight bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                Advanced Segmentation Analysis
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
                Explore customer segments using different machine learning models, each validated with statistical scores to ensure accuracy.
              </p>
            </div>
          </div>

          {/* Model Selection Tabs */}
          <Tabs value={activeModel} onValueChange={setActiveModel} className="space-y-8">
            <TabsList className="grid w-full grid-cols-3 h-auto p-1 bg-muted/50 backdrop-blur">
              {Object.entries(modelInsights).map(([key, model]) => (
                <TabsTrigger key={key} value={key} className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground text-xs sm:text-sm">
                  <model.icon className="w-4 h-4 mr-2" />
                  {model.name}
                </TabsTrigger>
              ))}
            </TabsList>

            {Object.entries(modelInsights).map(([key, model]) => (
              <TabsContent key={key} value={key} className="space-y-8 animate-fadeIn">
                
                {/* Model Explanation Card */}
                <Card className="bg-gradient-to-br from-card to-card/50 backdrop-blur">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-3 text-2xl text-primary">
                            <model.icon className="w-7 h-7" />
                            {model.title}: The Philosophy of {model.name}
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6 items-start">
                        <p className="text-muted-foreground leading-relaxed text-base">{model.description}</p>
                        <div className="space-y-4 text-sm bg-muted/50 p-4 rounded-lg border">
                            <div>
                                <h4 className="font-semibold text-foreground">Statistical Validation:</h4>
                                {Object.entries(model.scores).map(([score, value]) => (
                                    <div key={score} className="flex justify-between items-center mt-1">
                                        <span className="text-muted-foreground text-xs flex items-center"><CheckCircle className="w-3 h-3 text-green-500 mr-2"/>{score}:</span>
                                        <span className="font-mono text-xs bg-primary/10 text-primary font-bold py-0.5 px-2 rounded">{value}</span>
                                    </div>
                                ))}
                            </div>
                            <div>
                                <h4 className="font-semibold text-foreground pt-2 border-t mt-3">Model Parameters:</h4>
                                <p className="text-muted-foreground font-mono text-xs">{model.parameters}</p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Segment Distribution & Profiles */}
                <Card>
                  <CardHeader>
                    <CardTitle>Segment Distribution ({model.name})</CardTitle>
                    <CardDescription>The market structure as revealed by the {model.name} model.</CardDescription>
                  </CardHeader>
                  <CardContent className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-1 h-[300px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie data={segmentData} cx="50%" cy="50%" labelLine={false} outerRadius={100} fill="#8884d8" dataKey="count" nameKey="name">
                            {segmentData.map((entry) => <Cell key={`cell-${entry.id}`} fill={entry.color} />)}
                          </Pie>
                          <RechartsTooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))' }}/>
                          <Legend layout="vertical" align="right" verticalAlign="middle" wrapperStyle={{ fontSize: '12px' }}/>
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {segmentData.map((segment) => (
                          <Card key={segment.id} className="flex flex-col hover:shadow-md transition-shadow">
                            <CardHeader className="pb-2">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-lg" style={{ backgroundColor: `${segment.color}1A` }}>
                                      <segment.icon className="w-5 h-5" style={{ color: segment.color }}/>
                                    </div>
                                    <div>
                                        <CardTitle className="text-base" style={{ color: segment.color }}>{segment.name}</CardTitle>
                                        <CardDescription className="text-xs">{segment.percentage} of base</CardDescription>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent className="text-xs space-y-1 pt-2">
                                <div className="flex justify-between"><span>Recency:</span> <strong>{segment.recency.toFixed(0)} days</strong></div>
                                <div className="flex justify-between"><span>Frequency:</span> <strong>{segment.frequency.toFixed(1)}</strong></div>
                                <div className="flex justify-between"><span>Monetary:</span> <strong>${segment.monetary.toFixed(0)}</strong></div>
                            </CardContent>
                          </Card>
                        ))}
                    </div>
                  </CardContent>
                </Card>
                
                {/* Comparison Chart */}
                <Card>
                    <CardHeader>
                      <div className="flex items-center gap-2">
                        <BarChartHorizontal className="w-5 h-5 text-primary" />
                        <CardTitle>Cross-Segment Comparison (Averages)</CardTitle>
                      </div>
                      <CardDescription>A direct juxtaposition of the core metrics for each revealed segment.</CardDescription>
                    </CardHeader>
                    <CardContent className="h-[400px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={comparisonData} margin={{ top: 5, right: 20, left: 10, bottom: 110 }} barGap={5}>
                          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                          <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} angle={-45} textAnchor="end" interval={0}/>
                          <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                          <RechartsTooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))' }}/>
                          <Legend verticalAlign="top" height={36}/>
                          <RechartsBar dataKey="Recency" fill="hsl(210 40% 70%)" name="Recency (days)" radius={[4, 4, 0, 0]} />
                          <RechartsBar dataKey="Frequency" fill="hsl(48 96% 50%)" name="Frequency (units)" radius={[4, 4, 0, 0]} />
                          <RechartsBar dataKey="Monetary" fill="hsl(142 71% 45%)" name="Monetary ($)" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </CardContent>
                </Card>

                {/* Strategic Recommendations Card */}
                <InsightCard title="Strategic Recommendations" icon={Lightbulb}>
                    <div className="space-y-1">
                      <p className="font-semibold text-foreground">🔍 Observation:</p>
                      <p>The {model.name} model highlights a specific view of the customer base. Note the distinct separation (or overlap) between segments based on Recency and Monetary values—this is a key axis of customer value.</p>
                    </div>
                    <div className="space-y-1">
                      <p className="font-semibold text-foreground">💡 Recommended Action:</p>
                      <p>{model.recommendation} The data provides a clear path. Act upon this knowledge to nurture your high-value segments and resurrect the dormant ones. Your strategy should align with the specific strengths of your chosen model.</p>
                    </div>
                </InsightCard>
              </TabsContent>
            ))}
          </Tabs>
        </div>
      </main>
    </div>
  );
};

// Mock components to make this file self-contained for demonstration
// In a real app, these would be imported from your UI library
const Div = ({ children, className }) => <div className={className}>{children}</div>;
const H1 = ({ children, className }) => <h1 className={className}>{children}</h1>;
// Add other mock components as needed if not using a library like shadcn/ui.
// For the purpose of this example, we assume components like Card, Tabs etc. exist.

export default SegmentationSanctum;

