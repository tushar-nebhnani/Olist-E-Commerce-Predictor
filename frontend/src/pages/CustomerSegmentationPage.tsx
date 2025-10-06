import Navigation from "@/components/Navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Users, UserCheck, UserX } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Bar } from 'recharts';

// --- UPDATED ---
// Data is now directly from your final segment_analysis_advanced.json file.
const segmentData = [
  { 
    name: 'Single-Purchase Core', // Renamed from "Champions" for clarity
    icon: Users,
    description: "The vast majority of your customer base. They have made a single purchase and form the foundation of your business.",
    action: "Encourage a second purchase through targeted follow-ups with related products or introductory offers for new categories.",
    count: 89280, 
    recency: 240.3,
    frequency: 1.0,
    monetary: 160.9
  },
  { 
    name: 'At-Risk Loyalists', // Renamed from "At-Risk" for clarity
    icon: UserX,
    description: "A small, high-value segment of repeat customers who spend more but have not purchased recently.",
    action: "Target immediately with 'We miss you!' campaigns and personalized incentives to prevent churn.",
    count: 2744,
    recency: 222.4,
    frequency: 2.11,
    monetary: 308.45
  },
];

// --- MODIFIED ---
// Simplified color palette for two segments.
const SEGMENT_COLORS = [
  'hsl(16 100% 60%)',   // Orange for the Core group
  'hsl(142 71% 45%)',   // Green for the Loyalists
];

const CustomerSegmentation = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="container mx-auto px-6 pt-24 pb-16">
        <div className="max-w-6xl mx-auto space-y-12">
          {/* Hero Section */}
          <div className="text-center space-y-4">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent mb-4">
              <Users className="w-8 h-8 text-accent-foreground" />
            </div>
            <h1 className="text-5xl font-bold tracking-tight">
              Customer Segmentation
            </h1>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              An unsupervised learning approach using RFM analysis and a Gaussian Mixture Model to identify distinct customer groups.
            </p>
          </div>

          {/* Segment Distribution Chart */}
          <Card className="hover:shadow-lg hover:border-primary/50 transition-all">
            <CardHeader>
              <CardTitle>Segment Distribution</CardTitle>
              <CardDescription>The proportion of total customers in each segment.</CardDescription>
            </CardHeader>
            <CardContent className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={segmentData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={110}
                    fill="hsl(var(--primary))"
                    dataKey="count"
                    nameKey="name"
                  >
                    {segmentData.map((_entry, index) => (
                      <Cell key={`cell-${index}`} fill={SEGMENT_COLORS[index]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    formatter={(value) => `${(value as number).toLocaleString()} customers`}
                    contentStyle={{ 
                      backgroundColor: 'hsl(var(--card))', 
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Segment Profiles */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {segmentData.map((segment, index) => (
              <Card key={segment.name} className="flex flex-col hover:shadow-lg hover:border-primary/50 transition-all">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div 
                      className="p-2 rounded-lg" 
                      style={{ backgroundColor: SEGMENT_COLORS[index].replace(')', ' / 0.1)').replace('hsl(', 'hsla(') }}
                    >
                      <segment.icon className="w-5 h-5" style={{ color: SEGMENT_COLORS[index] }} />
                    </div>
                    <CardTitle className="text-xl">{segment.name}</CardTitle>
                  </div>
                  <CardDescription>{segment.description}</CardDescription>
                </CardHeader>
                <CardContent className="flex-grow space-y-4">
                  <div>
                    <h4 className="font-semibold text-sm mb-3">Key Characteristics</h4>
                    <div className="grid grid-cols-3 gap-2 text-center text-sm">
                      <div className="p-3 rounded-lg bg-accent">
                        <div className="font-bold text-lg">{segment.recency.toFixed(0)}</div>
                        <div className="text-xs text-muted-foreground mt-1">days<br/>Recency</div>
                      </div>
                      <div className="p-3 rounded-lg bg-accent">
                        <div className="font-bold text-lg">{segment.frequency.toFixed(1)}</div>
                        <div className="text-xs text-muted-foreground mt-1">purchases<br/>Frequency</div>
                      </div>
                      <div className="p-3 rounded-lg bg-accent">
                        <div className="font-bold text-lg">${segment.monetary.toFixed(0)}</div>
                        <div className="text-xs text-muted-foreground mt-1">average<br/>Monetary</div>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-semibold text-sm mb-2">Recommended Action</h4>
                    <p className="text-sm text-muted-foreground">{segment.action}</p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          
          {/* Segment Comparison Chart */}
          <Card className="hover:shadow-lg hover:border-primary/50 transition-all">
            <CardHeader>
              <CardTitle>Segment Comparison (Averages)</CardTitle>
              <CardDescription>Comparing the average RFM values across all segments.</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={segmentData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }} barGap={50}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis 
                    dataKey="name" 
                    stroke="hsl(var(--muted-foreground))" 
                    fontSize={12} 
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'hsl(var(--card))', 
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                  <Bar dataKey="recency" fill={SEGMENT_COLORS[0]} name="Recency (days)" />
                  <Bar dataKey="frequency" fill={SEGMENT_COLORS[1]} name="Frequency" />
                  <Bar dataKey="monetary" fill={'hsl(var(--primary))'} name="Monetary ($)" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default CustomerSegmentation;