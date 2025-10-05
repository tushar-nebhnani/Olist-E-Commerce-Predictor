// src/pages/CustomerSegmentsPage.tsx
import Navigation from "@/components/Navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Users, UserPlus, UserX, UserCheck } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Bar } from 'recharts';

// --- Data from your segmentation analysis ---
const segmentData = [
  { 
    name: 'New Customers', 
    icon: UserPlus,
    description: "Recent, single-purchase customers with moderate spending. They are the future of the business.",
    action: "Engage with welcome offers and onboarding campaigns to encourage a second purchase.",
    count: 15313,
    recency: 41,
    frequency: 1,
    monetary: 136
  },
  { 
    name: 'At-Risk Loyal Customers', 
    icon: UserX,
    description: "High-value, frequent buyers who haven't purchased in a long time. They are in danger of churning.",
    action: "Target with personalized 'We miss you!' campaigns and high-value discounts to win them back.",
    count: 2744,
    recency: 222,
    frequency: 2.11,
    monetary: 308
  },
  { 
    name: 'Hibernating High Spenders', 
    icon: UserCheck,
    description: "Made a single, high-value purchase a long time ago but never returned.",
    action: "Re-engage with information on new, premium products similar to their original purchase.",
    count: 32128,
    recency: 274,
    frequency: 1,
    monetary: 294
  },
  { 
    name: 'Lost Low-Value Customers', 
    icon: UserX,
    description: "The largest group, consisting of customers who made one low-value purchase long ago.",
    action: "Include in general brand awareness campaigns but avoid spending significant marketing budget.",
    count: 41839,
    recency: 287,
    frequency: 1,
    monetary: 68
  },
];

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042'];

const CustomerSegmentsPage = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="container mx-auto px-6 pt-24 pb-16">
        <div className="max-w-6xl mx-auto space-y-12">
          {/* --- Hero Section --- */}
          <div className="text-center space-y-4">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
              <Users className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-5xl font-bold tracking-tight">
              Customer Segmentation
            </h1>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              An unsupervised learning approach using RFM analysis and K-Means clustering to identify distinct customer groups.
            </p>
          </div>

          {/* --- Segment Distribution Chart --- */}
          <Card className="border-2">
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
                    fill="#8884d8"
                    dataKey="count"
                    nameKey="name"
                  >
                    {segmentData.map((_entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => `${(value as number).toLocaleString()} customers`} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* --- Segment Profiles --- */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {segmentData.map((segment, index) => (
              <Card key={segment.name} className="border-2 flex flex-col">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <segment.icon className="w-5 h-5" style={{ color: COLORS[index] }} />
                    {segment.name}
                  </CardTitle>
                  <CardDescription>{segment.description}</CardDescription>
                </CardHeader>
                <CardContent className="flex-grow space-y-4">
                  <div>
                    <h4 className="font-semibold text-sm mb-2">Key Characteristics</h4>
                    <div className="grid grid-cols-3 gap-2 text-center text-sm">
                      <div className="p-2 rounded bg-primary/5"><span className="font-bold">{segment.recency.toFixed(0)}</span> days<br/>Recency</div>
                      <div className="p-2 rounded bg-primary/5"><span className="font-bold">{segment.frequency.toFixed(1)}</span><br/>Frequency</div>
                      <div className="p-2 rounded bg-primary/5"><span className="font-bold">${segment.monetary.toFixed(0)}</span><br/>Monetary</div>
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
          
          {/* --- Segment Comparison Chart --- */}
          <Card className="border-2">
              <CardHeader>
                <CardTitle>Segment Comparison (Averages)</CardTitle>
                <CardDescription>Comparing the average RFM values across all segments.</CardDescription>
              </CardHeader>
              <CardContent className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={segmentData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted))" />
                        <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                        <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                        <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--background))', border: '1px solid hsl(var(--border))' }} />
                        <Legend />
                        <Bar dataKey="recency" fill={COLORS[0]} />
                        <Bar dataKey="frequency" fill={COLORS[1]} />
                        <Bar dataKey="monetary" fill={COLORS[2]} />
                    </BarChart>
                </ResponsiveContainer>
              </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default CustomerSegmentsPage;