import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  TrendingUp, BarChartHorizontal, Clock, Repeat, DollarSign, Hourglass, CalendarDays, UserCheck, UserX, Users
} from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, BarChart, CartesianGrid, XAxis, YAxis, Tooltip as RechartsTooltip, Bar as RechartsBar } from 'recharts';

// Define the type for a single segment's data for strong typing
type Segment = {
  id: string;
  name: string;
  count: number;
  percentage: string;
  color: string;
  recency: number;
  frequency: number;
  monetary: number;
  tenure: number;
  interpurchase: number;
  persona: string;
  icon: React.ElementType;
};

// Define the type for the comparison chart data
type ComparisonData = {
  name: string;
  Recency: number;
  Frequency: number;
  Monetary: number;
};

// Define the props for the component
interface SegmentationTabProps {
  segmentData: Segment[];
  comparisonData: ComparisonData[];
  segmentColors: string[];
}

const InsightCard = ({ children }: { children: React.ReactNode }) => (
  <Card className="bg-gradient-to-br from-muted/50 to-muted/30 border-dashed border-2 border-primary/20 shadow-lg">
    <CardHeader>
      <CardTitle className="flex items-center gap-3 text-lg">
        <div className="p-2 rounded-lg bg-primary/10">
          {/* <Lightbulb className="w-5 h-5 text-primary"/>  */}
        </div>
        Strategic Insights & Recommendations
      </CardTitle>
    </CardHeader>
    <CardContent className="text-sm text-muted-foreground space-y-3">
      {children}
    </CardContent>
  </Card>
);

export const SegmentationTab = ({ segmentData, comparisonData, segmentColors }: SegmentationTabProps) => {
  return (
    <div className="space-y-8">
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
                  style={{ backgroundColor: `${segment.color}15` }}
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
              <RechartsBar dataKey="Recency" fill={segmentColors[4]} name="Recency (days)" radius={[4, 4, 0, 0]} />
              <RechartsBar dataKey="Frequency" fill={segmentColors[1]} name="Frequency (units)" radius={[4, 4, 0, 0]} />
              <RechartsBar dataKey="Monetary" fill={segmentColors[0]} name="Monetary ($)" radius={[4, 4, 0, 0]} />
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
    </div>
  );
};