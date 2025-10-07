import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "next-themes";
import Index from "./pages/Index";
import SatisfactionPredictorV1 from "./pages/SatisfactionPredictor_v1";
import CustomerSegmentationPage  from './pages/CustomerSegmentationPage'; 
import BusinessPerformancePage from "./pages/BusinessInsightsPage";
import PurchasePredictorV1 from "./pages/PurchasePredictionV1";
import PurchasePredictorV2 from "./pages/PurchasePredictionV2";
import NotFound from "./pages/NotFound";
import SatisfactionPredictorFinal from "./pages/SatisfactionPredictor_Final";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/satisfaction-predictor-v1" element={<SatisfactionPredictorV1 />} />
            <Route path="/satisfaction-predictor-v2" element={<SatisfactionPredictorFinal />} />
            <Route path="/customer-segmentation" element={<CustomerSegmentationPage />} /> 
            <Route path="/business-insights" element={<BusinessPerformancePage />} /> 
            <Route path="/purchase-prediction-v1" element={<PurchasePredictorV1 />} /> 
            <Route path="/purchase-prediction-v2" element={<PurchasePredictorV2 />} /> 
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
