import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import SignUp from "./pages/SignUp";
import Login from "./pages/Login";
import Selection from "./pages/Selection";
import FindLocation from "./pages/FindLocation";
import FindShopType from "./pages/FindShopType";
import ShopAnalysis from "./pages/ShopAnalysis";
import LocationResults from "./pages/LocationResults";
import ShopTypeResults from "./pages/ShopTypeResults";
import AnalysisResults from "./pages/AnalysisResults";
import NotFound from "./pages/NotFound";
import ClusterDetails from "./pages/ClusterDetails";
import ShopTypeAnalyzer from "./pages/shopTypeAnalyzer";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/login" element={<Login />} />
          <Route path="/selection" element={<Selection />} />
          <Route path="/find-location" element={<FindLocation />} />
          <Route path="/find-shop-type" element={<FindShopType />} />
          <Route path="/shop-analysis" element={<ShopAnalysis />} />
          <Route path="/location-results" element={<LocationResults />} />
          <Route path="/shop-type-results" element={<ShopTypeResults />} />
          <Route path="/analysis-results" element={<AnalysisResults />} />
          <Route path="/cluster-details" element={<ClusterDetails />} />
          <Route path="/shop-type-analyzer" element={<ShopTypeAnalyzer />} />




          
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;


