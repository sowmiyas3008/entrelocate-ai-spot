import { useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { BarChart3, TrendingUp, DollarSign, Calendar } from "lucide-react";

const AnalysisResults = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { shopId, email } = location.state || {};

  const mockData = {
    dailyIncome: "$1,250",
    monthlyProfit: "$28,400",
    growthRate: "+15%",
    customerCount: "156/day",
  };

  return (
    <div className="min-h-screen bg-background px-4 py-12">
      <div className="container mx-auto max-w-4xl">
        <div className="mb-8 animate-fade-in">
          <h1 className="text-4xl font-display font-bold mb-2 text-foreground">
            Shop Performance Dashboard
          </h1>
          <p className="text-xl text-muted-foreground">Shop ID: {shopId}</p>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <Card className="p-6 animate-slide-up">
            <div className="flex items-center gap-3 mb-2">
              <DollarSign className="h-8 w-8 text-secondary" />
              <h3 className="text-lg font-display font-semibold text-muted-foreground">Daily Income</h3>
            </div>
            <p className="text-4xl font-bold text-foreground">{mockData.dailyIncome}</p>
          </Card>

          <Card className="p-6 animate-slide-up" style={{ animationDelay: "0.1s" }}>
            <div className="flex items-center gap-3 mb-2">
              <TrendingUp className="h-8 w-8 text-accent" />
              <h3 className="text-lg font-display font-semibold text-muted-foreground">Monthly Profit</h3>
            </div>
            <p className="text-4xl font-bold text-foreground">{mockData.monthlyProfit}</p>
          </Card>

          <Card className="p-6 animate-slide-up" style={{ animationDelay: "0.2s" }}>
            <div className="flex items-center gap-3 mb-2">
              <BarChart3 className="h-8 w-8 text-primary" />
              <h3 className="text-lg font-display font-semibold text-muted-foreground">Growth Rate</h3>
            </div>
            <p className="text-4xl font-bold text-secondary">{mockData.growthRate}</p>
          </Card>

          <Card className="p-6 animate-slide-up" style={{ animationDelay: "0.3s" }}>
            <div className="flex items-center gap-3 mb-2">
              <Calendar className="h-8 w-8 text-accent" />
              <h3 className="text-lg font-display font-semibold text-muted-foreground">Daily Customers</h3>
            </div>
            <p className="text-4xl font-bold text-foreground">{mockData.customerCount}</p>
          </Card>
        </div>

        <Card className="p-6 mb-8">
          <h3 className="text-xl font-display font-semibold mb-4 text-foreground">Performance Chart</h3>
          <div className="h-64 bg-muted rounded-lg flex items-center justify-center border border-border">
            <p className="text-muted-foreground">Income & Profit Trends (Chart Placeholder)</p>
          </div>
        </Card>

        <Card className="p-6 bg-secondary/10 border-secondary/20">
          <h3 className="text-xl font-display font-semibold mb-3 text-foreground">Insights & Recommendations</h3>
          <ul className="space-y-2 text-muted-foreground">
            <li>• Peak hours: 12 PM - 2 PM and 6 PM - 8 PM</li>
            <li>• Customer retention rate is above average</li>
            <li>• Consider expanding menu/inventory during weekends</li>
            <li>• Location foot traffic is growing steadily</li>
          </ul>
        </Card>

        <div className="mt-8 flex gap-4">
          <Button variant="outline" onClick={() => navigate("/selection")}>
            Back to Selection
          </Button>
          <Button onClick={() => navigate("/shop-analysis")} className="bg-accent text-accent-foreground hover:bg-accent/90">
            New Analysis
          </Button>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults;
