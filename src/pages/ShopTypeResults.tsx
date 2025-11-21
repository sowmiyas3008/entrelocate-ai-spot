import { useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Building2, TrendingUp, DollarSign } from "lucide-react";

const ShopTypeResults = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { city } = location.state || {};

  const mockShopTypes = [
    { type: "Coffee Shop", potential: "High", avgRevenue: "$250K/year", competition: "Medium" },
    { type: "Fitness Studio", potential: "Very High", avgRevenue: "$180K/year", competition: "Low" },
    { type: "Co-working Space", potential: "High", avgRevenue: "$320K/year", competition: "Medium" },
  ];

  return (
    <div className="min-h-screen bg-background px-4 py-12">
      <div className="container mx-auto max-w-4xl">
        <div className="mb-8 animate-fade-in">
          <h1 className="text-4xl font-display font-bold mb-2 text-foreground">
            Top Business Types in {city}
          </h1>
          <p className="text-xl text-muted-foreground">Based on market analysis and trends</p>
        </div>

        <div className="space-y-4 mb-8">
          {mockShopTypes.map((shop, index) => (
            <Card
              key={index}
              className="p-6 hover:shadow-lg transition-all animate-slide-up"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-3">
                    <Building2 className="h-5 w-5 text-accent" />
                    <h3 className="text-2xl font-display font-bold text-foreground">{shop.type}</h3>
                  </div>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-secondary" />
                      <span className="text-muted-foreground">Potential: <strong className="text-foreground">{shop.potential}</strong></span>
                    </div>
                    <div className="flex items-center gap-2">
                      <DollarSign className="h-4 w-4 text-secondary" />
                      <span className="text-muted-foreground">Avg Revenue: <strong className="text-foreground">{shop.avgRevenue}</strong></span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">Competition: <strong className="text-foreground">{shop.competition}</strong></span>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>

        <Card className="p-6 bg-muted/50 border-border">
          <h3 className="text-xl font-display font-semibold mb-3 text-foreground">Ideal Areas</h3>
          <p className="text-muted-foreground mb-4">
            Based on your city, here are the recommended neighborhoods for these business types:
          </p>
          <ul className="space-y-2 text-muted-foreground">
            <li>• Downtown Business District - High foot traffic</li>
            <li>• Residential Suburbs - Growing population</li>
            <li>• University Area - Young demographic</li>
          </ul>
        </Card>

        <div className="mt-8 flex gap-4">
          <Button variant="outline" onClick={() => navigate("/selection")}>
            Back to Selection
          </Button>
          <Button onClick={() => navigate("/find-shop-type")} className="bg-accent text-accent-foreground hover:bg-accent/90">
            New Search
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ShopTypeResults;
