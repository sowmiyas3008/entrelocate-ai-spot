import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Building2 } from "lucide-react";

const FindShopType = () => {
  const navigate = useNavigate();
  const [city, setCity] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    navigate("/shop-type-results", { state: { city } });
  };

  return (
    <div className="min-h-screen bg-background px-4 py-12">
      <div className="container mx-auto max-w-2xl">
        <Card className="p-8 animate-fade-in">
          <div className="flex items-center gap-3 mb-6">
            <Building2 className="h-10 w-10 text-accent" />
            <div>
              <h1 className="text-3xl font-display font-bold text-foreground">Find Best Shop Type</h1>
              <p className="text-muted-foreground">For your chosen city</p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <Label htmlFor="city">City</Label>
              <Input
                id="city"
                placeholder="e.g., New York, London, Tokyo"
                value={city}
                onChange={(e) => setCity(e.target.value)}
                required
                className="mt-1"
              />
              <p className="text-sm text-muted-foreground mt-2">
                We'll analyze market trends and demographics to suggest the best business types
              </p>
            </div>

            <div className="flex gap-3">
              <Button
                type="button"
                variant="outline"
                onClick={() => navigate("/selection")}
                className="flex-1"
              >
                Back
              </Button>
              <Button
                type="submit"
                className="flex-1 bg-accent text-accent-foreground hover:bg-accent/90"
              >
                Analyze City
              </Button>
            </div>
          </form>
        </Card>
      </div>
    </div>
  );
};

export default FindShopType;
