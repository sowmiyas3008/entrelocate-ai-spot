import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { MapPin } from "lucide-react";

const FindLocation = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    city: "",
    shopCategory: "",
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    navigate("/location-results", { state: formData });
  };

  return (
    <div className="min-h-screen bg-background px-4 py-12">
      <div className="container mx-auto max-w-2xl">
        <Card className="p-8 animate-fade-in">
          <div className="flex items-center gap-3 mb-6">
            <MapPin className="h-10 w-10 text-secondary" />
            <div>
              <h1 className="text-3xl font-display font-bold text-foreground">Find Best Location</h1>
              <p className="text-muted-foreground">For your shop category</p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <Label htmlFor="city">City</Label>
              <Input
                id="city"
                placeholder="e.g., New York, London, Tokyo"
                value={formData.city}
                onChange={(e) => setFormData({ ...formData, city: e.target.value })}
                required
                className="mt-1"
              />
            </div>

            <div>
              <Label htmlFor="shopCategory">Shop Category</Label>
              <Input
                id="shopCategory"
                placeholder="e.g., Restaurant, Retail Store, Gym"
                value={formData.shopCategory}
                onChange={(e) => setFormData({ ...formData, shopCategory: e.target.value })}
                required
                className="mt-1"
              />
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
                Find Locations
              </Button>
            </div>
          </form>
        </Card>
      </div>
    </div>
  );
};

export default FindLocation;
