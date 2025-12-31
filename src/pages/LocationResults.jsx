


import { useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { MapPin } from "lucide-react";

const LocationResults = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const { city, shopCategory, clusters } = location.state || {};

  if (!clusters || clusters.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-muted-foreground text-lg">No data found</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background px-4 py-12">
      <div className="container mx-auto max-w-4xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-display font-bold text-foreground">
            Best Locations for {shopCategory}
          </h1>
          <p className="text-xl text-muted-foreground">in {city}</p>
        </div>

        {/* Cluster Cards */}
        <div className="space-y-4">
          {clusters.map((cluster, index) => (
            <Card
              key={index}
              className="p-6 hover:shadow-lg transition-all cursor-pointer"
              onClick={() =>
                navigate("/cluster-details", {
                  state: {
                    city,
                    shopCategory,
                    cluster,
                  },
                })
              }
            >
              <div className="flex items-center gap-2 mb-3">
                <MapPin className="h-5 w-5 text-secondary" />
                <h3 className="text-2xl font-display font-bold text-foreground">
                  {cluster.cluster}
                </h3>
                <span className="ml-auto text-2xl font-bold text-secondary">
                  {cluster.population}
                </span>
              </div>

              <div className="text-sm text-muted-foreground">
                Neighborhoods:
                <strong className="text-foreground">
                  {" "}
                  {cluster.neighborhood_areas.join(", ")}
                </strong>
              </div>
            </Card>
          ))}
        </div>

        {/* Buttons */}
        <div className="mt-8 flex gap-4">
          <Button variant="outline" onClick={() => navigate("/selection")}>
            Back to Selection
          </Button>
          <Button
            onClick={() => navigate("/find-location")}
            className="bg-accent text-accent-foreground hover:bg-accent/90"
          >
            New Search
          </Button>
        </div>
      </div>
    </div>
  );
};

export default LocationResults;




