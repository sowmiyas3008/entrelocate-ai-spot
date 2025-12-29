// import { useLocation, useNavigate } from "react-router-dom";
// import { Button } from "@/components/ui/button";
// import { Card } from "@/components/ui/card";
// import { MapPin, TrendingUp, Users } from "lucide-react";

// const LocationResults = () => {
//   const location = useLocation();
//   const navigate = useNavigate();
//   const { city, shopCategory } = location.state || {};

//   const mockLocations = [
//     { name: "Downtown District", score: 95, traffic: "High", demographics: "Young Professionals" },
//     { name: "Shopping Mall Area", score: 88, traffic: "Very High", demographics: "Mixed" },
//     { name: "Suburban Center", score: 82, traffic: "Medium", demographics: "Families" },
//   ];

//   return (
//     <div className="min-h-screen bg-background px-4 py-12">
//       <div className="container mx-auto max-w-4xl">
//         <div className="mb-8 animate-fade-in">
//           <h1 className="text-4xl font-display font-bold mb-2 text-foreground">
//             Best Locations for {shopCategory}
//           </h1>
//           <p className="text-xl text-muted-foreground">in {city}</p>
//         </div>

//         <div className="space-y-4 mb-8">
//           {mockLocations.map((loc, index) => (
//             <Card
//               key={index}
//               className="p-6 hover:shadow-lg transition-all animate-slide-up"
//               style={{ animationDelay: `${index * 0.1}s` }}
//             >
//               <div className="flex items-start justify-between">
//                 <div className="flex-1">
//                   <div className="flex items-center gap-2 mb-3">
//                     <MapPin className="h-5 w-5 text-secondary" />
//                     <h3 className="text-2xl font-display font-bold text-foreground">{loc.name}</h3>
//                     <span className="ml-auto text-3xl font-bold text-secondary">{loc.score}</span>
//                   </div>
//                   <div className="grid grid-cols-2 gap-4 text-sm">
//                     <div className="flex items-center gap-2">
//                       <TrendingUp className="h-4 w-4 text-accent" />
//                       <span className="text-muted-foreground">Traffic: <strong className="text-foreground">{loc.traffic}</strong></span>
//                     </div>
//                     <div className="flex items-center gap-2">
//                       <Users className="h-4 w-4 text-accent" />
//                       <span className="text-muted-foreground">Demographics: <strong className="text-foreground">{loc.demographics}</strong></span>
//                     </div>
//                   </div>
//                 </div>
//               </div>
//             </Card>
//           ))}
//         </div>

//         <div className="h-64 bg-muted rounded-lg flex items-center justify-center border border-border">
//           <p className="text-muted-foreground">Map View (Placeholder)</p>
//         </div>

//         <div className="mt-8 flex gap-4">
//           <Button variant="outline" onClick={() => navigate("/selection")}>
//             Back to Selection
//           </Button>
//           <Button onClick={() => navigate("/find-location")} className="bg-accent text-accent-foreground hover:bg-accent/90">
//             New Search
//           </Button>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default LocationResults;


///////tsx -> jsx

// import { useLocation, useNavigate } from "react-router-dom";
// import { Button } from "@/components/ui/button";
// import { Card } from "@/components/ui/card";
// import { MapPin, TrendingUp, Users } from "lucide-react";

// const LocationResults = () => {
//   const location = useLocation();
//   const navigate = useNavigate();
//   const { city, shopCategory } = location.state || {};

//   const mockLocations = [
//     {
//       name: "Downtown District",
//       score: 95,
//       traffic: "High",
//       demographics: "Young Professionals",
//     },
//     {
//       name: "Shopping Mall Area",
//       score: 88,
//       traffic: "Very High",
//       demographics: "Mixed",
//     },
//     {
//       name: "Suburban Center",
//       score: 82,
//       traffic: "Medium",
//       demographics: "Families",
//     },
//   ];

//   return (
//     <div className="min-h-screen bg-background px-4 py-12">
//       <div className="container mx-auto max-w-4xl">
//         <div className="mb-8 animate-fade-in">
//           <h1 className="text-4xl font-display font-bold mb-2 text-foreground">
//             Best Locations for {shopCategory}
//           </h1>
//           <p className="text-xl text-muted-foreground">in {city}</p>
//         </div>

//         <div className="space-y-4 mb-8">
//           {mockLocations.map((loc, index) => (
//             <Card
//               key={index}
//               className="p-6 hover:shadow-lg transition-all animate-slide-up"
//               style={{ animationDelay: `${index * 0.1}s` }}
//             >
//               <div className="flex items-start justify-between">
//                 <div className="flex-1">
//                   <div className="flex items-center gap-2 mb-3">
//                     <MapPin className="h-5 w-5 text-secondary" />
//                     <h3 className="text-2xl font-display font-bold text-foreground">
//                       {loc.name}
//                     </h3>
//                     <span className="ml-auto text-3xl font-bold text-secondary">
//                       {loc.score}
//                     </span>
//                   </div>

//                   <div className="grid grid-cols-2 gap-4 text-sm">
//                     <div className="flex items-center gap-2">
//                       <TrendingUp className="h-4 w-4 text-accent" />
//                       <span className="text-muted-foreground">
//                         Traffic:{" "}
//                         <strong className="text-foreground">
//                           {loc.traffic}
//                         </strong>
//                       </span>
//                     </div>

//                     <div className="flex items-center gap-2">
//                       <Users className="h-4 w-4 text-accent" />
//                       <span className="text-muted-foreground">
//                         Demographics:{" "}
//                         <strong className="text-foreground">
//                           {loc.demographics}
//                         </strong>
//                       </span>
//                     </div>
//                   </div>
//                 </div>
//               </div>
//             </Card>
//           ))}
//         </div>

//         {/* <div className="h-64 bg-muted rounded-lg flex items-center justify-center border border-border">
//           <p className="text-muted-foreground">Map View (Placeholder)</p>
//         </div> */}

//         <div className="mt-8 flex gap-4">
//           <Button variant="outline" onClick={() => navigate("/selection")}>
//             Back to Selection
//           </Button>
//           <Button
//             onClick={() => navigate("/find-location")}
//             className="bg-accent text-accent-foreground hover:bg-accent/90"
//           >
//             New Search
//           </Button>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default LocationResults;


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




