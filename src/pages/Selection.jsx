// import { useNavigate } from "react-router-dom";
// import { Card } from "@/components/ui/card";
// import { MapPin, Building2, BarChart3 } from "lucide-react";

// const Selection = () => {
//   const navigate = useNavigate();

//   const options = [
//     {
//       icon: MapPin,
//       title: "Find Location",
//       subtitle: "Shop Category Given",
//       description: "Already know your business type? Find the best location for it.",
//       path: "/find-location",
//       color: "text-secondary",
//     },
//     {
//       icon: Building2,
//       title: "Find Shop Type",
//       subtitle: "City Given",
//       description: "Have a location in mind? Discover which business thrives there.",
//       path: "/find-shop-type",
//       color: "text-accent",
//     },
//     {
//       icon: BarChart3,
//       title: "View Shop Analysis",
//       subtitle: "Existing User",
//       description: "Track your business performance with detailed analytics.",
//       path: "/shop-analysis",
//       color: "text-primary",
//     },
//   ];

//   return (
//     <div className="min-h-screen bg-background px-4 py-12">
//       <div className="container mx-auto max-w-6xl">
//         <div className="text-center mb-12 animate-fade-in">
//           <h1 className="text-4xl md:text-5xl font-display font-bold mb-4 text-foreground">
//             What would you like to do?
//           </h1>
//           <p className="text-xl text-muted-foreground">
//             Choose the service that fits your needs
//           </p>
//         </div>

//         <div className="grid md:grid-cols-3 gap-6">
//           {options.map((option, index) => {
//             const Icon = option.icon;
//             return (
//               <Card
//                 key={index}
//                 onClick={() => navigate(option.path)}
//                 className="p-8 cursor-pointer hover:shadow-xl transition-all hover:-translate-y-2 border-border group animate-slide-up"
//                 style={{ animationDelay: `${index * 0.1}s` }}
//               >
//                 <Icon className={`h-16 w-16 ${option.color} mb-6 group-hover:scale-110 transition-transform`} />
//                 <h2 className="text-2xl font-display font-bold mb-2 text-foreground">
//                   {option.title}
//                 </h2>
//                 <p className="text-sm text-secondary font-semibold mb-3">
//                   {option.subtitle}
//                 </p>
//                 <p className="text-muted-foreground">
//                   {option.description}
//                 </p>
//               </Card>
//             );
//           })}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Selection;
import { useNavigate } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { MapPin, Building2, BarChart3 } from "lucide-react";

const Selection = () => {
  const navigate = useNavigate();

  const options = [
    { icon: MapPin, title: "Find Location", subtitle: "Shop Category Given", description: "Already know your business type? Find the best location for it.", path: "/find-location", color: "text-secondary" },
    { icon: Building2, title: "Find Shop Type", subtitle: "City Given", description: "Have a location in mind? Discover which business thrives there.", path: "/shop-type-analyzer", color: "text-accent" },
    { icon: BarChart3, title: "View Shop Analysis", subtitle: "Existing User", description: "Track your business performance with detailed analytics.", path: "/shop-analysis", color: "text-primary" },
  ];

  return (
    <div className="min-h-screen bg-background px-4 py-12">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-12 animate-fade-in">
          <h1 className="text-4xl md:text-5xl font-display font-bold mb-4 text-foreground">
            What would you like to do?
          </h1>
          <p className="text-xl text-muted-foreground">Choose the service that fits your needs</p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {options.map((option, index) => {
            const Icon = option.icon;
            return (
              <Card
                key={index}
                onClick={() => navigate(option.path)}
                className="p-8 cursor-pointer hover:shadow-xl transition-all hover:-translate-y-2 border-border group animate-slide-up"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <Icon className={`h-16 w-16 ${option.color} mb-6 group-hover:scale-110 transition-transform`} />
                <h2 className="text-2xl font-display font-bold mb-2 text-foreground">{option.title}</h2>
                <p className="text-sm text-secondary font-semibold mb-3">{option.subtitle}</p>
                <p className="text-muted-foreground">{option.description}</p>
              </Card>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Selection;
