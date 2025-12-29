// import { useState } from "react";
// import { useNavigate } from "react-router-dom";
// import { Button } from "@/components/ui/button";
// import { Input } from "@/components/ui/input";
// import { Card } from "@/components/ui/card";
// import { Label } from "@/components/ui/label";
// import { BarChart3 } from "lucide-react";

// const ShopAnalysis = () => {
//   const navigate = useNavigate();
//   const [formData, setFormData] = useState({
//     shopId: "",
//     email: "",
//   });

//   const handleSubmit = (e: React.FormEvent) => {
//     e.preventDefault();
//     navigate("/analysis-results", { state: formData });
//   };

//   return (
//     <div className="min-h-screen bg-background px-4 py-12">
//       <div className="container mx-auto max-w-2xl">
//         <Card className="p-8 animate-fade-in">
//           <div className="flex items-center gap-3 mb-6">
//             <BarChart3 className="h-10 w-10 text-primary" />
//             <div>
//               <h1 className="text-3xl font-display font-bold text-foreground">View Shop Analysis</h1>
//               <p className="text-muted-foreground">Access your business insights</p>
//             </div>
//           </div>

//           <form onSubmit={handleSubmit} className="space-y-6">
//             <div>
//               <Label htmlFor="shopId">Shop ID</Label>
//               <Input
//                 id="shopId"
//                 placeholder="Enter your shop ID"
//                 value={formData.shopId}
//                 onChange={(e) => setFormData({ ...formData, shopId: e.target.value })}
//                 required
//                 className="mt-1"
//               />
//             </div>

//             <div>
//               <Label htmlFor="email">Email</Label>
//               <Input
//                 id="email"
//                 type="email"
//                 placeholder="you@example.com"
//                 value={formData.email}
//                 onChange={(e) => setFormData({ ...formData, email: e.target.value })}
//                 required
//                 className="mt-1"
//               />
//             </div>

//             <div className="flex gap-3">
//               <Button
//                 type="button"
//                 variant="outline"
//                 onClick={() => navigate("/selection")}
//                 className="flex-1"
//               >
//                 Back
//               </Button>
//               <Button
//                 type="submit"
//                 className="flex-1 bg-accent text-accent-foreground hover:bg-accent/90"
//               >
//                 View Analysis
//               </Button>
//             </div>
//           </form>
//         </Card>
//       </div>
//     </div>
//   );
// };

// export default ShopAnalysis;
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { BarChart3 } from "lucide-react";

const ShopAnalysis = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({ shopId: "", email: "" });

  const handleSubmit = (e) => {
    e.preventDefault();
    navigate("/analysis-results", { state: formData });
  };

  return (
    <div className="min-h-screen bg-background px-4 py-12">
      <div className="container mx-auto max-w-2xl">
        <Card className="p-8 animate-fade-in">
          <div className="flex items-center gap-3 mb-6">
            <BarChart3 className="h-10 w-10 text-primary" />
            <div>
              <h1 className="text-3xl font-display font-bold text-foreground">View Shop Analysis</h1>
              <p className="text-muted-foreground">Access your business insights</p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <Label htmlFor="shopId">Shop ID</Label>
              <Input
                id="shopId"
                placeholder="Enter your shop ID"
                value={formData.shopId}
                onChange={(e) => setFormData({ ...formData, shopId: e.target.value })}
                required
                className="mt-1"
              />
            </div>

            <div>
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="you@example.com"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                required
                className="mt-1"
              />
            </div>

            <div className="flex gap-3">
              <Button type="button" variant="outline" onClick={() => navigate("/selection")} className="flex-1">
                Back
              </Button>
              <Button type="submit" className="flex-1 bg-accent text-accent-foreground hover:bg-accent/90">
                View Analysis
              </Button>
            </div>
          </form>
        </Card>
      </div>
    </div>
  );
};

export default ShopAnalysis;
