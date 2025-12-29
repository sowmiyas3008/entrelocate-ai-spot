// import { useLocation, useNavigate } from "react-router-dom";
// import { Button } from "@/components/ui/button";
// import { Card } from "@/components/ui/card";
// import { BarChart3, TrendingUp, DollarSign, Calendar } from "lucide-react";

// const AnalysisResults = () => {
//   const location = useLocation();
//   const navigate = useNavigate();
//   const { shopId, email } = location.state || {};

//   const mockData = {
//     dailyIncome: "$1,250",
//     monthlyProfit: "$28,400",
//     growthRate: "+15%",
//     customerCount: "156/day",
//   };

//   return (
//     <div className="min-h-screen bg-background px-4 py-12">
//       <div className="container mx-auto max-w-4xl">
//         <div className="mb-8 animate-fade-in">
//           <h1 className="text-4xl font-display font-bold mb-2 text-foreground">
//             Shop Performance Dashboard
//           </h1>
//           <p className="text-xl text-muted-foreground">Shop ID: {shopId}</p>
//         </div>

//         <div className="grid md:grid-cols-2 gap-6 mb-8">
//           <Card className="p-6 animate-slide-up">
//             <div className="flex items-center gap-3 mb-2">
//               <DollarSign className="h-8 w-8 text-secondary" />
//               <h3 className="text-lg font-display font-semibold text-muted-foreground">Daily Income</h3>
//             </div>
//             <p className="text-4xl font-bold text-foreground">{mockData.dailyIncome}</p>
//           </Card>

//           <Card className="p-6 animate-slide-up" style={{ animationDelay: "0.1s" }}>
//             <div className="flex items-center gap-3 mb-2">
//               <TrendingUp className="h-8 w-8 text-accent" />
//               <h3 className="text-lg font-display font-semibold text-muted-foreground">Monthly Profit</h3>
//             </div>
//             <p className="text-4xl font-bold text-foreground">{mockData.monthlyProfit}</p>
//           </Card>

//           <Card className="p-6 animate-slide-up" style={{ animationDelay: "0.2s" }}>
//             <div className="flex items-center gap-3 mb-2">
//               <BarChart3 className="h-8 w-8 text-primary" />
//               <h3 className="text-lg font-display font-semibold text-muted-foreground">Growth Rate</h3>
//             </div>
//             <p className="text-4xl font-bold text-secondary">{mockData.growthRate}</p>
//           </Card>

//           <Card className="p-6 animate-slide-up" style={{ animationDelay: "0.3s" }}>
//             <div className="flex items-center gap-3 mb-2">
//               <Calendar className="h-8 w-8 text-accent" />
//               <h3 className="text-lg font-display font-semibold text-muted-foreground">Daily Customers</h3>
//             </div>
//             <p className="text-4xl font-bold text-foreground">{mockData.customerCount}</p>
//           </Card>
//         </div>

//         <Card className="p-6 mb-8">
//           <h3 className="text-xl font-display font-semibold mb-4 text-foreground">Performance Chart</h3>
//           <div className="h-64 bg-muted rounded-lg flex items-center justify-center border border-border">
//             <p className="text-muted-foreground">Income & Profit Trends (Chart Placeholder)</p>
//           </div>
//         </Card>

//         <Card className="p-6 bg-secondary/10 border-secondary/20">
//           <h3 className="text-xl font-display font-semibold mb-3 text-foreground">Insights & Recommendations</h3>
//           <ul className="space-y-2 text-muted-foreground">
//             <li>• Peak hours: 12 PM - 2 PM and 6 PM - 8 PM</li>
//             <li>• Customer retention rate is above average</li>
//             <li>• Consider expanding menu/inventory during weekends</li>
//             <li>• Location foot traffic is growing steadily</li>
//           </ul>
//         </Card>

//         <div className="mt-8 flex gap-4">
//           <Button variant="outline" onClick={() => navigate("/selection")}>
//             Back to Selection
//           </Button>
//           <Button onClick={() => navigate("/shop-analysis")} className="bg-accent text-accent-foreground hover:bg-accent/90">
//             New Analysis
//           </Button>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default AnalysisResults;

/////////tsx- > jsx

// import { useLocation, useNavigate } from "react-router-dom";
// import { Button } from "@/components/ui/button";
// import { Card } from "@/components/ui/card";
// import { BarChart3, TrendingUp, DollarSign, Calendar } from "lucide-react";

// const AnalysisResults = () => {
//   const location = useLocation();
//   const navigate = useNavigate();
//   const { shopId } = location.state || {};

//   const mockData = {
//     dailyIncome: "$1,250",
//     monthlyProfit: "$28,400",
//     growthRate: "+15%",
//     customerCount: "156/day",
//   };

//   return (
//     <div className="min-h-screen bg-background px-4 py-12">
//       <div className="container mx-auto max-w-4xl">
//         <div className="mb-8 animate-fade-in">
//           <h1 className="text-4xl font-display font-bold mb-2 text-foreground">
//             Shop Performance Dashboard
//           </h1>
//           <p className="text-xl text-muted-foreground">Shop ID: {shopId}</p>
//         </div>

//         <div className="grid md:grid-cols-2 gap-6 mb-8">
//           <Card className="p-6 animate-slide-up">
//             <div className="flex items-center gap-3 mb-2">
//               <DollarSign className="h-8 w-8 text-secondary" />
//               <h3 className="text-lg font-display font-semibold text-muted-foreground">Daily Income</h3>
//             </div>
//             <p className="text-4xl font-bold text-foreground">{mockData.dailyIncome}</p>
//           </Card>

//           <Card className="p-6 animate-slide-up" style={{ animationDelay: "0.1s" }}>
//             <div className="flex items-center gap-3 mb-2">
//               <TrendingUp className="h-8 w-8 text-accent" />
//               <h3 className="text-lg font-display font-semibold text-muted-foreground">Monthly Profit</h3>
//             </div>
//             <p className="text-4xl font-bold text-foreground">{mockData.monthlyProfit}</p>
//           </Card>

//           <Card className="p-6 animate-slide-up" style={{ animationDelay: "0.2s" }}>
//             <div className="flex items-center gap-3 mb-2">
//               <BarChart3 className="h-8 w-8 text-primary" />
//               <h3 className="text-lg font-display font-semibold text-muted-foreground">Growth Rate</h3>
//             </div>
//             <p className="text-4xl font-bold text-secondary">{mockData.growthRate}</p>
//           </Card>

//           <Card className="p-6 animate-slide-up" style={{ animationDelay: "0.3s" }}>
//             <div className="flex items-center gap-3 mb-2">
//               <Calendar className="h-8 w-8 text-accent" />
//               <h3 className="text-lg font-display font-semibold text-muted-foreground">Daily Customers</h3>
//             </div>
//             <p className="text-4xl font-bold text-foreground">{mockData.customerCount}</p>
//           </Card>
//         </div>

//         <Card className="p-6 mb-8">
//           <h3 className="text-xl font-display font-semibold mb-4 text-foreground">Performance Chart</h3>
//           <div className="h-64 bg-muted rounded-lg flex items-center justify-center border border-border">
//             <p className="text-muted-foreground">Income & Profit Trends (Chart Placeholder)</p>
//           </div>
//         </Card>

//         <Card className="p-6 bg-secondary/10 border-secondary/20">
//           <h3 className="text-xl font-display font-semibold mb-3 text-foreground">Insights & Recommendations</h3>
//           <ul className="space-y-2 text-muted-foreground">
//             <li>• Peak hours: 12 PM - 2 PM and 6 PM - 8 PM</li>
//             <li>• Customer retention rate is above average</li>
//             <li>• Consider expanding menu/inventory during weekends</li>
//             <li>• Location foot traffic is growing steadily</li>
//           </ul>
//         </Card>

//         <div className="mt-8 flex gap-4">
//           <Button variant="outline" onClick={() => navigate("/selection")}>
//             Back to Selection
//           </Button>
//           <Button onClick={() => navigate("/shop-analysis")} className="bg-accent text-accent-foreground hover:bg-accent/90">
//             New Analysis
//           </Button>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default AnalysisResults;



import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { BarChart3, TrendingUp, DollarSign, Calendar } from "lucide-react";
import { auth } from "@/firebase/firebase";
import {
  collection,
  doc,
  getDocs,
  setDoc,
  Timestamp,
} from "firebase/firestore";
import { db } from "@/firebase/firebase";

const AnalysisResults = () => {
  const navigate = useNavigate();
  const user = auth.currentUser;

  const [metrics, setMetrics] = useState({
    dailyIncome: 0,
    monthlyProfit: 0,
    growthRate: 0,
    dailyCustomers: 0,
  });

  const [todayData, setTodayData] = useState({
    income: "",
    customers: "",
    expenses: "",
  });

  const shopId = user?.uid; // simple 1 user = 1 shop

  // 🔹 Fetch metrics
  useEffect(() => {
    if (!shopId) return;

    const fetchMetrics = async () => {
      const metricsRef = collection(db, "shops", shopId, "daily_metrics");
      const snapshot = await getDocs(metricsRef);

      let totalIncome = 0;
      let totalCustomers = 0;
      let count = 0;

      snapshot.forEach((doc) => {
        const data = doc.data();
        totalIncome += data.income || 0;
        totalCustomers += data.customers || 0;
        count++;
      });

      setMetrics({
        dailyIncome: count ? totalIncome / count : 0,
        monthlyProfit: totalIncome,
        growthRate: count > 1 ? 15 : 0, // simple placeholder logic
        dailyCustomers: count ? totalCustomers / count : 0,
      });
    };

    fetchMetrics();
  }, [shopId]);

  // 🔹 Save today's data
  const saveTodayData = async () => {
    if (!shopId) return;

    const today = new Date().toISOString().split("T")[0];

    await setDoc(doc(db, "shops", shopId, "daily_metrics", today), {
      income: Number(todayData.income),
      customers: Number(todayData.customers),
      expenses: Number(todayData.expenses || 0),
      createdAt: Timestamp.now(),
    });

    window.location.reload();
  };

  return (
    <div className="min-h-screen bg-background px-4 py-12">
      <div className="container mx-auto max-w-4xl">

        <h1 className="text-4xl font-display font-bold mb-2">
          Shop Performance Dashboard
        </h1>
        <p className="text-muted-foreground mb-8">Shop ID: {shopId}</p>

        {/* METRIC CARDS */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <Card className="p-6">
            <DollarSign className="h-8 w-8 text-secondary mb-2" />
            <p className="text-sm text-muted-foreground">Daily Income</p>
            <p className="text-3xl font-bold">
              ₹{metrics.dailyIncome.toFixed(0)}
            </p>
          </Card>

          <Card className="p-6">
            <TrendingUp className="h-8 w-8 text-accent mb-2" />
            <p className="text-sm text-muted-foreground">Monthly Profit</p>
            <p className="text-3xl font-bold">
              ₹{metrics.monthlyProfit.toFixed(0)}
            </p>
          </Card>

          <Card className="p-6">
            <BarChart3 className="h-8 w-8 text-primary mb-2" />
            <p className="text-sm text-muted-foreground">Growth Rate</p>
            <p className="text-3xl font-bold text-secondary">
              +{metrics.growthRate}%
            </p>
          </Card>

          <Card className="p-6">
            <Calendar className="h-8 w-8 text-accent mb-2" />
            <p className="text-sm text-muted-foreground">Daily Customers</p>
            <p className="text-3xl font-bold">
              {metrics.dailyCustomers.toFixed(0)}/day
            </p>
          </Card>
        </div>

        {/* ADD TODAY DATA */}
        <Card className="p-6 mb-8">
          <h3 className="text-lg font-semibold mb-4">Add Today’s Data</h3>

          <div className="grid md:grid-cols-3 gap-4">
            <input
              placeholder="Income"
              className="border p-2 rounded"
              value={todayData.income}
              onChange={(e) =>
                setTodayData({ ...todayData, income: e.target.value })
              }
            />
            <input
              placeholder="Customers"
              className="border p-2 rounded"
              value={todayData.customers}
              onChange={(e) =>
                setTodayData({ ...todayData, customers: e.target.value })
              }
            />
            <input
              placeholder="Expenses"
              className="border p-2 rounded"
              value={todayData.expenses}
              onChange={(e) =>
                setTodayData({ ...todayData, expenses: e.target.value })
              }
            />
          </div>

          <Button className="mt-4" onClick={saveTodayData}>
            Save Today’s Data
          </Button>
        </Card>

        {/* NAV */}
        <div className="flex gap-4">
          <Button variant="outline" onClick={() => navigate("/selection")}>
            Back
          </Button>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults;
