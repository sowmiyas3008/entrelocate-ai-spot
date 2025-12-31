// import { Link } from "react-router-dom";
// import { Button } from "@/components/ui/button";
// import { Card } from "@/components/ui/card";
// import Navbar from "@/components/Navbar";
// import { MapPin, TrendingUp, BarChart3, Mail } from "lucide-react";

// const Index = () => {
//   return (
//     <div className="min-h-screen bg-background">
//       <Navbar />
      
//       {/* Hero Section */}
//       <section className="pt-32 pb-20 px-4">
//         <div className="container mx-auto text-center">
//           <div className="animate-fade-in">
//             <h1 className="text-5xl md:text-7xl font-display font-bold mb-6 text-foreground">
//               Find Your Perfect
//               <span className="block text-secondary mt-2">Business Location</span>
//             </h1>
//             <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
//               AI-powered insights to help you discover the ideal location to start or grow your business
//             </p>
//             <div className="flex gap-4 justify-center">
//               <Link to="/signup">
//                 <Button size="lg" className="bg-accent text-accent-foreground hover:bg-accent/90 font-semibold">
//                   Get Started
//                 </Button>
//               </Link>
//               <Link to="/login">
//                 <Button size="lg" variant="outline" className="font-semibold">
//                   Login
//                 </Button>
//               </Link>
//             </div>
//           </div>
//         </div>
//       </section>

//       {/* Features */}
//       <section className="py-20 px-4 bg-muted/30">
//         <div className="container mx-auto">
//           <h2 className="text-3xl md:text-4xl font-display font-bold text-center mb-12 text-foreground">
//             Powered by AI Location Intelligence
//           </h2>
//           <div className="grid md:grid-cols-3 gap-8">
//             <Card className="p-6 hover:shadow-lg transition-all hover:-translate-y-1 border-border">
//               <MapPin className="h-12 w-12 text-secondary mb-4" />
//               <h3 className="text-xl font-display font-semibold mb-3 text-foreground">Smart Location Search</h3>
//               <p className="text-muted-foreground">
//                 Find the best areas based on your business category and target market
//               </p>
//             </Card>
//             <Card className="p-6 hover:shadow-lg transition-all hover:-translate-y-1 border-border">
//               <TrendingUp className="h-12 w-12 text-secondary mb-4" />
//               <h3 className="text-xl font-display font-semibold mb-3 text-foreground">Business Type Analysis</h3>
//               <p className="text-muted-foreground">
//                 Discover which business types thrive in your chosen location
//               </p>
//             </Card>
//             <Card className="p-6 hover:shadow-lg transition-all hover:-translate-y-1 border-border">
//               <BarChart3 className="h-12 w-12 text-secondary mb-4" />
//               <h3 className="text-xl font-display font-semibold mb-3 text-foreground">Performance Insights</h3>
//               <p className="text-muted-foreground">
//                 Track daily income, profit trends, and business analytics
//               </p>
//             </Card>
//           </div>
//         </div>
//       </section>

//       {/* About Section */}
//       <section id="about" className="py-20 px-4">
//         <div className="container mx-auto max-w-4xl">
//           <h2 className="text-3xl md:text-4xl font-display font-bold text-center mb-8 text-foreground">
//             About EntreLocate
//           </h2>
//           <p className="text-lg text-muted-foreground text-center leading-relaxed">
//             EntreLocate uses advanced AI algorithms and location data to help entrepreneurs and business owners
//             make informed decisions about where to establish their business. Our platform analyzes foot traffic,
//             demographics, competition, and market trends to provide actionable insights that drive success.
//           </p>
//         </div>
//       </section>

//       {/* Contact Section */}
//       <section id="contact" className="py-20 px-4 bg-muted/30">
//         <div className="container mx-auto max-w-2xl text-center">
//           <Mail className="h-12 w-12 text-secondary mx-auto mb-4" />
//           <h2 className="text-3xl md:text-4xl font-display font-bold mb-4 text-foreground">Get in Touch</h2>
//           <p className="text-lg text-muted-foreground mb-6">
//             Have questions? We'd love to hear from you.
//           </p>
//           <a href="mailto:contact@entrelocate.com">
//             <Button size="lg" className="bg-secondary text-secondary-foreground hover:bg-secondary/90 font-semibold">
//               Contact Us
//             </Button>
//           </a>
//         </div>
//       </section>

//       {/* Footer */}
//       <footer className="py-8 px-4 border-t border-border">
//         <div className="container mx-auto text-center text-muted-foreground">
//           <p>&copy; 2024 EntreLocate. All rights reserved.</p>
//         </div>
//       </footer>
//     </div>
//   );
// };

// export default Index;
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import Navbar from "@/components/Navbar";
import { MapPin, TrendingUp, BarChart3, Mail } from "lucide-react";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      
      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4">
        <div className="container mx-auto text-center">
          <div className="animate-fade-in">
            <h1 className="text-5xl md:text-7xl font-display font-bold mb-6 text-foreground">
              Find Your Perfect
              <span className="block text-secondary mt-2">Business Location</span>
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
              AI-powered insights to help you discover the ideal location to start or grow your business
            </p>
            <div className="flex gap-4 justify-center">
              <Link to="/signup">
                <Button size="lg" className="bg-accent text-accent-foreground hover:bg-accent/90 font-semibold">
                  Get Started
                </Button>
              </Link>
              <Link to="/login">
                <Button size="lg" variant="outline" className="font-semibold">
                  Login
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-20 px-4 bg-muted/30">
        <div className="container mx-auto">
          <h2 className="text-3xl md:text-4xl font-display font-bold text-center mb-12 text-foreground">
            Powered by AI Location Intelligence
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <Card className="p-6 hover:shadow-lg transition-all hover:-translate-y-1 border-border">
              <MapPin className="h-12 w-12 text-secondary mb-4" />
              <h3 className="text-xl font-display font-semibold mb-3 text-foreground">Smart Location Search</h3>
              <p className="text-muted-foreground">
                Find the best areas based on your business category and target market
              </p>
            </Card>
            <Card className="p-6 hover:shadow-lg transition-all hover:-translate-y-1 border-border">
              <TrendingUp className="h-12 w-12 text-secondary mb-4" />
              <h3 className="text-xl font-display font-semibold mb-3 text-foreground">Business Type Analysis</h3>
              <p className="text-muted-foreground">
                Discover which business types thrive in your chosen location
              </p>
            </Card>
            <Card className="p-6 hover:shadow-lg transition-all hover:-translate-y-1 border-border">
              <BarChart3 className="h-12 w-12 text-secondary mb-4" />
              <h3 className="text-xl font-display font-semibold mb-3 text-foreground">Performance Insights</h3>
              <p className="text-muted-foreground">
                Track daily income, profit trends, and business analytics
              </p>
            </Card>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-20 px-4">
        <div className="container mx-auto max-w-4xl">
          <h2 className="text-3xl md:text-4xl font-display font-bold text-center mb-8 text-foreground">
            About EntreLocate
          </h2>
          <p className="text-lg text-muted-foreground text-center leading-relaxed">
            EntreLocate uses advanced AI algorithms and location data to help entrepreneurs and business owners
            make informed decisions about where to establish their business. Our platform analyzes foot traffic,
            demographics, competition, and market trends to provide actionable insights that drive success.
          </p>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="py-20 px-4 bg-muted/30">
        <div className="container mx-auto max-w-2xl text-center">
          <Mail className="h-12 w-12 text-secondary mx-auto mb-4" />
          <h2 className="text-3xl md:text-4xl font-display font-bold mb-4 text-foreground">Get in Touch</h2>
          <p className="text-lg text-muted-foreground mb-6">
            Have questions? We'd love to hear from you.
          </p>
          <a href="mailto:itsmemiya30@gmail.com">
            <Button size="lg" className="bg-secondary text-secondary-foreground hover:bg-secondary/90 font-semibold">
              Contact Us
            </Button>
          </a>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-border">
        <div className="container mx-auto text-center text-muted-foreground">
          <p>&copy; 2026 EntreLocate. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
