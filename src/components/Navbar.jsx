// import { Link, useLocation } from "react-router-dom";
// import { Button } from "@/components/ui/button";
// import { MapPin } from "lucide-react";

// const Navbar = () => {
//   const location = useLocation();
  
//   return (
//     <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b border-border">
//       <div className="container mx-auto px-4">
//         <div className="flex items-center justify-between h-16">
//           <Link to="/" className="flex items-center gap-2 group">
//             <MapPin className="h-6 w-6 text-secondary transition-transform group-hover:scale-110" />
//             <span className="text-xl font-display font-bold text-foreground">EntreLocate</span>
//           </Link>
          
//           <div className="hidden md:flex items-center gap-6">
//             <Link 
//               to="/" 
//               className={`text-sm font-medium transition-colors hover:text-secondary ${
//                 location.pathname === "/" ? "text-secondary" : "text-muted-foreground"
//               }`}
//             >
//               Home
//             </Link>
//             <a 
//               href="#about" 
//               className="text-sm font-medium text-muted-foreground transition-colors hover:text-secondary"
//             >
//               About
//             </a>
//             <a 
//               href="#contact" 
//               className="text-sm font-medium text-muted-foreground transition-colors hover:text-secondary"
//             >
//               Contact
//             </a>
//           </div>
          
//           <div className="flex items-center gap-3">
//             <Link to="/login">
//               <Button variant="ghost" size="sm" className="text-foreground hover:text-secondary">
//                 Login
//               </Button>
//             </Link>
//             <Link to="/signup">
//               <Button size="sm" className="bg-accent text-accent-foreground hover:bg-accent/90">
//                 Sign Up
//               </Button>
//             </Link>
//           </div>
//         </div>
//       </div>
//     </nav>
//   );
// };

// export default Navbar;

import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { MapPin } from "lucide-react";

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b border-border">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center gap-2 group">
            <MapPin className="h-6 w-6 text-secondary transition-transform group-hover:scale-110" />
            <span className="text-xl font-display font-bold text-foreground">
              EntreLocate
            </span>
          </Link>

          <div className="hidden md:flex items-center gap-6">
            <Link
              to="/"
              className={`text-sm font-medium transition-colors hover:text-secondary ${
                location.pathname === "/"
                  ? "text-secondary"
                  : "text-muted-foreground"
              }`}
            >
              Home
            </Link>

            <a
              href="#about"
              className="text-sm font-medium text-muted-foreground transition-colors hover:text-secondary"
            >
              About
            </a>

            <a
              href="#contact"
              className="text-sm font-medium text-muted-foreground transition-colors hover:text-secondary"
            >
              Contact
            </a>
          </div>

          <div className="flex items-center gap-3">
            <Link to="/login">
              <Button
                variant="ghost"
                size="sm"
                className="text-foreground hover:text-secondary"
              >
                Login
              </Button>
            </Link>

            <Link to="/signup">
              <Button
                size="sm"
                className="bg-accent text-accent-foreground hover:bg-accent/90"
              >
                Sign Up
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

