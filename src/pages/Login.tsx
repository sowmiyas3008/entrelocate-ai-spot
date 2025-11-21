import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { MapPin } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const Login = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    toast({
      title: "Welcome back!",
      description: "Successfully logged in",
    });
    navigate("/selection");
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4">
      <Card className="w-full max-w-md p-8 animate-fade-in">
        <div className="flex items-center justify-center gap-2 mb-6">
          <MapPin className="h-8 w-8 text-secondary" />
          <span className="text-2xl font-display font-bold text-foreground">EntreLocate</span>
        </div>
        
        <h1 className="text-3xl font-display font-bold text-center mb-2 text-foreground">Welcome Back</h1>
        <p className="text-center text-muted-foreground mb-6">Login to access your dashboard</p>
        
        <form onSubmit={handleSubmit} className="space-y-4">
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
          
          <div>
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              placeholder="••••••••"
              value={formData.password}
              onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              required
              className="mt-1"
            />
          </div>
          
          <Button type="submit" className="w-full bg-accent text-accent-foreground hover:bg-accent/90 font-semibold">
            Login
          </Button>
        </form>
        
        <p className="text-center text-sm text-muted-foreground mt-6">
          Don't have an account?{" "}
          <Link to="/signup" className="text-secondary hover:underline font-medium">
            Sign Up
          </Link>
        </p>
      </Card>
    </div>
  );
};

export default Login;
