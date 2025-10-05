import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "./ThemeToggle";
import { ChevronDown } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";

const Navigation = () => {
  const location = useLocation();

  const navLinks = [
    { path: "/", label: "Home" },
  ];

  const predictorVersions = [
    { path: "/satisfaction-predictor-v1", label: "Version 1" },
    { path: "/satisfaction-predictor-v2", label: "Version 2" },
  ];

  const additionalLinks = [
    { path: "/customer-segmentation", label: "Customer Segmentation" },
    { path: "/delivery-prediction", label: "Delivery Prediction" },
    { path: "/eda", label: "EDA" },
  ];

  const isPredictorActive = location.pathname.startsWith("/satisfaction-predictor");

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center space-x-2">
          <h1 className="text-xl font-bold bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent">
            Olist ML Insights Dashboard
          </h1>
        </Link>
        
        <div className="flex items-center space-x-1">
          {navLinks.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              className={cn(
                "px-4 py-2 rounded-md text-sm font-medium transition-colors",
                location.pathname === link.path
                  ? "text-primary bg-primary/10"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
              )}
            >
              {link.label}
            </Link>
          ))}
          
          {additionalLinks.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              className={cn(
                "px-4 py-2 rounded-md text-sm font-medium transition-colors",
                location.pathname === link.path
                  ? "text-primary bg-primary/10"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
              )}
            >
              {link.label}
            </Link>
          ))}
          
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                className={cn(
                  "px-4 py-2 rounded-md text-sm font-medium transition-all group",
                  isPredictorActive
                    ? "text-primary bg-primary/10"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                )}
              >
                Satisfaction Predictor
                <ChevronDown className="ml-1 h-4 w-4 transition-transform group-data-[state=open]:rotate-180" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent 
              align="end" 
              className="w-56 bg-background/95 backdrop-blur-sm border-border/40"
            >
              {predictorVersions.map((version) => (
                <DropdownMenuItem key={version.path} asChild>
                  <Link
                    to={version.path}
                    className={cn(
                      "cursor-pointer transition-colors",
                      location.pathname === version.path
                        ? "text-primary bg-primary/10 font-medium"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {version.label}
                  </Link>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          
          <div className="ml-2">
            <ThemeToggle />
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
