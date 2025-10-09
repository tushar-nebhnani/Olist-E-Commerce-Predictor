import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "./ThemeToggle";
import { ChevronDown, Menu } from "lucide-react";
import { useState } from "react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

const Navigation = () => {
  const location = useLocation();
  const [open, setOpen] = useState(false);

  const navLinks = [
    { path: "/", label: "Home" },
  ];

  // Links for the satisfaction predictor dropdown
  const satisfactionVersions = [
    { path: "/satisfaction-predictor-v1", label: "Baseline (V1)" },
    { path: "/satisfaction-predictor-final", label: "Final (XGBoost)" },
  ];
  
  // Links for the purchase predictor dropdown
  const purchaseVersions = [
    { path: "/purchase-prediction-v1", label: "Baseline Model" },
    { path: "/purchase-prediction-v2", label: "Advanced Model" },
  ];

  // Other top-level links
  const additionalLinks = [
    { path: "/business-insights", label: "Business Insights" },
    { path: "/product-recommendation", label: "Product Recommendations" },
  ];

  const isSatisfactionActive = location.pathname.startsWith("/satisfaction-predictor");
  const isPurchaseActive = location.pathname.startsWith("/purchase-prediction");

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center space-x-2">
          <h1 className="text-xl font-bold bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent">
            Olist ML Insights Dashboard
          </h1>
        </Link>
        
        {/* Desktop Navigation */}
        <div className="hidden lg:flex items-center space-x-1">
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
                  isSatisfactionActive
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
              {satisfactionVersions.map((version) => (
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

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                className={cn(
                  "px-4 py-2 rounded-md text-sm font-medium transition-all group",
                  isPurchaseActive
                    ? "text-primary bg-primary/10"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                )}
              >
                Purchase Predictor
                <ChevronDown className="ml-1 h-4 w-4 transition-transform group-data-[state=open]:rotate-180" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent 
              align="end" 
              className="w-56 bg-background/95 backdrop-blur-sm border-border/40"
            >
              {purchaseVersions.map((version) => (
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

        {/* Mobile Navigation */}
        <div className="flex lg:hidden items-center gap-2">
          <ThemeToggle />
          
          <Sheet open={open} onOpenChange={setOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon">
                <Menu className="h-5 w-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-80">
              <div className="flex flex-col gap-2 mt-8">
                {navLinks.map((link) => (
                  <Link
                    key={link.path}
                    to={link.path}
                    onClick={() => setOpen(false)}
                    className={cn(
                      "px-4 py-3 rounded-md text-base font-medium transition-colors",
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
                    onClick={() => setOpen(false)}
                    className={cn(
                      "px-4 py-3 rounded-md text-base font-medium transition-colors",
                      location.pathname === link.path
                        ? "text-primary bg-primary/10"
                        : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                    )}
                  >
                    {link.label}
                  </Link>
                ))}
                
                <Collapsible>
                  <CollapsibleTrigger asChild>
                    <Button
                      variant="ghost"
                      className={cn(
                        "w-full justify-between px-4 py-3 rounded-md text-base font-medium",
                        isSatisfactionActive
                          ? "text-primary bg-primary/10"
                          : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                      )}
                    >
                      Satisfaction Predictor
                      <ChevronDown className="h-4 w-4" />
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="ml-4 mt-1 space-y-1">
                    {satisfactionVersions.map((version) => (
                      <Link
                        key={version.path}
                        to={version.path}
                        onClick={() => setOpen(false)}
                        className={cn(
                          "block px-4 py-2 rounded-md text-sm transition-colors",
                          location.pathname === version.path
                            ? "text-primary bg-primary/10 font-medium"
                            : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                        )}
                      >
                        {version.label}
                      </Link>
                    ))}
                  </CollapsibleContent>
                </Collapsible>

                <Collapsible>
                  <CollapsibleTrigger asChild>
                    <Button
                      variant="ghost"
                      className={cn(
                        "w-full justify-between px-4 py-3 rounded-md text-base font-medium",
                        isPurchaseActive
                          ? "text-primary bg-primary/10"
                          : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                      )}
                    >
                      Purchase Predictor
                      <ChevronDown className="h-4 w-4" />
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="ml-4 mt-1 space-y-1">
                    {purchaseVersions.map((version) => (
                      <Link
                        key={version.path}
                        to={version.path}
                        onClick={() => setOpen(false)}
                        className={cn(
                          "block px-4 py-2 rounded-md text-sm transition-colors",
                          location.pathname === version.path
                            ? "text-primary bg-primary/10 font-medium"
                            : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                        )}
                      >
                        {version.label}
                      </Link>
                    ))}
                  </CollapsibleContent>
                </Collapsible>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
