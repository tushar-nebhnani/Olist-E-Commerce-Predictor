import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface DatasetCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
  details?: string;
}

const DatasetCard = ({ icon: Icon, title, description, details }: DatasetCardProps) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div
      className="relative group"
      onMouseEnter={() => setIsExpanded(true)}
      onMouseLeave={() => setIsExpanded(false)}
    >
      <Card className={cn(
        "relative overflow-hidden border-2 transition-all duration-300 cursor-pointer",
        "hover:border-primary/50 hover:shadow-lg hover:shadow-primary/20",
        "bg-card/50 backdrop-blur",
        isExpanded && "scale-105 border-primary/70 shadow-xl shadow-primary/30"
      )}>
        <CardContent className="p-6">
          <div className="flex flex-col space-y-4">
            <div className={cn(
              "w-12 h-12 rounded-lg flex items-center justify-center transition-all duration-300",
              "bg-primary/10 group-hover:bg-primary/20",
              isExpanded && "bg-primary/20 scale-110"
            )}>
              <Icon className={cn(
                "w-6 h-6 transition-all duration-300",
                "text-primary",
                isExpanded && "scale-110"
              )} />
            </div>
            
            <div>
              <h3 className="text-xl font-semibold mb-2">{title}</h3>
              <p className="text-sm text-muted-foreground">{description}</p>
            </div>

            {/* Expanded content */}
            <div className={cn(
              "overflow-hidden transition-all duration-300",
              isExpanded ? "max-h-96 opacity-100" : "max-h-0 opacity-0"
            )}>
              {details && (
                <div className="pt-4 border-t border-border/50">
                  <p className="text-sm text-muted-foreground">{details}</p>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default DatasetCard;
