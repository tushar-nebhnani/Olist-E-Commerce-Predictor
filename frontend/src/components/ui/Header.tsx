const Header = () => {
  return (
    <header className="fixed top-0 left-0 right-0 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 z-50">
      <div className="container flex h-14 items-center">
        <div className="mr-4 flex">
          <a href="/" className="mr-6 flex items-center space-x-2">
            <span className="font-bold">Olist Analytics</span>
          </a>
        </div>
        <nav className="flex items-center space-x-6 text-sm font-medium">
          <a href="/" className="transition-colors hover:text-foreground/80">Home</a>
          <a href="/satisfaction-predictor-v1" className="transition-colors hover:text-foreground/80">Satisfaction Predictor V1</a>
          <a href="/satisfaction-predictor-v2" className="transition-colors hover:text-foreground/80">Satisfaction Predictor V2</a>
          <a href="/customer-segments" className="transition-colors hover:text-foreground/80">Customer Segments</a>
        </nav>
      </div>
    </header>
  );
};

export default Header;