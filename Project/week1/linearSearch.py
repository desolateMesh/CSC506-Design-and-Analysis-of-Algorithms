class MarketplaceItem:
    def __init__(self, id, name, price, category, description, brand, in_stock=True):
        self.id = id
        self.name = name
        self.price = price
        self.category = category
        self.description = description
        self.brand = brand
        self.in_stock = in_stock

    def __str__(self):
        stock_status = "In Stock" if self.in_stock else "Out of Stock"
        return (f"Item: {self.name}\n"
                f"Brand: {self.brand}\n"
                f"Price: ${self.price:.2f}\n"
                f"Category: {self.category}\n"
                f"Status: {stock_status}\n"
                f"Description: {self.description}")

class FlyFishingMarketplace:
    def __init__(self):
        self.items = []
        self.categories = set()
        self.brands = set()
    
    def add_item(self, item):
        if not isinstance(item, MarketplaceItem):
            raise ValueError("Can only add MarketplaceItem objects")
        
        self.items.append(item)
        self.categories.add(item.category)
        self.brands.add(item.brand)
    
    def linear_search(self, search_term, filters=None):
        if not search_term and not filters:
            raise ValueError("Must provide either search term or filters")
        
        results = []
        comparisons = 0
        search_term = search_term.lower() if search_term else ""
        
        for item in self.items:
            comparisons += 1
            
            matches = not search_term or (
                search_term in item.name.lower() or
                search_term in item.description.lower() or
                search_term in item.category.lower() or
                search_term in item.brand.lower()
            )
            
            if matches and filters:
                if 'category' in filters and filters['category']:
                    matches = matches and item.category == filters['category']
                if 'brand' in filters and filters['brand']:
                    matches = matches and item.brand == filters['brand']
                if 'price_range' in filters and filters['price_range']:
                    min_price, max_price = filters['price_range']
                    matches = matches and min_price <= item.price <= max_price
                if 'in_stock' in filters:
                    matches = matches and item.in_stock == filters['in_stock']
            
            if matches:
                results.append(item)
        
        return results, comparisons
    
    def get_categories(self):
        """Return all available categories"""
        return sorted(list(self.categories))
    
    def get_brands(self):
        """Return all available brands"""
        return sorted(list(self.brands))

def main():
    # Initialize marketplace
    shop = FlyFishingMarketplace()
    
    # Add sample inventory
    sample_items = [
        MarketplaceItem(1, "Sage X Fly Rod", 899.99, "Rods", 
                       "Premium fast-action fly rod, 9' 5-weight", "Sage"),
        MarketplaceItem(2, "Orvis Hydros Fly Line", 79.99, "Lines", 
                       "Weight-forward floating line, ideal for trout fishing", "Orvis"),
        MarketplaceItem(3, "RIO Gold Fly Line", 89.99, "Lines",
                       "Premium trout line with welded loops", "RIO"),
        MarketplaceItem(4, "Simms G3 Guide Waders", 499.99, "Apparel",
                       "Durable GORE-TEX Pro waders", "Simms"),
        MarketplaceItem(5, "Fishpond Net", 129.99, "Accessories",
                       "Carbon fiber frame with rubber net", "Fishpond"),
        MarketplaceItem(6, "Patagonia Fishing Vest", 129.99, "Apparel",
                       "Lightweight vest with multiple pockets", "Patagonia"),
        MarketplaceItem(7, "Ross Evolution LTX Reel", 385.00, "Reels",
                       "Lightweight click-pawl reel for trout", "Ross"),
        MarketplaceItem(8, "Elk Hair Caddis #14", 3.99, "Flies",
                       "Classic dry fly pattern", "Umpqua", True),
        MarketplaceItem(9, "Sage Foundation Rod", 450.00, "Rods",
                       "All-water rod, 9' 6-weight", "Sage", False),
        MarketplaceItem(10, "Scientific Anglers Leader", 5.99, "Leaders",
                       "9ft 5X Trout Leader", "Scientific Anglers"),
        MarketplaceItem(11, "Echo Base Rod", 179.99, "Rods",
                       "Perfect beginner rod, 9' 5-weight", "Echo", True),
        MarketplaceItem(12, "Redington Path Rod", 149.99, "Rods",
                       "Versatile medium-action rod, 9' 5-weight", "Redington", True)
    ]
    
    for item in sample_items:
        shop.add_item(item)
    
    print("\nAvailable Categories:", shop.get_categories())
    print("Available Brands:", shop.get_brands())
    
    print("\n1. Basic Search for 'Sage':")
    results, comparisons = shop.linear_search("Sage")
    print(f"Found {len(results)} results in {comparisons} comparisons:")
    for item in results:
        print(f"\n{item}")
        print("-" * 50)
    
    print("\n2. Filtered Search for in-stock rods under $500:")
    filters = {
        'category': 'Rods',
        'price_range': (0, 500),
        'in_stock': True
    }
    results, comparisons = shop.linear_search("", filters=filters)
    print(f"Found {len(results)} results in {comparisons} comparisons:")
    for item in results:
        print(f"\n{item}")
        print("-" * 50)

if __name__ == "__main__":
    main()