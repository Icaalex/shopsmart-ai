"""
data.py — Generates a realistic synthetic e-commerce product dataset
and simulated user purchase/rating history.

As a beginner, think of this file as your "fake database".
In a real product you'd replace this with real data from a database or CSV.
"""

import pandas as pd
import numpy as np

# Fix the random seed so results are consistent every run
np.random.seed(42)


def generate_products() -> pd.DataFrame:
    """
    Returns a DataFrame of 120 products across 6 categories.
    Each product has features the ML model will use for recommendations.
    """

    categories = {
        "Electronics": {
            "items": [
                ("Wireless Bluetooth Headphones", 79.99, ["audio", "wireless", "music", "bluetooth", "portable"]),
                ("Noise-Cancelling Earbuds", 129.99, ["audio", "wireless", "music", "noise-cancelling", "earbuds"]),
                ("Mechanical Keyboard", 89.99, ["keyboard", "gaming", "typing", "rgb", "mechanical"]),
                ("USB-C Hub 7-in-1", 39.99, ["usb", "hub", "laptop", "portable", "adapter"]),
                ("Portable Charger 20000mAh", 49.99, ["charging", "battery", "portable", "power-bank"]),
                ("Smart Watch Fitness Band", 199.99, ["smartwatch", "fitness", "health", "wearable", "bluetooth"]),
                ("4K Webcam", 89.99, ["webcam", "video", "streaming", "4k", "camera"]),
                ("Gaming Mouse", 59.99, ["gaming", "mouse", "rgb", "wireless", "precision"]),
                ("LED Monitor 27inch", 299.99, ["monitor", "display", "4k", "gaming", "screen"]),
                ("Mini Bluetooth Speaker", 34.99, ["audio", "bluetooth", "portable", "speaker", "music"]),
                ("Wireless Charging Pad", 29.99, ["charging", "wireless", "smartphone", "fast-charge"]),
                ("Laptop Stand Adjustable", 45.99, ["laptop", "ergonomic", "stand", "portable", "aluminum"]),
                ("Ring Light 18 inch", 54.99, ["lighting", "photography", "streaming", "camera", "selfie"]),
                ("Raspberry Pi 5", 79.99, ["computing", "programming", "electronics", "diy", "projects"]),
                ("Smart Home Plug", 19.99, ["smart-home", "wifi", "automation", "energy", "alexa"]),
                ("External SSD 1TB", 119.99, ["storage", "ssd", "portable", "fast", "backup"]),
                ("VR Headset Entry Level", 349.99, ["vr", "gaming", "virtual-reality", "immersive", "headset"]),
                ("Desk Lamp with USB", 32.99, ["lighting", "desk", "usb", "led", "eye-care"]),
                ("Cable Management Kit", 14.99, ["organization", "cables", "desk", "tidy", "home-office"]),
                ("Screen Protector Kit", 12.99, ["protection", "screen", "smartphone", "scratch", "film"]),
            ]
        },
        "Fashion": {
            "items": [
                ("Classic White Sneakers", 64.99, ["shoes", "casual", "fashion", "white", "sneakers"]),
                ("Leather Crossbody Bag", 89.99, ["bag", "leather", "fashion", "handbag", "accessories"]),
                ("Slim Fit Chino Pants", 54.99, ["pants", "fashion", "casual", "slim", "chino"]),
                ("Oversized Cotton Hoodie", 44.99, ["hoodie", "casual", "cotton", "streetwear", "comfortable"]),
                ("Minimalist Watch", 149.99, ["watch", "fashion", "minimalist", "leather", "accessories"]),
                ("Denim Jacket", 79.99, ["jacket", "denim", "casual", "fashion", "outerwear"]),
                ("Running Shoes", 99.99, ["shoes", "running", "sport", "athletic", "fitness"]),
                ("Canvas Backpack 25L", 59.99, ["backpack", "bag", "canvas", "travel", "casual"]),
                ("Polarized Sunglasses", 34.99, ["sunglasses", "fashion", "accessories", "uv-protection", "outdoor"]),
                ("Graphic Print T-Shirt", 24.99, ["tshirt", "casual", "fashion", "cotton", "streetwear"]),
                ("Waterproof Rain Jacket", 89.99, ["jacket", "outdoor", "waterproof", "rain", "hiking"]),
                ("Sports Bra", 29.99, ["sportswear", "fitness", "gym", "comfortable", "activewear"]),
                ("Compression Socks 3-pack", 19.99, ["socks", "compression", "sport", "health", "athletic"]),
                ("Leather Wallet Slim", 39.99, ["wallet", "leather", "accessories", "slim", "fashion"]),
                ("Baseball Cap", 22.99, ["cap", "hat", "casual", "fashion", "accessories"]),
                ("Yoga Pants", 49.99, ["yoga", "fitness", "activewear", "comfortable", "stretch"]),
                ("Ankle Boots", 109.99, ["boots", "shoes", "fashion", "autumn", "leather"]),
                ("Silk Scarf", 44.99, ["scarf", "fashion", "accessories", "silk", "elegant"]),
                ("Winter Gloves Touchscreen", 19.99, ["gloves", "winter", "touchscreen", "accessories", "warm"]),
                ("Swimwear Set", 54.99, ["swimwear", "beach", "summer", "fashion", "sport"]),
            ]
        },
        "Home & Kitchen": {
            "items": [
                ("Air Fryer 5L", 89.99, ["kitchen", "cooking", "air-fryer", "healthy", "appliance"]),
                ("Coffee Maker Drip", 59.99, ["coffee", "kitchen", "appliance", "morning", "brewing"]),
                ("Bamboo Cutting Board Set", 29.99, ["kitchen", "cooking", "bamboo", "eco", "cutting-board"]),
                ("Cast Iron Skillet 12\"", 44.99, ["cooking", "kitchen", "cast-iron", "oven-safe", "durable"]),
                ("Instant Pot 6Qt", 119.99, ["cooking", "kitchen", "pressure-cooker", "appliance", "meal-prep"]),
                ("Blender High Speed", 79.99, ["blender", "kitchen", "smoothie", "cooking", "appliance"]),
                ("Knife Set 15pc", 69.99, ["kitchen", "cooking", "knives", "chef", "tools"]),
                ("Non-stick Pan Set", 49.99, ["kitchen", "cooking", "non-stick", "pan", "healthy"]),
                ("Electric Kettle Glass", 34.99, ["kettle", "kitchen", "tea", "coffee", "appliance"]),
                ("Food Storage Containers", 24.99, ["storage", "kitchen", "meal-prep", "containers", "eco"]),
                ("Spice Rack Organizer", 19.99, ["kitchen", "organization", "spices", "cooking", "home"]),
                ("Silicone Baking Mats", 14.99, ["baking", "kitchen", "silicone", "eco", "cooking"]),
                ("Dish Drying Rack", 24.99, ["kitchen", "organization", "cleaning", "dishes", "home"]),
                ("Reusable Water Bottle 1L", 29.99, ["bottle", "hydration", "eco", "sport", "outdoor"]),
                ("Toaster Oven", 69.99, ["cooking", "kitchen", "toaster", "appliance", "baking"]),
                ("Herb Garden Kit Indoor", 39.99, ["gardening", "herbs", "kitchen", "indoor", "eco"]),
                ("Weighted Blanket 8kg", 64.99, ["bedding", "home", "comfort", "sleep", "weighted"]),
                ("Aromatherapy Diffuser", 29.99, ["home", "aromatherapy", "wellness", "diffuser", "relaxation"]),
                ("Decorative Plant Pots Set", 34.99, ["home", "plants", "decor", "pots", "indoor"]),
                ("Smart Thermostat", 149.99, ["smart-home", "thermostat", "energy", "wifi", "automation"]),
            ]
        },
        "Books & Education": {
            "items": [
                ("Python Crash Course 3rd Ed", 34.99, ["python", "programming", "learning", "beginner", "coding"]),
                ("Clean Code by Robert Martin", 39.99, ["programming", "software", "best-practices", "coding", "engineering"]),
                ("The Pragmatic Programmer", 44.99, ["programming", "software", "career", "engineering", "coding"]),
                ("Deep Learning with Python", 49.99, ["python", "ai", "machine-learning", "deep-learning", "data-science"]),
                ("Atomic Habits", 24.99, ["habits", "self-improvement", "productivity", "psychology", "lifestyle"]),
                ("The Lean Startup", 19.99, ["startup", "entrepreneurship", "business", "innovation", "management"]),
                ("Cracking the Coding Interview", 39.99, ["interview", "coding", "programming", "career", "algorithms"]),
                ("Designing Data-Intensive Apps", 54.99, ["systems", "programming", "databases", "engineering", "backend"]),
                ("Zero to One by Peter Thiel", 22.99, ["startup", "entrepreneurship", "business", "innovation", "strategy"]),
                ("The Psychology of Money", 17.99, ["finance", "psychology", "money", "investing", "lifestyle"]),
                ("Introduction to Algorithms", 79.99, ["algorithms", "computer-science", "programming", "data-structures", "technical"]),
                ("Sketch Notes & Journal Set", 22.99, ["stationery", "drawing", "creativity", "notes", "art"]),
                ("Flashcard Set Programming", 14.99, ["programming", "learning", "coding", "study", "flashcards"]),
                ("Online Course Gift Card", 49.99, ["learning", "online-course", "education", "skills", "programming"]),
                ("Data Science Handbook", 44.99, ["data-science", "python", "machine-learning", "statistics", "analytics"]),
                ("Business Model Canvas Pad", 19.99, ["business", "entrepreneurship", "planning", "strategy", "startup"]),
                ("English Grammar Workbook", 16.99, ["english", "grammar", "learning", "language", "writing"]),
                ("World History Illustrated", 29.99, ["history", "education", "reading", "illustrated", "knowledge"]),
                ("Digital Photography Guide", 27.99, ["photography", "camera", "digital", "learning", "art"]),
                ("The Art of Public Speaking", 21.99, ["communication", "public-speaking", "skills", "career", "self-improvement"]),
            ]
        },
        "Sports & Fitness": {
            "items": [
                ("Resistance Bands Set 5pc", 24.99, ["fitness", "gym", "resistance", "training", "home-gym"]),
                ("Yoga Mat Premium 6mm", 39.99, ["yoga", "fitness", "mat", "exercise", "non-slip"]),
                ("Adjustable Dumbbells 20kg", 149.99, ["dumbbells", "fitness", "gym", "strength", "home-gym"]),
                ("Jump Rope Speed Cable", 19.99, ["cardio", "fitness", "jump-rope", "training", "portable"]),
                ("Foam Roller Deep Tissue", 29.99, ["recovery", "fitness", "massage", "muscle", "flexibility"]),
                ("Pull-up Bar Doorframe", 34.99, ["fitness", "gym", "pull-up", "strength", "home-gym"]),
                ("Protein Shaker Bottle", 14.99, ["fitness", "nutrition", "protein", "gym", "shaker"]),
                ("Gym Gloves Workout", 16.99, ["fitness", "gym", "gloves", "training", "grip"]),
                ("Fitness Tracker Band", 49.99, ["fitness", "wearable", "health", "tracker", "steps"]),
                ("Ab Roller Wheel", 19.99, ["fitness", "core", "abs", "gym", "strength"]),
                ("Kettlebell 16kg", 54.99, ["fitness", "gym", "kettlebell", "strength", "home-gym"]),
                ("Battle Rope 10m", 69.99, ["cardio", "fitness", "battle-rope", "strength", "gym"]),
                ("Massage Gun Percussive", 89.99, ["recovery", "massage", "fitness", "muscle", "therapy"]),
                ("Cycling Gloves", 24.99, ["cycling", "sport", "gloves", "outdoor", "fitness"]),
                ("Swim Goggles Anti-fog", 19.99, ["swimming", "sport", "goggles", "fitness", "aquatic"]),
                ("Soccer Ball Size 5", 29.99, ["soccer", "sport", "football", "outdoor", "team"]),
                ("Badminton Racket Set", 49.99, ["badminton", "sport", "outdoor", "racket", "fun"]),
                ("Sports Water Bottle 750ml", 22.99, ["hydration", "sport", "bottle", "fitness", "outdoor"]),
                ("Running Belt Waist", 17.99, ["running", "sport", "belt", "outdoor", "phone-holder"]),
                ("Skipping Board Balance", 34.99, ["balance", "fitness", "core", "training", "fun"]),
            ]
        },
        "Beauty & Health": {
            "items": [
                ("Vitamin C Serum 30ml", 24.99, ["skincare", "vitamin-c", "serum", "anti-aging", "glow"]),
                ("Sunscreen SPF 50+", 18.99, ["skincare", "sunscreen", "spf", "protection", "daily"]),
                ("Electric Toothbrush", 49.99, ["dental", "oral-care", "electric", "health", "hygiene"]),
                ("Facial Roller Jade", 19.99, ["skincare", "beauty", "jade", "roller", "wellness"]),
                ("Hair Growth Supplements", 29.99, ["hair", "health", "supplements", "growth", "vitamins"]),
                ("Retinol Night Cream", 34.99, ["skincare", "retinol", "anti-aging", "night-cream", "beauty"]),
                ("Natural Deodorant", 12.99, ["deodorant", "natural", "health", "aluminum-free", "wellness"]),
                ("Eye Cream Anti-puff", 27.99, ["skincare", "eye-cream", "beauty", "anti-puff", "glow"]),
                ("Multivitamin Daily Pack", 22.99, ["health", "vitamins", "supplements", "immunity", "wellness"]),
                ("Probiotics Capsules", 24.99, ["health", "gut", "probiotics", "supplements", "wellness"]),
                ("Hyaluronic Acid Serum", 22.99, ["skincare", "hyaluronic", "hydration", "serum", "beauty"]),
                ("Lip Balm Set SPF", 14.99, ["lips", "skincare", "spf", "beauty", "protection"]),
                ("Beard Oil Grooming", 17.99, ["grooming", "beard", "men", "care", "health"]),
                ("Clay Face Mask", 16.99, ["skincare", "mask", "clay", "pores", "beauty"]),
                ("Body Lotion Shea Butter", 19.99, ["skincare", "lotion", "body", "hydration", "shea"]),
                ("Nail Care Kit", 16.99, ["nails", "beauty", "grooming", "care", "wellness"]),
                ("Essential Oil Set 6pc", 29.99, ["aromatherapy", "wellness", "essential-oils", "relaxation", "health"]),
                ("Collagen Powder", 39.99, ["collagen", "health", "supplements", "skin", "joints"]),
                ("Hand Cream Intensive", 12.99, ["skincare", "hands", "cream", "moisturizer", "beauty"]),
                ("Sleep Aid Melatonin", 14.99, ["sleep", "health", "melatonin", "supplements", "wellness"]),
            ]
        },
    }

    rows = []
    product_id = 1

    for category, data in categories.items():
        for name, price, tags in data["items"]:
            # Generate realistic ratings (most products 3.5 - 5.0)
            rating = round(np.random.uniform(3.5, 5.0), 1)
            num_reviews = np.random.randint(12, 2400)
            in_stock = np.random.choice([True, False], p=[0.85, 0.15])

            rows.append({
                "product_id": product_id,
                "name": name,
                "category": category,
                "price": price,
                "rating": rating,
                "num_reviews": num_reviews,
                "in_stock": in_stock,
                "tags": " ".join(tags),  # Space-separated for TF-IDF
                "tag_list": tags,        # List form for display
                "description": f"{name} - {category}. Tags: {', '.join(tags)}. Great quality product with {num_reviews} customer reviews.",
            })
            product_id += 1

    return pd.DataFrame(rows)


def generate_user_ratings(products_df: pd.DataFrame, n_users: int = 200) -> pd.DataFrame:
    """
    Simulates user rating history.
    Each user rates between 5 and 25 products they've bought.

    This data powers collaborative filtering —
    'users who liked X also liked Y'.
    """
    n_products = len(products_df)
    rows = []

    # Create user preference profiles (some users prefer certain categories)
    category_list = products_df["category"].unique()

    for user_id in range(1, n_users + 1):
        # Each user has 1-2 preferred categories they rate higher
        preferred_cats = np.random.choice(category_list, size=np.random.randint(1, 3), replace=False)

        # Number of products this user has rated
        n_rated = np.random.randint(5, 25)
        rated_products = np.random.choice(products_df["product_id"].values, size=n_rated, replace=False)

        for pid in rated_products:
            product = products_df[products_df["product_id"] == pid].iloc[0]
            # Users rate preferred categories higher
            if product["category"] in preferred_cats:
                rating = np.random.choice([4, 4, 4, 5, 5, 3], p=[0.3, 0.3, 0.1, 0.2, 0.05, 0.05])
            else:
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.35, 0.35, 0.15])

            rows.append({
                "user_id": user_id,
                "product_id": int(pid),
                "rating": int(rating),
            })

    return pd.DataFrame(rows)

