"""
recommender.py — The brain of ShopSmart AI

This file contains two recommendation algorithms:

1. CONTENT-BASED FILTERING
   "Find products similar to this one"
   How it works: converts product tags into numerical vectors using TF-IDF,
   then measures how similar products are using cosine similarity.
   Think of it like comparing word fingerprints between products.

2. COLLABORATIVE FILTERING
   "Users who liked this also liked..."
   How it works: builds a user-product rating matrix, then finds
   products that are often rated together by the same users.

3. HYBRID RECOMMENDER
   Blends both scores for better recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class ContentBasedRecommender:
    """
    Recommends products similar to a given product
    based on category, tags, and price range.
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=500,   # Only keep the 500 most important words
            ngram_range=(1, 2), # Use single words AND pairs of words
        )
        self.similarity_matrix = None
        self.products_df = None

    def fit(self, products_df: pd.DataFrame):
        """
        Train the model.
        'Training' here just means computing all similarity scores upfront.
        """
        self.products_df = products_df.copy().reset_index(drop=True)

        # Build a rich text feature: combine category + tags + name
        # This is called "feature engineering" — making raw data useful for ML
        self.products_df["features"] = (
            self.products_df["category"] + " " +
            self.products_df["tags"] + " " +
            self.products_df["name"].str.lower()
        )

        # Convert text to TF-IDF matrix (each product = a row of numbers)
        tfidf_matrix = self.tfidf.fit_transform(self.products_df["features"])

        # Compute cosine similarity between ALL pairs of products
        # Result: 120×120 matrix where cell [i][j] = similarity score 0-1
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        return self

    def recommend(self, product_id: int, n: int = 6, min_price: float = 0, max_price: float = 9999) -> pd.DataFrame:
        """
        Returns the top-n products most similar to product_id.
        """
        # Find the row index for this product
        idx = self.products_df[self.products_df["product_id"] == product_id].index[0]

        # Get similarity scores for this product vs all others
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity, highest first, exclude the product itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [(i, score) for i, score in sim_scores if i != idx]

        # Build results dataframe
        results = []
        for i, score in sim_scores:
            product = self.products_df.iloc[i]
            if min_price <= product["price"] <= max_price:
                results.append({
                    **product.to_dict(),
                    "similarity_score": round(score, 3),
                    "recommendation_reason": "Similar product"
                })
            if len(results) >= n:
                break

        return pd.DataFrame(results) if results else pd.DataFrame()


class CollaborativeRecommender:
    """
    Recommends products based on what similar users liked.
    Uses item-item collaborative filtering.
    """

    def __init__(self):
        self.item_similarity = None
        self.user_item_matrix = None
        self.products_df = None

    def fit(self, products_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """
        Build the user-item matrix and compute item-item similarity.
        """
        self.products_df = products_df.copy()

        # Create a matrix: rows = users, columns = products, values = ratings
        # Missing values (not rated) = 0
        self.user_item_matrix = ratings_df.pivot_table(
            index="user_id",
            columns="product_id",
            values="rating",
            fill_value=0
        )

        # Compute cosine similarity between items (not users)
        # This tells us: "products that are rated similarly by the same users"
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)

        # Convert to DataFrame for easy lookup
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )

        return self

    def recommend(self, product_id: int, n: int = 6) -> pd.DataFrame:
        """
        Returns products that are frequently co-rated with product_id.
        """
        if product_id not in self.item_similarity_df.index:
            return pd.DataFrame()

        # Get similarity scores for this product
        sim_scores = self.item_similarity_df[product_id].sort_values(ascending=False)

        # Exclude the product itself
        sim_scores = sim_scores[sim_scores.index != product_id]

        results = []
        for pid, score in sim_scores.head(n * 2).items():
            product_row = self.products_df[self.products_df["product_id"] == pid]
            if not product_row.empty:
                product = product_row.iloc[0]
                results.append({
                    **product.to_dict(),
                    "similarity_score": round(score, 3),
                    "recommendation_reason": "Customers also bought"
                })
            if len(results) >= n:
                break

        return pd.DataFrame(results) if results else pd.DataFrame()


class HybridRecommender:
    """
    Combines content-based and collaborative filtering.
    This is what real e-commerce companies use (Amazon, Jumia, etc.)

    Formula: hybrid_score = (alpha × content_score) + ((1-alpha) × collab_score)
    alpha = 0.5 means equal weight. Increase to favour content-based.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.content_rec = ContentBasedRecommender()
        self.collab_rec = CollaborativeRecommender()
        self.products_df = None
        self.scaler = MinMaxScaler()

    def fit(self, products_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.products_df = products_df.copy()
        self.content_rec.fit(products_df)
        self.collab_rec.fit(products_df, ratings_df)
        return self

    def recommend(self, product_id: int, n: int = 6, min_price: float = 0, max_price: float = 9999) -> pd.DataFrame:
        """
        Blends both recommendation types and returns the best combined results.
        """
        # Get recommendations from both engines (ask for more than needed to blend)
        content_recs = self.content_rec.recommend(product_id, n=20, min_price=min_price, max_price=max_price)
        collab_recs = self.collab_rec.recommend(product_id, n=20)

        if content_recs.empty and collab_recs.empty:
            return pd.DataFrame()

        # Build a score dictionary: product_id → hybrid score
        scores = {}

        if not content_recs.empty:
            for _, row in content_recs.iterrows():
                pid = row["product_id"]
                scores[pid] = scores.get(pid, 0) + self.alpha * row["similarity_score"]

        if not collab_recs.empty:
            for _, row in collab_recs.iterrows():
                pid = row["product_id"]
                # Apply price filter to collab results too
                if min_price <= row["price"] <= max_price:
                    scores[pid] = scores.get(pid, 0) + (1 - self.alpha) * row["similarity_score"]

        # Sort by hybrid score
        sorted_pids = sorted(scores, key=scores.get, reverse=True)[:n]

        results = []
        for pid in sorted_pids:
            product_row = self.products_df[self.products_df["product_id"] == pid]
            if not product_row.empty:
                product = product_row.iloc[0]
                reason = "Recommended for you"
                if not content_recs.empty and pid in content_recs["product_id"].values:
                    reason = "Similar product"
                if not collab_recs.empty and pid in collab_recs["product_id"].values:
                    reason = "Customers also bought"
                if (not content_recs.empty and pid in content_recs["product_id"].values and
                        not collab_recs.empty and pid in collab_recs["product_id"].values):
                    reason = "Top pick for you"

                results.append({
                    **product.to_dict(),
                    "hybrid_score": round(scores[pid], 3),
                    "recommendation_reason": reason,
                })

        return pd.DataFrame(results)

    def get_category_recommendations(self, category: str, n: int = 8) -> pd.DataFrame:
        """
        Returns top-rated products in a given category.
        Used for the homepage 'Trending in [category]' section.
        """
        cat_products = self.products_df[
            self.products_df["category"] == category
        ].copy()

        # Score = weighted combo of rating and popularity (log of reviews)
        cat_products["popularity_score"] = (
            cat_products["rating"] * 0.7 +
            np.log1p(cat_products["num_reviews"]) * 0.3
        )

        return cat_products.nlargest(n, "popularity_score")

    def search_products(self, query: str) -> pd.DataFrame:
        """
        Simple keyword search across product names, categories, and tags.
        """
        query = query.lower().strip()
        if not query:
            return self.products_df

        mask = (
            self.products_df["name"].str.lower().str.contains(query, na=False) |
            self.products_df["category"].str.lower().str.contains(query, na=False) |
            self.products_df["tags"].str.lower().str.contains(query, na=False)
        )
        return self.products_df[mask].head(20)

