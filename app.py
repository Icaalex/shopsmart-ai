"""
app.py — ShopSmart AI: Product Recommendation Engine
Run with: streamlit run app.py

This is the main web app. Streamlit turns Python scripts into
interactive web apps without any HTML or JavaScript needed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import our own modules
from data import generate_products, generate_user_ratings
from recommender import HybridRecommender

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopSmart AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .product-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #e5e7eb;
        height: 100%;
        transition: box-shadow 0.2s;
    }
    .product-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.3rem;
        line-height: 1.4;
    }
    .product-price {
        font-size: 1.2rem;
        font-weight: 700;
        color: #059669;
    }
    .product-category {
        font-size: 0.75rem;
        background: #eff6ff;
        color: #1d4ed8;
        padding: 2px 8px;
        border-radius: 12px;
        display: inline-block;
        margin-bottom: 0.4rem;
    }
    .rec-reason {
        font-size: 0.75rem;
        color: #7c3aed;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    .star-rating {
        color: #f59e0b;
        font-size: 0.85rem;
    }
    .stat-card {
        background: #f9fafb;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #6b7280;
    }
    .tag-chip {
        background: #f3f4f6;
        color: #374151;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.72rem;
        margin: 2px;
        display: inline-block;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
</style>
""", unsafe_allow_html=True)


# ── DATA & MODEL LOADING (cached so it only runs once) ────────────────────────
@st.cache_resource
def load_model():
    """
    This function runs once and caches the result.
    The @st.cache_resource decorator is KEY for performance —
    without it, the model would retrain on every click.
    """
    products = generate_products()
    ratings = generate_user_ratings(products)
    model = HybridRecommender(alpha=0.5)
    model.fit(products, ratings)
    return model, products, ratings


# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def render_stars(rating: float) -> str:
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


def render_product_card(product: dict, show_score: bool = False, score_label: str = "Match"):
    """Renders a single product card using HTML."""
    in_stock_badge = (
        '<span style="color:#059669;font-size:0.75rem;">✓ In Stock</span>'
        if product.get("in_stock", True)
        else '<span style="color:#dc2626;font-size:0.75rem;">✗ Out of Stock</span>'
    )
    tags = product.get("tag_list", [])
    if isinstance(tags, str):
        tags = tags.split(",")
    tag_chips = "".join([f'<span class="tag-chip">{t.strip()}</span>' for t in tags[:4]])

    score_html = ""
    if show_score:
        score_val = product.get("hybrid_score", product.get("similarity_score", 0))
        pct = int(score_val * 100)
        score_html = f"""
        <div style="margin-top:0.6rem;">
            <div style="font-size:0.72rem;color:#6b7280;margin-bottom:3px;">{score_label}: {pct}%</div>
            <div style="background:#e5e7eb;border-radius:4px;height:6px;overflow:hidden;">
                <div style="background:#7c3aed;width:{pct}%;height:100%;border-radius:4px;"></div>
            </div>
        </div>
        """

    reason_html = ""
    if product.get("recommendation_reason"):
        reason_html = f'<div class="rec-reason">✦ {product["recommendation_reason"]}</div>'

    st.markdown(f"""
    <div class="product-card">
        <div class="product-category">{product["category"]}</div>
        <div class="product-name">{product["name"]}</div>
        <div class="star-rating">{render_stars(product["rating"])} 
            <span style="color:#6b7280;font-size:0.78rem;">({product["num_reviews"]:,})</span>
        </div>
        <div style="margin-top:0.5rem;">
            <span class="product-price">${product["price"]:.2f}</span>
            &nbsp; {in_stock_badge}
        </div>
        <div style="margin-top:0.6rem;">{tag_chips}</div>
        {score_html}
        {reason_html}
    </div>
    """, unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
def render_sidebar(products_df, model):
    with st.sidebar:
        st.markdown("## 🛍️ ShopSmart AI")
        st.markdown("*Powered by ML recommendations*")
        st.divider()

        page = st.radio(
            "Navigate",
            ["🏠 Home", "🔍 Search & Recommend", "📊 Analytics Dashboard", "🤖 How it Works"],
            label_visibility="collapsed"
        )

        st.divider()
        st.markdown("**Filters**")
        price_range = st.slider(
            "Price range ($)",
            min_value=0,
            max_value=400,
            value=(0, 400),
            step=5,
        )
        categories = ["All"] + sorted(products_df["category"].unique().tolist())
        selected_category = st.selectbox("Category", categories)

        st.divider()
        st.markdown("**Model Settings**")
        alpha = st.slider(
            "Content vs Collaborative weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0 = pure collaborative, 1 = pure content-based"
        )
        model.alpha = alpha

        n_recs = st.slider("Number of recommendations", 4, 12, 6, 2)

    return page, price_range, selected_category, n_recs


# ── PAGES ────────────────────────────────────────────────────────────────────
def page_home(products_df, model):
    st.markdown('<h1 class="main-header">🛍️ ShopSmart AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered product recommendations • 120 products • 6 categories</p>', unsafe_allow_html=True)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Products", "120")
    with col2:
        st.metric("Categories", "6")
    with col3:
        st.metric("Simulated Users", "200")
    with col4:
        avg_rating = round(products_df["rating"].mean(), 1)
        st.metric("Avg Rating", f"⭐ {avg_rating}")

    st.divider()

    # Trending per category
    categories = products_df["category"].unique()
    selected_cat = st.selectbox(
        "🔥 Trending in...",
        categories,
        label_visibility="visible"
    )

    trending = model.get_category_recommendations(selected_cat, n=6)

    cols = st.columns(3)
    for i, (_, product) in enumerate(trending.iterrows()):
        with cols[i % 3]:
            render_product_card(product.to_dict())
            st.write("")


def page_search(products_df, model, price_range, n_recs):
    st.markdown('<h2 class="section-title">🔍 Search & Get Recommendations</h2>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Search Products", "Browse All"])

    with tab1:
        query = st.text_input("Search products...", placeholder="e.g. wireless, yoga, coffee, python...")

        if query:
            results = model.search_products(query)
            if results.empty:
                st.warning("No products found. Try a different search term.")
            else:
                st.success(f"Found {len(results)} products")
                cols = st.columns(3)
                for i, (_, product) in enumerate(results.iterrows()):
                    with cols[i % 3]:
                        render_product_card(product.to_dict())
                        st.write("")

    with tab2:
        # Filter by price
        filtered = products_df[
            (products_df["price"] >= price_range[0]) &
            (products_df["price"] <= price_range[1])
        ]
        st.caption(f"Showing {len(filtered)} products | Price: ${price_range[0]} – ${price_range[1]}")

        sort_by = st.selectbox("Sort by", ["Rating", "Price (low to high)", "Price (high to low)", "Most reviewed"])
        if sort_by == "Rating":
            filtered = filtered.sort_values("rating", ascending=False)
        elif sort_by == "Price (low to high)":
            filtered = filtered.sort_values("price")
        elif sort_by == "Price (high to low)":
            filtered = filtered.sort_values("price", ascending=False)
        elif sort_by == "Most reviewed":
            filtered = filtered.sort_values("num_reviews", ascending=False)

        cols = st.columns(3)
        for i, (_, product) in enumerate(filtered.head(18).iterrows()):
            with cols[i % 3]:
                render_product_card(product.to_dict())
                st.write("")

    st.divider()

    # ── RECOMMENDATION SECTION ──────────────────────────────────────────────
    st.markdown('<h2 class="section-title">🤖 AI Recommendations</h2>', unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        product_names = products_df["name"].tolist()
        selected_product_name = st.selectbox(
            "Pick a product to get recommendations for:",
            product_names,
        )
    with col_b:
        rec_type = st.selectbox(
            "Algorithm",
            ["Hybrid (Best)", "Content-based only", "Collaborative only"]
        )

    selected_product = products_df[products_df["name"] == selected_product_name].iloc[0]

    # Show selected product
    st.markdown("**Selected product:**")
    col1, col2 = st.columns([1, 3])
    with col1:
        render_product_card(selected_product.to_dict())
    with col2:
        st.markdown(f"**Category:** {selected_product['category']}")
        st.markdown(f"**Price:** ${selected_product['price']:.2f}")
        st.markdown(f"**Rating:** {render_stars(selected_product['rating'])} ({selected_product['num_reviews']:,} reviews)")
        tags = selected_product['tag_list']
        if isinstance(tags, str):
            tags = tags.split(",")
        st.markdown("**Tags:** " + " ".join([f"`{t}`" for t in tags]))

    st.markdown(f"**Showing {n_recs} recommendations using {rec_type}:**")

    pid = selected_product["product_id"]

    if rec_type == "Hybrid (Best)":
        recs = model.recommend(pid, n=n_recs, min_price=price_range[0], max_price=price_range[1])
        score_label = "Hybrid match"
    elif rec_type == "Content-based only":
        recs = model.content_rec.recommend(pid, n=n_recs, min_price=price_range[0], max_price=price_range[1])
        score_label = "Content match"
    else:
        recs = model.collab_rec.recommend(pid, n=n_recs)
        score_label = "Collab match"

    if recs.empty:
        st.warning("No recommendations found with current filters. Try adjusting the price range.")
    else:
        cols = st.columns(3)
        for i, (_, rec) in enumerate(recs.iterrows()):
            with cols[i % 3]:
                render_product_card(rec.to_dict(), show_score=True, score_label=score_label)
                st.write("")


def page_analytics(products_df, ratings_df):
    st.markdown('<h2 class="section-title">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Products by category
        cat_counts = products_df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.bar(
            cat_counts, x="Category", y="Count",
            title="Products by category",
            color="Count",
            color_continuous_scale="blues",
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Average rating by category
        avg_ratings = products_df.groupby("category")["rating"].mean().reset_index()
        avg_ratings.columns = ["Category", "Avg Rating"]
        avg_ratings = avg_ratings.sort_values("Avg Rating", ascending=True)
        fig2 = px.bar(
            avg_ratings, x="Avg Rating", y="Category",
            orientation="h",
            title="Average rating by category",
            color="Avg Rating",
            color_continuous_scale="greens",
        )
        fig2.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Price distribution
        fig3 = px.histogram(
            products_df, x="price",
            nbins=20,
            title="Price distribution",
            color_discrete_sequence=["#7c3aed"],
        )
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Rating distribution
        fig4 = px.histogram(
            products_df, x="rating",
            nbins=15,
            title="Rating distribution",
            color_discrete_sequence=["#059669"],
        )
        fig4.update_layout(height=350)
        st.plotly_chart(fig4, use_container_width=True)

    # Rating activity heatmap by category
    st.markdown("#### User Rating Activity by Category")
    category_ratings = ratings_df.merge(
        products_df[["product_id", "category"]], on="product_id"
    )
    heatmap_data = category_ratings.groupby(["category", "rating"]).size().unstack(fill_value=0)
    fig5 = px.imshow(
        heatmap_data,
        title="Rating distribution heatmap (categories vs star ratings)",
        color_continuous_scale="purples",
        labels={"x": "Star Rating", "y": "Category", "color": "Count"},
    )
    fig5.update_layout(height=350)
    st.plotly_chart(fig5, use_container_width=True)

    # Top 10 products by reviews
    st.markdown("#### Top 10 Most Reviewed Products")
    top_products = products_df.nlargest(10, "num_reviews")[["name", "category", "rating", "num_reviews", "price"]]
    top_products.columns = ["Product", "Category", "Rating", "Reviews", "Price ($)"]
    st.dataframe(top_products, use_container_width=True, hide_index=True)


def page_how_it_works():
    st.markdown('<h2 class="section-title">🤖 How the AI Works</h2>', unsafe_allow_html=True)

    st.markdown("""
    ShopSmart AI uses **two recommendation algorithms** combined into a hybrid system.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 1. Content-Based Filtering")
        st.markdown("""
        **The idea:** Recommend products that are *similar to this product*.

        **How it works:**
        1. Each product's name, category, and tags are converted into numbers using **TF-IDF**
        2. TF-IDF asks: "which words matter most for each product?"
        3. Then we measure how close two products are in that numeric space using **cosine similarity**
        4. Score of 1.0 = identical products. Score of 0 = totally different.

        **Example:** You look at "Yoga Mat Premium" →
        - High similarity: Resistance Bands, Foam Roller, Yoga Pants
        - Low similarity: USB Hub, Leather Wallet, Coffee Maker

        **Best for:** Finding similar items in the same category
        """)

    with col2:
        st.markdown("### 2. Collaborative Filtering")
        st.markdown("""
        **The idea:** "Users who liked this also liked..."

        **How it works:**
        1. Build a matrix: 200 users × 120 products, filled with their ratings (0 = not rated)
        2. Find products that tend to be rated together by the same users
        3. Use **cosine similarity** again — but now comparing products as "rating vectors"

        **Example:** 50 users rated both "Python Crash Course" and "Clean Code" highly →
        those products become similar in the collaborative sense

        **Best for:** Cross-category discovery (bought running shoes → also needs protein powder)
        """)

    st.divider()

    st.markdown("### 3. Hybrid System")
    st.markdown("""
    The hybrid recommender combines both scores:

    ```
    hybrid_score = (alpha × content_score) + ((1 - alpha) × collaborative_score)
    ```

    - **alpha = 1.0** → pure content-based (safe, stays in category)
    - **alpha = 0.0** → pure collaborative (adventurous, cross-category)
    - **alpha = 0.5** → balanced (our default — best of both)

    You can adjust the alpha slider in the sidebar to see how it changes results!
    """)

    st.divider()

    st.markdown("### Tech Stack")
    tech_cols = st.columns(4)
    techs = [
        ("Python", "The programming language"),
        ("pandas", "Data manipulation"),
        ("scikit-learn", "TF-IDF + cosine similarity"),
        ("Streamlit", "This web app UI"),
    ]
    for col, (name, desc) in zip(tech_cols, techs):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size:1rem;font-weight:600;color:#111827">{name}</div>
                <div style="font-size:0.78rem;color:#6b7280;margin-top:4px">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ── MAIN APP ─────────────────────────────────────────────────────────────────
def main():
    # Load model (cached — only runs once)
    model, products_df, ratings_df = load_model()

    # Render sidebar and get user selections
    page, price_range, selected_category, n_recs = render_sidebar(products_df, model)

    # Apply category filter if selected
    if selected_category != "All":
        filtered_products = products_df[products_df["category"] == selected_category]
    else:
        filtered_products = products_df

    # Render the selected page
    if page == "🏠 Home":
        page_home(products_df, model)
    elif page == "🔍 Search & Recommend":
        page_search(products_df, model, price_range, n_recs)
    elif page == "📊 Analytics Dashboard":
        page_analytics(products_df, ratings_df)
    elif page == "🤖 How it Works":
        page_how_it_works()

    # Footer
    st.divider()
    st.markdown(
        '<div style="text-align:center;color:#9ca3af;font-size:0.8rem;">'
        'ShopSmart AI — Built with Python, scikit-learn & Streamlit'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

