 ShopSmart AI — Product Recommendation Engine

An AI-powered product recommendation system that helps online shoppers discover relevant products instantly — built with Python, scikit-learn, and Streamlit.

🔴 Live Demo →

The Problem
Every day, millions of online shoppers land on e-commerce platforms and leave without buying anything — not because the product doesn't exist, but because they couldn't find it.
The average e-commerce store carries thousands of products. Without intelligent recommendations:

Shoppers waste time scrolling through irrelevant results
Stores lose sales from products that never get discovered
The shopping experience feels frustrating and impersonal

This is the problem Amazon, Jumia, and every major retailer spend millions solving. ShopSmart AI brings that same technology to life from scratch.

The Solution
ShopSmart AI is a hybrid recommendation engine that suggests the right products to the right user at the right time — using two complementary machine learning algorithms working together.
When a shopper views a product, the system instantly recommends:

Similar products they might prefer
Products other customers commonly bought together
Cross-category discoveries they wouldn't have found on their own


How It Works
Algorithm 1 — Content-Based Filtering
Converts each product's name, category, and tags into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency). Then measures how similar any two products are using cosine similarity.
"Wireless Headphones" → [0.8, 0.0, 0.6, 0.3, ...]
"Bluetooth Earbuds"   → [0.7, 0.0, 0.5, 0.4, ...]
Similarity score = 0.91  → highly similar ✅
Algorithm 2 — Collaborative Filtering
Builds a 200 × 120 user-item rating matrix and finds products that are frequently rated together by the same users — even across different categories.
Users who rated "Air Fryer" highly also rated "Instant Pot" highly
→ Recommend Instant Pot to Air Fryer viewers
Algorithm 3 — Hybrid Recommender
Blends both scores using a tunable alpha parameter:
hybrid_score = (alpha × content_score) + ((1 - alpha) × collaborative_score)
Users can adjust alpha in real time to shift between content-based and collaborative results.

Features

AI Recommendations — real-time product suggestions powered by ML
Smart Search — keyword search across 120 products and 6 categories
Price Filtering — filter recommendations by budget
Analytics Dashboard — interactive charts showing product and rating data
Algorithm Explainer — visual breakdown of how the ML works
Adjustable Model — tune the content vs collaborative weight with a live slider


Tech Stack
TechnologyPurposePython 3.13Core programming languagescikit-learnTF-IDF vectorizer + cosine similaritypandasData manipulation and analysisnumpyNumerical operationsStreamlitWeb application frameworkPlotlyInteractive data visualizations

Project Structure
shopsmart-ai/
├── app.py            # Streamlit web app — all UI and pages
├── recommender.py    # ML engine — content, collaborative, hybrid
├── data.py           # Product catalog + user rating generator
├── requirements.txt  # Python dependencies
└── README.md         # This file

Run Locally
bash# Clone the repository
git clone https://github.com/Icaalex/shopsmart-ai.git
cd shopsmart-ai

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
Open your browser at http://localhost:8501

What I Learned

How TF-IDF converts text into numerical feature vectors
How cosine similarity measures distance between vectors in high-dimensional space
How collaborative filtering works using a user-item rating matrix
How to combine multiple ML models into a hybrid ensemble system
How to build and deploy a data-driven web application end-to-end


Roadmap

 Integrate a real Amazon/Jumia product dataset from Kaggle
 Add user login so ratings persist across sessions
 Implement matrix factorization (SVD) for improved collaborative filtering
 Add A/B testing framework to compare algorithm performance
 Connect a PostgreSQL database to replace synthetic data


Author
UJU NDUKWU

GitHub: @Icaalex
Built as part of a self-taught ML engineering portfolio


Built with Python and scikit-learn/ Deployed on Streamlit Cloud
