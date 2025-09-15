# 🎯 Internship Recommender System  

An **AI-driven recommendation system** that intelligently matches students with the most suitable internships using **semantic embeddings (SBERT)**, **FAISS retrieval**, and **personalized scoring filters**. Unlike traditional keyword-based search platforms, this system understands **context, meaning, and user preferences**, making internship discovery more effective.  

---

## 📌 Features  

- 🔹 **Semantic Matching**: Uses **Sentence-BERT embeddings** to understand internship postings and user queries beyond simple keywords.  
- 🔹 **Efficient Retrieval**: Fast similarity search powered by **FAISS** or **Milvus** for large-scale datasets.  
- 🔹 **Personalized Filters**: Score adjustment based on **location, skills, stipend range, and recency**.  
- 🔹 **Future Ready**: Supports **collaborative filtering** to recommend internships based on user interactions (bookmarks, saves, clicks).  
- 🔹 **Interactive Frontend**: Built with **Streamlit** for easy exploration of recommendations.  
- 🔹 **Scalable Backend**: Modular API using **FastAPI** and **SQLAlchemy** for database interactions.  

---

## 🛠️ Tech Stack  

**Programming Language**  
- Python  

**Frameworks & Libraries**  
- Streamlit  
- FastAPI  
- SQLAlchemy  
- SentenceTransformers (SBERT)  
- Scikit-learn  

**Database**  
- SQLite (default)  
- PostgreSQL / MySQL (for scaling)  

**Vector Store / Caching**  
- FAISS (default)  
- Milvus (optional advanced)  

**AI / Machine Learning**  
- SBERT embeddings to convert internship data & user profiles into vectors  
- Collaborative Filtering (future enhancement)  

---

## ⚙️ Workflow  

1. **Data Preparation**  
   - Load internship JSON data  
   - Generate embeddings (SBERT)  
   - Store embeddings in FAISS index / cache  

2. **User Input**  
   - Candidate enters role, skills, location, stipend, duration  
   - Generate user embedding  

3. **Matching**  
   - Compute similarity between user embedding & internship embeddings  

4. **Score Adjustment**  
   - Apply boosters for **location, stipend range, skills, recency**  
   - Calculate final match score  

5. **Collaborative Filtering (Future)**  
   - Track user interactions (bookmarks, saves)  
   - Build User-Item matrix  
   - Recommend internships liked by similar users  

6. **Output**  
   - Display top recommendations (Title, Org, Location, Duration, Stipend, Apply By, Match Score)  
   - Log user interactions  

---

## 🚀 Installation  

```bash
# Clone repository
git clone https://github.com/SmitSutariya0205/internship-recommender.git
cd internship-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

