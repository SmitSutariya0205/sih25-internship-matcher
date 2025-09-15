import streamlit as st
import json
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import datetime

# ----------------------------
# Page Config (MUST be first Streamlit command)
# ----------------------------
st.set_page_config(page_title="Internship Recommender", page_icon="🎯", layout="wide")


# ----------------------------
# Load Internship Data
# ----------------------------
try:
    with open("internship.json", "r", encoding="utf-8") as f:
        internships = json.load(f)
except FileNotFoundError:
    st.error("Error: 'internship.json' file not found. Please make sure the file is in the same directory as the script.")
    st.stop()
except json.JSONDecodeError:
    st.error("Error: 'internship.json' is not a valid JSON file. Please check its format.")
    st.stop()


# ----------------------------
# Load SBERT Model
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

model = load_model()


# ----------------------------
# Create / Update Embeddings
# ----------------------------
def create_or_update_embeddings(internships, cache_file="embeddings_cache.json"):
    """
    Creates or updates a cache of internship embeddings.
    Only new internships are embedded to save time.
    """
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                st.warning("Corrupted embeddings cache, recreating...")
                cache = {}

    updated = False
    for internship in internships:
        iid = str(internship["internship_id"])
        if iid not in cache:
            # Create a comprehensive text for embedding
            text = f"{internship['title']} {internship['organization']} {internship.get('description', '')} {internship.get('skills', '')}"
            embedding = model.encode(text).tolist()
            cache[iid] = embedding
            updated = True

    if updated:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f)

    # Convert cached embeddings to a numpy matrix
    embeddings_matrix = np.array([cache[str(i["internship_id"])] for i in internships])
    return embeddings_matrix

internship_embeddings = create_or_update_embeddings(internships)


# ----------------------------
# Helper: Parse stipend to int
# ----------------------------
def parse_stipend(stipend_str):
    """Parses a stipend string and returns a single integer value."""
    try:
        if isinstance(stipend_str, (int, float)):
            return int(stipend_str)
        if stipend_str.lower() == "unpaid":
            return 0
        parts = stipend_str.replace(",", "").split()
        if "-" in parts[0]:  # handle ranges like "5000-8000"
            low, high = parts[0].split("-")
            return (int(low) + int(high)) // 2
        return int(parts[0])
    except (ValueError, IndexError):
        return 0


# ----------------------------
# Recommendation Function
# ----------------------------
def recommend_internships(title, location, duration, stipend_min, stipend_max, top_k=3):
    """
    Recommends internships based on a hybrid scoring system.
    Combines semantic similarity with boosts for filters.
    """
    candidate_embedding = model.encode(title).reshape(1, -1)
    similarities = cosine_similarity(candidate_embedding, internship_embeddings)[0]

    adjusted_scores = []
    for idx, sim in enumerate(similarities):
        internship = internships[idx]
        score = sim  # base similarity

        # ✅ Location boost
        if location != "Any" and location.lower() in internship["location"].lower():
            score += 0.15

        # ✅ Duration boost
        if duration != "Any" and duration.lower().replace("months", "").strip() in internship["duration"].lower().replace("months", "").strip():
            score += 0.05

        # ✅ Stipend boost
        stipend_value = parse_stipend(internship["stipend"])
        if stipend_min <= stipend_value <= stipend_max:
            score += 0.10

        # ✅ Recency boost for upcoming deadlines
        try:
            apply_by = datetime.datetime.strptime(internship["apply_by"], "%d-%b-%Y").date()
            days_left = (apply_by - datetime.date.today()).days
            if 0 < days_left <= 30:
                recency_factor = (30 - days_left) / 30.0
                score += 0.10 * recency_factor
        except (ValueError, TypeError):
            pass

        adjusted_scores.append(score)

    top_indices = np.argsort(adjusted_scores)[::-1][:top_k]
    top_results = [(internships[i], adjusted_scores[i]) for i in top_indices]

    best_score = top_results[0][1] if top_results else 0

    # -------------------------
    # ✅ Suggestion Rules
    # -------------------------
    if best_score < 0.30:  # <-- Changed from 0.20 to 0.30
        return "no_results", []
    elif best_score < 0.45:
        return "suggestion", top_results
    else:
        return "results", top_results


# ----------------------------
# Streamlit UI
# ----------------------------
st.markdown("""
<style>
    .header-container {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .internship-card {
        padding: 20px;
        margin: 15px 0;
        border-radius: 15px;
        border: 2px solid #4fc3f7;
        background: #0f141a;
        box-shadow: 0 6px 15px rgba(0,0,0,0.4);
        transition: transform 0.2s;
    }
    .internship-card:hover {
        transform: translateY(-5px);
    }
    .card-title {
        color: #4fc3f7;
        font-size: 24px;
        margin-bottom: 10px;
    }
    .stipend-paid { color: #66bb6a; font-weight: bold; }
    .stipend-unpaid { color: #bdbdbd; }
    .st-emotion-cache-1j02j0g { padding-top: 0rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-container">🎯 Internship Recommender System</div>', unsafe_allow_html=True)

st.write("Find the best internships tailored to your preferences 🚀")

# Dropdown Options
locations = sorted(list(set([i["location"] for i in internships])))
durations = sorted(list(set([i["duration"] for i in internships])))

stipend_ranges = [
    "Any",
    "Unpaid",
    "0 – 5000",
    "5001 – 10000",
    "10001 – 20000",
    "20001+"
]

# Candidate Input
st.subheader("📝 Your Preferences")
col1, col2, col3 = st.columns(3)

with col1:
    title = st.text_input("Preferred Role/Title", "")

with col2:
    location = st.selectbox("Preferred Location", ["Any"] + locations)

with col3:
    duration = st.selectbox("Preferred Duration", ["Any"] + durations)

stipend_choice = st.selectbox("Preferred Stipend Range", stipend_ranges)

# Convert stipend_choice to numeric range
stipend_min, stipend_max = 0, 999999
if stipend_choice == "Unpaid":
    stipend_min, stipend_max = 0, 0
elif stipend_choice == "0 – 5000":
    stipend_min, stipend_max = 1, 5000
elif stipend_choice == "5001 – 10000":
    stipend_min, stipend_max = 5001, 10000
elif stipend_choice == "10001 – 20000":
    stipend_min, stipend_max = 10001, 20000
elif stipend_choice == "20001+":
    stipend_min, stipend_max = 20001, 999999


# Button Action
if st.button("🔍 Find Internships"):
    if not title.strip():
        st.warning("Please enter a preferred role or title to get started.")
    else:
        status, results = recommend_internships(
            title, location, duration, stipend_min, stipend_max, top_k=3
        )
        
        st.subheader("💡 Your Top Matches")

        if status == "no_results":
            st.error("🚫 No internships found matching your query. Try a different title or broader filters.")
        
        elif status == "suggestion":
            st.warning(f"🤔 We found some potential matches, but they might not be perfect. Here are the closest results:")
            for internship, score in results:
                stipend_css_class = "stipend-paid" if parse_stipend(internship['stipend']) > 0 else "stipend-unpaid"
                st.markdown(f"""
<div class="internship-card">
    <h3 class="card-title">{internship['title']}</h3>
    <p>🏢 <b>Organization:</b> {internship['organization']}</p>
    <p>📍 <b>Location:</b> {internship['location']}</p>
    <p>⏳ <b>Duration:</b> {internship['duration']}</p>
    <p><span class="{stipend_css_class}">💰 <b>Stipend:</b> {internship['stipend']}</span></p>
    <p>🗓 <b>Apply By:</b> {internship['apply_by']}</p>
    <p>⭐ <b>Match Score:</b> {score:.3f}</p>
</div>
""", unsafe_allow_html=True)
        
        else:  # normal results
            for internship, score in results:
                stipend_css_class = "stipend-paid" if parse_stipend(internship['stipend']) > 0 else "stipend-unpaid"
                st.markdown(f"""
<div class="internship-card">
    <h3 class="card-title">{internship['title']}</h3>
    <p>🏢 <b>Organization:</b> {internship['organization']}</p>
    <p>📍 <b>Location:</b> {internship['location']}</p>
    <p>⏳ <b>Duration:</b> {internship['duration']}</p>
    <p><span class="{stipend_css_class}">💰 <b>Stipend:</b> {internship['stipend']}</span></p>
    <p>🗓 <b>Apply By:</b> {internship['apply_by']}</p>
    <p>⭐ <b>Match Score:</b> {score:.3f}</p>
</div>
""", unsafe_allow_html=True)