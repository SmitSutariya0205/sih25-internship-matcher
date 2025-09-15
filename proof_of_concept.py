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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    :root {
        --bg-1: #0b1220;
        --bg-2: #0e1526;
        --card: rgba(255, 255, 255, 0.05);
        --border: rgba(255, 255, 255, 0.08);
        --text: #e6edf3;
        --muted: #9da9bb;
        --primary: #59c3ff;
        --primary-2: #4aa3e0;
        --success: #66bb6a;
        --warning: #ffd166;
        --danger: #ef476f;
        --shadow: 0 10px 30px rgba(0,0,0,0.35);
    }

    .stApp {
        background: radial-gradient(1200px 800px at 20% -10%, #1b2b4b 0%, rgba(27,43,75,0) 60%),
                    radial-gradient(1200px 800px at 120% 10%, #0d1b2a 0%, rgba(13,27,42,0) 55%),
                    linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
        color: var(--text);
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, 'Apple Color Emoji', 'Segoe UI Emoji';
    }

    /* Header */
    .hero {
        background: linear-gradient(180deg, rgba(89,195,255,0.15) 0%, rgba(89,195,255,0.04) 100%);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        padding: 28px 28px 22px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 18px;
    }
    .hero h1 { margin: 0 0 6px; font-size: 28px; letter-spacing: 0.2px; }
    .hero p  { margin: 0; color: var(--muted); font-size: 14.5px; }

    /* Controls */
    .panel {
        border: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%);
        border-radius: 14px;
        padding: 14px 16px 4px;
        box-shadow: var(--shadow);
        margin-bottom: 12px;
    }

    /* Primary button */
    .stButton>button {
        background: linear-gradient(180deg, var(--primary) 0%, var(--primary-2) 100%) !important;
        color: #0b1220 !important;
        border: 0 !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px !important;
        box-shadow: 0 8px 20px rgba(73, 167, 224, 0.35) !important;
        transition: transform 0.05s ease, box-shadow 0.2s ease !important;
    }
    .stButton>button:hover { transform: translateY(-1px) !important; }
    .stButton>button:active { transform: translateY(0px) !important; }

    /* Cards */
    .internship-card {
        position: relative;
        padding: 18px 18px 14px;
        margin: 10px 0 12px;
        border-radius: 16px;
        border: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.03) 100%);
        box-shadow: var(--shadow);
        transition: transform .12s ease, border-color .2s ease;
    }
    .internship-card:hover { transform: translateY(-2px); border-color: rgba(89,195,255,0.35); }
    .card-title { color: var(--primary); font-size: 20px; margin: 0 0 6px; }
    .meta { color: var(--muted); font-size: 13px; margin: 0 0 4px; }
    .pill {
        display: inline-block; font-size: 12px; padding: 4px 8px; margin: 6px 6px 0 0;
        border-radius: 999px; border: 1px solid var(--border); background: rgba(255,255,255,0.04);
    }
    .pill.ok { border-color: rgba(102,187,106,0.4); background: rgba(102,187,106,0.08); color: var(--success); }
    .pill.dim { color: var(--muted); }

    /* Score badge */
    .score {
        position: absolute; right: 14px; top: 14px;
        background: rgba(89,195,255,0.15); border: 1px solid rgba(89,195,255,0.35);
        color: var(--primary); padding: 4px 8px; font-weight: 700; border-radius: 10px; font-size: 12px;
    }

    /* Misc */
    .stipend-paid { color: var(--success); font-weight: 700; }
    .stipend-unpaid { color: var(--muted); font-weight: 600; }
    .spacer { height: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🎯 Internship Recommender</h1><p>Find tailored internships with smart matching</p></div>', unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.write("📝 Your preferences")

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


# Close styled preferences panel
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

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
    <span class="score">⭐ {score:.3f}</span>
    <h3 class="card-title">{internship['title']}</h3>
    <p class="meta">🏢 <b>{internship['organization']}</b></p>
    <p class="meta">📍 {internship['location']} &nbsp;•&nbsp; ⏳ {internship['duration']} &nbsp;•&nbsp; <span class="{stipend_css_class}">💰 {internship['stipend']}</span> &nbsp;•&nbsp; 🗓 {internship['apply_by']}</p>
    <div>
        <span class="pill ok">Suggested match</span>
        <span class="pill dim">Refine your filters</span>
    </div>
</div>
""", unsafe_allow_html=True)
        
        else:  # normal results
            for internship, score in results:
                stipend_css_class = "stipend-paid" if parse_stipend(internship['stipend']) > 0 else "stipend-unpaid"
                st.markdown(f"""
<div class="internship-card">
    <span class="score">⭐ {score:.3f}</span>
    <h3 class="card-title">{internship['title']}</h3>
    <p class="meta">🏢 <b>{internship['organization']}</b></p>
    <p class="meta">📍 {internship['location']} &nbsp;•&nbsp; ⏳ {internship['duration']} &nbsp;•&nbsp; <span class="{stipend_css_class}">💰 {internship['stipend']}</span> &nbsp;•&nbsp; 🗓 {internship['apply_by']}</p>
    <div>
        <span class="pill ok">Top match</span>
        <span class="pill">Relevant role</span>
    </div>
</div>
""", unsafe_allow_html=True)