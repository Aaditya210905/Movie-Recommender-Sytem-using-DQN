"""
Streamlit Interface for DQN Movie Recommendation System
========================================================
A web UI for the Deep Q-Network movie recommender trained on 7,841 IMDB movies.
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DQN Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH = os.path.join(BASE_DIR, "imdb_movies_2025_cleaned.csv")


# ── Load Resources (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading DQN agent & models...")
def load_all():
    """Load agent, processor, catalog and training history once."""
    from core import DQNAgent, MovieFeatureProcessor, MovieRecommender, HybridPersonalizedRecommender

    agent = DQNAgent.load(os.path.join(MODELS_DIR, "dqn_recommender"))
    processor = MovieFeatureProcessor.load(os.path.join(MODELS_DIR, "feature_processor.pkl"))

    with open(os.path.join(MODELS_DIR, "movie_catalog.pkl"), "rb") as f:
        movie_catalog = pickle.load(f)

    training_history = None
    th_path = os.path.join(MODELS_DIR, "training_history.pkl")
    if os.path.exists(th_path):
        with open(th_path, "rb") as f:
            training_history = pickle.load(f)

    recommender = MovieRecommender(agent, movie_catalog, processor)
    hybrid = HybridPersonalizedRecommender(agent, movie_catalog, processor)

    return agent, processor, movie_catalog, training_history, recommender, hybrid


@st.cache_data(show_spinner="Loading dataset...")
def load_dataframe():
    return pd.read_csv(CSV_PATH)


# ── Initialise ───────────────────────────────────────────────────────────────
agent, processor, movie_catalog, training_history, recommender, hybrid = load_all()
df = load_dataframe()

FEEDBACK_REWARDS = {
    'like': 1.5,
    'watch': 0.8,
    'click': 0.4,
    'skip': -0.2,
    'ignore': -0.5,
}

# Gather all genres
all_genres_set = set()
for g in movie_catalog['genres']:
    for genre in g.split(','):
        all_genres_set.add(genre.strip())
ALL_GENRES = sorted(all_genres_set)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/color/96/movie-projector.png", width=80)
    st.title("DQN Movie Recommender")
    st.caption("Powered by Deep Reinforcement Learning")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "🔬 Pipeline Visualizer",
            "🎯 Get Recommendations",
            "🎭 Genre Explorer",
            "⚡ Hybrid Recommender",
            "💬 Interactive Session",
            "📊 Training Dashboard",
            "🧠 Model Architecture",
            "📈 Dataset Analytics",
        ],
    )
    st.divider()
    st.markdown(
        "**Model Stats**\n"
        f"- Movies: **{len(movie_catalog['names']):,}**\n"
        f"- Feature dim: **{processor.feature_dim}**\n"
        f"- Genres: **{processor.genre_dim}**\n"
        f"- Embedding: **{processor.embedding_dim}D**\n"
        f"- Epsilon: **{agent.epsilon:.4f}**"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

# ─── HOME ────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🎬 DQN Movie Recommendation System")
    st.markdown(
        """
        Welcome! This app showcases a **Deep Q-Network (DQN)** based movie recommendation engine
        trained on **7,841 IMDB movies** from 2025.

        ### How it works
        | Component | Details |
        |---|---|
        | **State** | 310-D user preference vector (209 genres + 100 TF-IDF/SVD embeddings + 1 rating) |
        | **Action** | Select one of 7,841 movies |
        | **Reward** | User feedback: like (+1.5) / watch (+0.8) / click (+0.4) / skip (−0.2) / ignore (−0.5) |
        | **Architecture** | Dueling DQN with LeakyReLU, BatchNorm, Dropout |
        | **Training** | ε-greedy, experience replay (50 K), target network sync every 50 steps |

        ### Pages
        - **Get Recommendations** — cold-start top-K from the DQN
        - **Genre Explorer** — genre-biased recommendations
        - **Hybrid Recommender** — blend Q-values with content similarity
        - **Interactive Session** — give feedback & watch the model adapt
        - **Training Dashboard** — loss curves, reward history
        - **Model Architecture** — layer-by-layer network details
        - **Dataset Analytics** — explore the IMDB movie dataset
        """
    )

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", f"{len(movie_catalog['names']):,}")
    col2.metric("Unique Genres", f"{len(ALL_GENRES)}")
    col3.metric("Avg Rating", f"{df['rating'].mean():.2f}")
    col4.metric("Network Params", f"{agent.q_network.count_params():,}")


# ─── GET RECOMMENDATIONS ────────────────────────────────────────────────────
elif page == "🎯 Get Recommendations":
    st.title("🎯 Cold-Start Recommendations")
    st.markdown("Generate top-K recommendations for a **new (random) user**.")

    col_a, col_b = st.columns(2)
    with col_a:
        k = st.slider("Number of recommendations", 5, 50, 10)
    with col_b:
        seed_val = st.number_input("Random seed (for reproducibility)", value=42, step=1)

    if st.button("Generate Recommendations", type="primary"):
        np.random.seed(int(seed_val))
        recs = recommender.get_top_k_recommendations(k=k)

        for rec in recs:
            with st.container():
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.markdown(f"### #{rec['rank']}")
                    st.metric("Q-value", f"{rec['q_value']:.4f}")
                with c2:
                    st.subheader(rec['movie_name'])
                    st.caption(f"⭐ {rec['rating']:.1f}/10  •  🎭 {rec['genre']}")
                    st.write(rec['description'][:300] + ("..." if len(rec['description']) > 300 else ""))
                st.divider()

        # Q-value bar chart
        names = [r['movie_name'][:40] for r in recs]
        qvals = [r['q_value'] for r in recs]
        fig = px.bar(
            x=qvals, y=names, orientation='h',
            labels={"x": "Q-Value", "y": "Movie"},
            title="Q-Value Distribution for Recommendations",
            color=qvals, color_continuous_scale="Viridis",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=max(400, k * 35))
        st.plotly_chart(fig, use_container_width=True)


# ─── GENRE EXPLORER ─────────────────────────────────────────────────────────
elif page == "🎭 Genre Explorer":
    st.title("🎭 Genre-Based Recommendations")
    st.markdown("Select your favourite genres and get personalised picks.")

    selected_genres = st.multiselect("Choose genres", ALL_GENRES, default=["Action", "Sci-Fi"])
    k = st.slider("How many?", 5, 30, 10, key="genre_k")

    if selected_genres and st.button("Find Movies", type="primary"):
        recs = recommender.recommend_by_genre_preference(selected_genres, k=k)

        st.success(f"Top {k} movies for **{', '.join(selected_genres)}**")
        rec_df = pd.DataFrame(recs)[['rank', 'movie_name', 'genre', 'rating', 'q_value']]
        rec_df.columns = ['#', 'Movie', 'Genres', 'Rating', 'Q-Value']
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

        # Genre overlap chart
        genre_hits = {}
        for r in recs:
            for g in r['genre'].split(','):
                g = g.strip()
                genre_hits[g] = genre_hits.get(g, 0) + 1
        fig = px.pie(
            names=list(genre_hits.keys()),
            values=list(genre_hits.values()),
            title="Genre Distribution in Recommendations",
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── HYBRID RECOMMENDER ─────────────────────────────────────────────────────
elif page == "⚡ Hybrid Recommender":
    st.title("⚡ Hybrid Personalised Recommender")
    st.markdown(
        "Blends **DQN Q-values** with **content-based cosine similarity**.  \n"
        "α = 0 → pure DQN · α = 1 → pure content similarity"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        alpha = st.slider("Alpha (similarity weight)", 0.0, 1.0, 0.5, 0.05)
    with col2:
        k = st.slider("Recommendations", 5, 30, 10, key="hybrid_k")
    with col3:
        seed_val = st.number_input("Seed", value=42, step=1, key="hybrid_seed")

    if st.button("Generate Hybrid Recommendations", type="primary"):
        np.random.seed(int(seed_val))
        user_state = np.random.randn(processor.feature_dim).astype(np.float32) * 0.3
        user_state = user_state / (np.linalg.norm(user_state) + 1e-8)

        recs = hybrid.personalized_recommend(user_state, k=k, alpha=alpha)

        rec_df = pd.DataFrame(recs)
        display_df = rec_df[['rank', 'movie_name', 'genre', 'rating', 'q_value', 'similarity', 'hybrid_score']]
        display_df.columns = ['#', 'Movie', 'Genres', 'Rating', 'Q-Value', 'Similarity', 'Hybrid Score']
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Score comparison chart
        fig = go.Figure()
        names = [r['movie_name'][:30] for r in recs]
        fig.add_trace(go.Bar(name='Q-Value (norm)', x=names,
                             y=[(r['q_value'] - min(r2['q_value'] for r2 in recs)) /
                                (max(r2['q_value'] for r2 in recs) - min(r2['q_value'] for r2 in recs) + 1e-8)
                                for r in recs]))
        fig.add_trace(go.Bar(name='Similarity (norm)', x=names,
                             y=[(r['similarity'] - min(r2['similarity'] for r2 in recs)) /
                                (max(r2['similarity'] for r2 in recs) - min(r2['similarity'] for r2 in recs) + 1e-8)
                                for r in recs]))
        fig.add_trace(go.Scatter(name='Hybrid Score', x=names,
                                 y=[r['hybrid_score'] for r in recs], mode='lines+markers'))
        fig.update_layout(barmode='group', title="Score Breakdown", xaxis_tickangle=-45,
                          height=500)
        st.plotly_chart(fig, use_container_width=True)


# ─── INTERACTIVE SESSION ────────────────────────────────────────────────────
elif page == "💬 Interactive Session":
    st.title("💬 Interactive Recommendation Session")
    st.markdown(
        "Give feedback on each recommendation and watch the model **adapt** its user-state in real time."
    )

    # Session state
    if "session_active" not in st.session_state:
        st.session_state.session_active = False
        st.session_state.user_state = None
        st.session_state.session_history = []
        st.session_state.excluded = []
        st.session_state.cumulative_reward = 0.0

    col_start, col_reset = st.columns(2)
    with col_start:
        if st.button("▶️  Start New Session", type="primary"):
            st.session_state.session_active = True
            st.session_state.user_state = np.random.randn(processor.feature_dim).astype(np.float32) * 0.1
            st.session_state.session_history = []
            st.session_state.excluded = []
            st.session_state.cumulative_reward = 0.0
            st.rerun()
    with col_reset:
        if st.button("🔄  Reset"):
            st.session_state.session_active = False
            st.rerun()

    if st.session_state.session_active:
        st.info(f"Step {len(st.session_state.session_history) + 1}  •  Cumulative reward: **{st.session_state.cumulative_reward:.2f}**")

        # Get next recommendation
        user_state = st.session_state.user_state
        state_tensor = np.expand_dims(user_state, axis=0)
        q_values = agent.q_network.predict(state_tensor, verbose=0)[0]

        valid_mask = np.ones(len(movie_catalog['names']), dtype=bool)
        for idx in st.session_state.excluded:
            valid_mask[idx] = False
        masked_q = np.where(valid_mask, q_values, -np.inf)
        movie_id = int(np.argmax(masked_q))

        # Display movie
        st.subheader(f"🎬 {movie_catalog['names'][movie_id]}")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Rating", f"{movie_catalog['ratings'][movie_id]:.1f}/10")
        mc2.metric("Q-Value", f"{q_values[movie_id]:.4f}")
        mc3.metric("Movie ID", movie_id)
        st.markdown(f"**Genres:** {movie_catalog['genres'][movie_id]}")
        st.write(movie_catalog['descriptions'][movie_id][:500])

        # Feedback buttons
        st.markdown("### Your Feedback")
        fb_cols = st.columns(5)
        feedback_given = None
        for i, (fb, rw) in enumerate(FEEDBACK_REWARDS.items()):
            emoji = {"like": "👍", "watch": "👀", "click": "👆", "skip": "⏭️", "ignore": "🚫"}[fb]
            if fb_cols[i].button(f"{emoji} {fb.title()} ({rw:+.1f})", key=f"fb_{fb}_{len(st.session_state.session_history)}"):
                feedback_given = fb

        if feedback_given is not None:
            reward = FEEDBACK_REWARDS[feedback_given]
            movie_features = movie_catalog['features'][movie_id]

            # Update user state
            update_rate = 0.3
            if reward > 0:
                user_state = (1 - update_rate) * user_state + update_rate * movie_features
            else:
                user_state = (1 + update_rate * 0.5) * user_state - update_rate * 0.5 * movie_features
            norm = np.linalg.norm(user_state)
            if norm > 0:
                user_state = user_state / norm

            st.session_state.user_state = user_state
            st.session_state.excluded.append(movie_id)
            st.session_state.cumulative_reward += reward
            st.session_state.session_history.append({
                'step': len(st.session_state.session_history) + 1,
                'movie': movie_catalog['names'][movie_id],
                'genre': movie_catalog['genres'][movie_id],
                'rating': movie_catalog['ratings'][movie_id],
                'feedback': feedback_given,
                'reward': reward,
                'cumulative': st.session_state.cumulative_reward,
            })
            st.rerun()

        # Show history
        if st.session_state.session_history:
            st.divider()
            st.subheader("Session History")
            hist_df = pd.DataFrame(st.session_state.session_history)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

            # Reward over time
            fig = px.line(
                hist_df, x='step', y='cumulative',
                title="Cumulative Reward Over Session",
                markers=True,
            )
            fig.update_layout(xaxis_title="Step", yaxis_title="Cumulative Reward")
            st.plotly_chart(fig, use_container_width=True)

            # Feedback distribution
            fb_counts = hist_df['feedback'].value_counts()
            fig2 = px.pie(names=fb_counts.index, values=fb_counts.values,
                          title="Feedback Distribution", color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig2, use_container_width=True)


# ─── TRAINING DASHBOARD ─────────────────────────────────────────────────────
elif page == "📊 Training Dashboard":
    st.title("📊 Training Dashboard")

    if training_history is None:
        st.warning("No training history found at `models/training_history.pkl`.")
    else:
        st.success(f"Training history loaded — **{len(training_history.get('episode_rewards', []))}** episodes")

        # Episode rewards
        if 'episode_rewards' in training_history:
            rewards = training_history['episode_rewards']
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=rewards, mode='lines', name='Episode Reward',
                                     line=dict(color='rgba(99,110,250,0.3)')))
            # Moving average
            window = min(50, len(rewards) // 5) if len(rewards) > 10 else 1
            ma = pd.Series(rewards).rolling(window).mean()
            fig.add_trace(go.Scatter(y=ma, mode='lines', name=f'{window}-Episode MA',
                                     line=dict(color='red', width=2)))
            fig.update_layout(title="Episode Rewards", xaxis_title="Episode",
                              yaxis_title="Cumulative Reward", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Loss curve
        if 'losses' in training_history:
            losses = training_history['losses']
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=losses, mode='lines', name='Loss',
                                      line=dict(color='rgba(239,85,59,0.3)')))
            window = min(100, len(losses) // 5) if len(losses) > 20 else 1
            ma = pd.Series(losses).rolling(window).mean()
            fig2.add_trace(go.Scatter(y=ma, mode='lines', name=f'{window}-Step MA',
                                      line=dict(color='blue', width=2)))
            fig2.update_layout(title="Training Loss", xaxis_title="Step",
                               yaxis_title="MSE Loss", height=400)
            st.plotly_chart(fig2, use_container_width=True)

        # Epsilon decay
        if 'epsilons' in training_history:
            fig3 = px.line(y=training_history['epsilons'], title="Epsilon Decay",
                           labels={'x': 'Episode', 'y': 'Epsilon'})
            st.plotly_chart(fig3, use_container_width=True)

        # Summary stats
        st.subheader("Training Summary")
        summary_data = {}
        for key, vals in training_history.items():
            if isinstance(vals, (list, np.ndarray)) and len(vals) > 0:
                arr = np.array(vals)
                if arr.dtype.kind in ('f', 'i'):
                    summary_data[key] = {
                        'count': len(arr),
                        'mean': float(np.mean(arr)),
                        'std': float(np.std(arr)),
                        'min': float(np.min(arr)),
                        'max': float(np.max(arr)),
                    }
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data).T, use_container_width=True)


# ─── MODEL ARCHITECTURE ─────────────────────────────────────────────────────
elif page == "🧠 Model Architecture":
    st.title("🧠 DQN Network Architecture")

    # Network summary
    st.subheader("Layer Details")
    layers_info = []
    for layer in agent.q_network.layers:
        w = layer.get_weights()
        n_params = sum(np.prod(wi.shape) for wi in w)
        layers_info.append({
            'Layer': layer.name,
            'Type': layer.__class__.__name__,
            'Output Shape': str(layer.output_shape),
            'Parameters': int(n_params),
        })
    layers_df = pd.DataFrame(layers_info)
    st.dataframe(layers_df, use_container_width=True, hide_index=True)

    total_params = agent.q_network.count_params()
    st.metric("Total Parameters", f"{total_params:,}")

    # Architecture diagram (Mermaid-like using Plotly)
    st.subheader("Dueling Architecture Diagram")
    fig = go.Figure()

    boxes = [
        ("Input (310D)", 0, 5, "#FFB3B3", "#8B0000"),
        ("Dense 512 · LeakyReLU + BN", 1, 5, "#B3FFB3", "#004D00"),
        ("Dense 256 · LeakyReLU + BN", 2, 5, "#B3FFB3", "#004D00"),
        ("Value Stream · 128 → 1", 3, 3, "#99E6FF", "#003D5C"),
        ("Advantage Stream · 128 → 7840", 3, 7, "#99E6FF", "#003D5C"),
        ("Q(s,a) = V + (A − mean(A)) · 7840 actions", 4, 5, "#D1B3FF", "#2D004D"),
    ]
    for label, row, col, color, text_color in boxes:
        fig.add_shape(type="rect", x0=col - 1.8, x1=col + 1.8, y0=-row * 2, y1=-row * 2 + 1.2,
                      fillcolor=color, line=dict(color="#222", width=2.5))
        fig.add_annotation(x=col, y=-row * 2 + 0.6, text=f"<b>{label}</b>", showarrow=False,
                           font=dict(size=14, color=text_color, family="Arial Black"),
                           align="center")

    # Arrows
    arrows = [(5, -0.8, 5, -1.2), (5, -2.8, 5, -3.2),
              (4, -4.8, 3, -5.2), (6, -4.8, 7, -5.2),
              (3, -6.8, 5, -7.2), (7, -6.8, 5, -7.2)]
    for x0, y0, x1, y1 in arrows:
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowsize=1.8, arrowwidth=2, arrowcolor="#333")

    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.5, 10.5]),
        yaxis=dict(visible=False, range=[-9.5, 2]),
        height=600, margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature breakdown
    st.subheader("Input Feature Breakdown")
    feature_data = {
        'Component': ['Genre (Multi-hot)', 'TF-IDF + SVD Embedding', 'Rating (Normalised)'],
        'Dimensions': [processor.genre_dim, processor.embedding_dim, 1],
    }
    fig_feat = px.pie(
        names=feature_data['Component'],
        values=feature_data['Dimensions'],
        title=f"State Vector Composition ({processor.feature_dim}D total)",
        color_discrete_sequence=["#FF6B6B", "#4ECDC4", "#FFE66D"],
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    # Weight distribution
    st.subheader("Weight Distributions")
    layer_to_show = st.selectbox("Select Layer", [l.name for l in agent.q_network.layers if l.get_weights()])
    for layer in agent.q_network.layers:
        if layer.name == layer_to_show:
            weights = layer.get_weights()
            if len(weights) > 0:
                w = weights[0].flatten()
                fig_w = px.histogram(x=w, nbins=100, title=f"Weight Distribution: {layer.name}",
                                     labels={'x': 'Weight Value', 'y': 'Count'},
                                     color_discrete_sequence=["#667eea"])
                st.plotly_chart(fig_w, use_container_width=True)
            if len(weights) > 1:
                b = weights[1].flatten()
                fig_b = px.histogram(x=b, nbins=50, title=f"Bias Distribution: {layer.name}",
                                     labels={'x': 'Bias Value', 'y': 'Count'},
                                     color_discrete_sequence=["#e74c3c"])
                st.plotly_chart(fig_b, use_container_width=True)
            break


# ─── DATASET ANALYTICS ───────────────────────────────────────────────────────
elif page == "📈 Dataset Analytics":
    st.title("📈 IMDB Movies Dataset Analytics")
    st.markdown(f"**{len(df):,}** movies loaded from `imdb_movies_2025_cleaned.csv`")

    # Rating distribution
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='rating', nbins=40, title="Rating Distribution",
                           color_discrete_sequence=["#FF6B6B"])
        fig.add_vline(x=df['rating'].mean(), line_dash="dash", line_color="blue",
                       annotation_text=f"Mean: {df['rating'].mean():.2f}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Mean Rating", f"{df['rating'].mean():.2f}")
        st.metric("Median Rating", f"{df['rating'].median():.2f}")
        st.metric("Std Dev", f"{df['rating'].std():.2f}")
        st.metric("Range", f"{df['rating'].min():.1f} – {df['rating'].max():.1f}")

    # Genre distribution
    all_genres_list = []
    for genres in df['genre'].str.split(', '):
        all_genres_list.extend(genres)
    genre_counts = pd.Series(all_genres_list).value_counts()

    fig_genres = px.bar(
        x=genre_counts.head(25).values,
        y=genre_counts.head(25).index,
        orientation='h',
        title="Top 25 Genres",
        labels={'x': 'Count', 'y': 'Genre'},
        color=genre_counts.head(25).values,
        color_continuous_scale="Sunset",
    )
    fig_genres.update_layout(yaxis=dict(autorange="reversed"), height=600)
    st.plotly_chart(fig_genres, use_container_width=True)

    # Description length
    df['desc_len'] = df['description'].fillna('').str.len()
    fig_desc = px.histogram(df, x='desc_len', nbins=50, title="Description Length Distribution",
                            color_discrete_sequence=["#4ECDC4"])
    st.plotly_chart(fig_desc, use_container_width=True)

    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)

    # Search movies
    st.subheader("🔍 Search Movies")
    query = st.text_input("Search by title or genre")
    if query:
        mask = (
            df['movie_name'].str.contains(query, case=False, na=False) |
            df['genre'].str.contains(query, case=False, na=False)
        )
        results = df[mask]
        st.write(f"Found **{len(results)}** results")
        st.dataframe(results.head(100), use_container_width=True, hide_index=True)


# ─── PIPELINE VISUALIZER ─────────────────────────────────────────────────────
elif page == "🔬 Pipeline Visualizer":
    st.title("🔬 Complete DQN Recommendation Pipeline")
    st.markdown("Watch the **end-to-end internal workflow** — from raw movie data to final recommendation — with live numbers from the trained model.")

    # ── Controls ─────────────────────────────────────────────────────────────
    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        movie_idx = st.selectbox(
            "Pick a movie to trace through the pipeline",
            range(len(movie_catalog['names'])),
            format_func=lambda i: f"{movie_catalog['names'][i]}  ({movie_catalog['genres'][i][:40]})",
            index=0,
        )
    with ctrl2:
        seed_pipe = st.number_input("User-state seed", value=42, step=1, key="pipe_seed")

    if st.button("▶  Run Pipeline", type="primary", key="run_pipe"):
        np.random.seed(int(seed_pipe))
        user_state = np.random.randn(processor.feature_dim).astype(np.float32) * 0.1
        user_state = user_state / (np.linalg.norm(user_state) + 1e-8)

        movie_name = movie_catalog['names'][movie_idx]
        movie_genre = movie_catalog['genres'][movie_idx]
        movie_rating = movie_catalog['ratings'][movie_idx]
        movie_desc = movie_catalog['descriptions'][movie_idx]
        movie_feat = movie_catalog['features'][movie_idx]

        # ================================================================
        # STEP 1 — RAW MOVIE DATA
        # ================================================================
        st.markdown("---")
        st.header("📽️ Step 1 — Raw Movie Data Input")
        s1a, s1b = st.columns([1, 2])
        with s1a:
            st.metric("Movie", movie_name)
            st.metric("Rating", f"{movie_rating:.1f} / 10")
        with s1b:
            st.markdown(f"**Genres:** {movie_genre}")
            st.markdown(f"**Description:** {movie_desc[:300]}{'...' if len(movie_desc)>300 else ''}")

        # ================================================================
        # STEP 2 — FEATURE ENGINEERING
        # ================================================================
        st.markdown("---")
        st.header("⚙️ Step 2 — Feature Engineering Pipeline")

        genres_list = [g.strip() for g in movie_genre.split(',')]
        genre_vector = processor.get_genre_vector(genres_list)
        active_idx = np.where(genre_vector == 1)[0]

        # 2A — Genre Multi-Hot
        st.subheader("2A · Genre Multi-Hot Encoding")
        s2a1, s2a2 = st.columns([1, 2])
        with s2a1:
            st.metric("Vocabulary size", processor.genre_dim)
            st.metric("Active genres", len(active_idx))
            st.write("**Active indices:**", list(active_idx.astype(int)))
        with s2a2:
            genre_bar_names = [processor.genre_list[i] for i in range(min(50, processor.genre_dim))]
            genre_bar_vals = genre_vector[:50].astype(float)
            colours = ['#2ecc71' if v == 1 else '#ecf0f1' for v in genre_bar_vals]
            fig_g = go.Figure(go.Bar(x=genre_bar_names, y=genre_bar_vals, marker_color=colours))
            fig_g.update_layout(title="Genre Vector (first 50 dims)", yaxis_title="Value",
                                height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig_g, use_container_width=True)

        # 2B — Description Embedding
        st.subheader("2B · Description Embedding (TF-IDF → SVD)")
        embed_start = processor.genre_dim
        embed_end = embed_start + processor.embedding_dim
        embedding = movie_feat[embed_start:embed_end]

        e1, e2 = st.columns([1, 2])
        with e1:
            st.markdown(
                f"**Pipeline:** Tokenize → TF-IDF (5 000 D sparse) → SVD → **{processor.embedding_dim}D dense**"
            )
            st.metric("Mean", f"{embedding.mean():.6f}")
            st.metric("Std", f"{embedding.std():.6f}")
            st.metric("Min / Max", f"{embedding.min():.4f} / {embedding.max():.4f}")
        with e2:
            top_abs = np.argsort(np.abs(embedding))[::-1][:15]
            fig_e = go.Figure(go.Bar(
                x=[f"Dim {i}" for i in top_abs],
                y=embedding[top_abs],
                marker_color=['#e74c3c' if v < 0 else '#3498db' for v in embedding[top_abs]],
            ))
            fig_e.update_layout(title="Top 15 Embedding Dimensions (by |value|)",
                                yaxis_title="Value", height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig_e, use_container_width=True)

        # 2C — Rating Normalisation
        st.subheader("2C · Rating Normalisation")
        norm_rating = movie_feat[-1]
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Original", f"{movie_rating:.1f} / 10")
        rc2.metric("Normalised", f"{norm_rating:.6f}")
        rc3.metric("Method", "Min-Max → [0, 1]")

        # Final feature vector summary
        st.subheader("✅ Final Feature Vector")
        fv1, fv2, fv3, fv4 = st.columns(4)
        fv1.metric("Genre dims", f"{processor.genre_dim}D")
        fv2.metric("Embedding dims", f"{processor.embedding_dim}D")
        fv3.metric("Rating dim", "1D")
        fv4.metric("TOTAL", f"{processor.feature_dim}D")

        fig_comp = px.pie(
            names=['Genre Multi-Hot', 'TF-IDF + SVD', 'Rating'],
            values=[processor.genre_dim, processor.embedding_dim, 1],
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#FFE66D'],
            title="Feature Vector Composition",
        )
        fig_comp.update_layout(height=280, margin=dict(t=40, b=0))
        st.plotly_chart(fig_comp, use_container_width=True)

        # ================================================================
        # STEP 3 — USER STATE
        # ================================================================
        st.markdown("---")
        st.header("🧑 Step 3 — User State Construction")

        us1, us2 = st.columns([1, 2])
        with us1:
            st.metric("Dimension", f"{len(user_state)}D")
            st.metric("L2 Norm", f"{np.linalg.norm(user_state):.4f}")
            st.metric("Mean", f"{user_state.mean():.6f}")
            sparsity = int(np.sum(np.abs(user_state) < 1e-6))
            st.metric("Sparsity", f"{sparsity}/{len(user_state)} ({100*sparsity/len(user_state):.1f}%)")
        with us2:
            # Top genre preferences in user state
            genre_prefs = user_state[:processor.genre_dim]
            top_genre_idx = np.argsort(genre_prefs)[::-1][:12]
            fig_up = go.Figure(go.Bar(
                x=[processor.genre_list[i] for i in top_genre_idx],
                y=genre_prefs[top_genre_idx],
                marker_color=['#27ae60' if v > 0 else '#e74c3c' for v in genre_prefs[top_genre_idx]],
            ))
            fig_up.update_layout(title="User's Top Genre Preferences (from state vector)",
                                 yaxis_title="Weight", height=320, margin=dict(t=40, b=0))
            st.plotly_chart(fig_up, use_container_width=True)

        st.info(
            "**State Update Rule:**\n\n"
            "- **Positive** feedback (like/watch):  `new = (1 − α) · old + α · movie_features`  → moves **toward** liked movie\n"
            "- **Negative** feedback (skip/ignore): `new = (1 + α/2) · old − (α/2) · movie_features` → moves **away**\n\n"
            "Where **α = 0.3** (update rate)"
        )

        # ================================================================
        # STEP 4 — NETWORK FORWARD PASS
        # ================================================================
        st.markdown("---")
        st.header("🧠 Step 4 — Deep Q-Network Forward Pass")

        import tensorflow as tf
        state_tensor = np.expand_dims(user_state, axis=0)

        # Collect intermediate outputs
        layer_data = []
        for layer in agent.q_network.layers:
            if len(layer.get_weights()) == 0 and layer.__class__.__name__ == 'InputLayer':
                continue
            try:
                inter_model = tf.keras.Model(inputs=agent.q_network.input, outputs=layer.output)
                out = inter_model.predict(state_tensor, verbose=0)[0]
                w = layer.get_weights()
                n_params = sum(np.prod(wi.shape) for wi in w)
                layer_data.append({
                    'name': layer.name,
                    'type': layer.__class__.__name__,
                    'output_shape': out.shape,
                    'output': out,
                    'n_params': int(n_params),
                    'weights': w,
                })
            except Exception:
                pass

        for i, ld in enumerate(layer_data):
            lname = ld['name'].upper().replace('_', ' ')
            with st.expander(f"Layer {i+1}: **{lname}** — {ld['type']}  |  Output {ld['output_shape']}  |  {ld['n_params']:,} params", expanded=(i < 3)):
                c1, c2 = st.columns([1, 2])
                out = ld['output']
                with c1:
                    st.metric("Mean", f"{out.mean():.6f}")
                    st.metric("Std", f"{out.std():.6f}")
                    st.metric("Min / Max", f"{out.min():.4f} / {out.max():.4f}")
                    if ld['type'] == 'Dense':
                        total = len(out)
                        positive = int(np.sum(out > 0))
                        negative = int(np.sum(out < 0))
                        near_zero = int(np.sum(np.abs(out) < 1e-6))
                        alive = total - near_zero  # LeakyReLU: alive = non-zero output
                        st.metric("Alive neurons (LeakyReLU)", f"{alive}/{total} ({100*alive/total:.1f}%)")
                        st.caption(f"⚡ {positive} positive · {negative} negative (leaky) · {near_zero} dead")
                    if ld['weights']:
                        wm = ld['weights'][0]
                        st.caption(f"Weight matrix: {wm.shape}  mean={wm.mean():.6f}  std={wm.std():.6f}")

                with c2:
                    if len(out) <= 1000:
                        top_n = min(20, len(out))
                        top_idx = np.argsort(np.abs(out))[::-1][:top_n]
                        fig_l = go.Figure(go.Bar(
                            x=[f"N{j}" for j in top_idx],
                            y=out[top_idx],
                            marker_color=['#e74c3c' if v < 0 else '#2980b9' for v in out[top_idx]],
                        ))
                        fig_l.update_layout(title=f"Top {top_n} Activations",
                                            height=250, margin=dict(t=35, b=0))
                        st.plotly_chart(fig_l, use_container_width=True, key=f"layer_act_{i}")
                    else:
                        # For very large outputs (advantage_output = 7840)
                        fig_h = px.histogram(x=out, nbins=80, title=f"Activation Distribution ({len(out):,} values)",
                                            color_discrete_sequence=['#8e44ad'])
                        fig_h.update_layout(height=250, margin=dict(t=35, b=0))
                        st.plotly_chart(fig_h, use_container_width=True, key=f"layer_hist_{i}")

                # Special annotations for value / advantage streams
                if 'value_output' in ld['name']:
                    st.success(f"💎 **Value Stream V(s) = {out[0]:.6f}** — overall quality of current user state")
                if 'advantage_output' in ld['name']:
                    st.success(f"⚔️ **Advantage Stream A(s,a)** — relative advantage for each of {len(out):,} movies  "
                               f"(mean {out.mean():.6f}, range [{out.min():.4f}, {out.max():.4f}])")

        # ================================================================
        # STEP 5 — Q-VALUE COMPUTATION
        # ================================================================
        st.markdown("---")
        st.header("🔀 Step 5 — Dueling Q-Value Aggregation")

        q_values = agent.q_network.predict(state_tensor, verbose=0)[0]

        st.latex(r"Q(s,a) = V(s) + \left[ A(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a') \right]")

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Total Actions", f"{len(q_values):,}")
        q2.metric("Mean Q", f"{q_values.mean():.6f}")
        q3.metric("Min Q", f"{q_values.min():.6f}")
        q4.metric("Max Q", f"{q_values.max():.6f}")

        fig_qh = px.histogram(x=q_values, nbins=100, title="Q-Value Distribution Across All 7,840 Movies",
                              labels={'x': 'Q-Value', 'y': 'Count'},
                              color_discrete_sequence=['#8e44ad'])
        fig_qh.add_vline(x=q_values.mean(), line_dash='dash', line_color='red',
                          annotation_text=f"mean={q_values.mean():.4f}")
        fig_qh.update_layout(height=350, margin=dict(t=40, b=0))
        st.plotly_chart(fig_qh, use_container_width=True)

        # ================================================================
        # STEP 6 — ACTION SELECTION & TOP-K
        # ================================================================
        st.markdown("---")
        st.header("🏆 Step 6 — Action Selection & Top-K Ranking")

        st.markdown(
            f"**During training** — ε-greedy: with ε = {agent.epsilon:.4f} pick random, else argmax(Q)  \n"
            f"**During inference** — always **argmax(Q)** (greedy)"
        )

        top_k = 15
        top_idx = np.argsort(q_values)[::-1][:top_k]

        rows = []
        for rank, idx in enumerate(top_idx, 1):
            sim = float(np.dot(user_state, movie_catalog['features'][idx]) / (
                np.linalg.norm(user_state) * np.linalg.norm(movie_catalog['features'][idx]) + 1e-8))
            rows.append({
                'Rank': rank,
                'Q-Value': f"{q_values[idx]:+.6f}",
                'Movie': movie_catalog['names'][idx],
                'Genre': movie_catalog['genres'][idx][:50],
                'Rating': f"{movie_catalog['ratings'][idx]:.1f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Bar chart of top-15 Q-values
        fig_top = go.Figure(go.Bar(
            y=[movie_catalog['names'][i][:35] for i in top_idx][::-1],
            x=[q_values[i] for i in top_idx][::-1],
            orientation='h',
            marker=dict(color=[q_values[i] for i in top_idx][::-1],
                        colorscale='Viridis', showscale=True,
                        colorbar=dict(title='Q-Value')),
        ))
        fig_top.update_layout(title=f"Top {top_k} Recommendations by Q-Value",
                              xaxis_title="Q-Value", height=500, margin=dict(l=200, t=40, b=0))
        st.plotly_chart(fig_top, use_container_width=True)

        # ================================================================
        # STEP 7 — LEARNING MECHANISM
        # ================================================================
        st.markdown("---")
        st.header("📚 Step 7 — How the Network Learns")

        st.markdown(
            """
| # | Stage | Detail |
|---|---|---|
| 1 | **Interaction** | Agent sees state **s** (310 D), picks action **a** (movie), gets reward **r**, next state **s'** |
| 2 | **Store** | Transition `(s, a, r, s', done)` → Replay Buffer (capacity 50 000) |
| 3 | **Sample** | Random mini-batch of **64** transitions |
| 4 | **Target** | `target_Q = r + γ · max Q_target(s', a')`  with **γ = 0.95** |
| 5 | **Loss** | MSE: `(1/64) Σ [Q(s,a) − target_Q]²` |
| 6 | **Backprop** | Adam optimizer, **lr = 0.001** |
| 7 | **Sync** | Copy Q → Target every **50** steps |
"""
        )

        st.markdown("**Reward mapping:**")
        rew_cols = st.columns(5)
        for i, (fb, rw) in enumerate(FEEDBACK_REWARDS.items()):
            emoji = {'like': '👍', 'watch': '👀', 'click': '👆', 'skip': '⏭️', 'ignore': '🚫'}[fb]
            rew_cols[i].metric(f"{emoji} {fb.title()}", f"{rw:+.1f}")

        # ================================================================
        # STEP 8 — SUMMARY DIAGRAM
        # ================================================================
        st.markdown("---")
        st.header("🗺️ Step 8 — End-to-End Pipeline Summary")

        fig_pipe = go.Figure()
        boxes_pipe = [
            ("📥  RAW MOVIE DATA\nGenres · Rating · Description", 0, 5, "#FFDEDE", "#8B0000"),
            ("⚙️  FEATURE ENGINEERING\n209 D genre + 100 D embed + 1 D rating = 310 D", 1, 5, "#FFF3CD", "#664D03"),
            ("🧑  USER STATE  (310 D)\nDynamic preference vector", 2, 5, "#D1ECF1", "#0C5460"),
            ("🧠  DQN FORWARD PASS\n310 → 512 → 256 → Dueling → 7 840", 3, 5, "#D4EDDA", "#155724"),
            ("🔀  Q(s,a) = V(s) + A(s,a) − mean(A)\n7 840 Q-values", 4, 5, "#E2D5F1", "#2D004D"),
            ("🏆  ACTION SELECTION\nargmax(Q) → Top-K Movies", 5, 5, "#FFD6E8", "#6B0030"),
            ("📚  LEARNING\nReplay → Target → Backprop", 6, 5, "#CCE5FF", "#003566"),
        ]
        for label, row, col, bg, tc in boxes_pipe:
            y0 = -row * 1.8
            fig_pipe.add_shape(type="rect", x0=col - 3.5, x1=col + 3.5, y0=y0, y1=y0 + 1.2,
                               fillcolor=bg, line=dict(color=tc, width=2), layer='below')
            fig_pipe.add_annotation(x=col, y=y0 + 0.6, text=f"<b>{label}</b>", showarrow=False,
                                   font=dict(size=13, color=tc, family='Arial'), align='center')
            if row < 6:
                fig_pipe.add_annotation(x=5, y=y0 - 0.55, ax=5, ay=y0 - 0.05,
                                        xref='x', yref='y', axref='x', ayref='y',
                                        showarrow=True, arrowhead=3, arrowsize=1.5,
                                        arrowwidth=2, arrowcolor='#555')

        fig_pipe.update_layout(
            xaxis=dict(visible=False, range=[0, 10]),
            yaxis=dict(visible=False, range=[-12, 2]),
            height=750, margin=dict(l=20, r=20, t=10, b=10),
            plot_bgcolor='white', paper_bgcolor='white',
        )
        st.plotly_chart(fig_pipe, use_container_width=True)

        st.success(
            f"**Pipeline complete!** — {processor.feature_dim}D features → "
            f"{agent.q_network.count_params():,} parameters → "
            f"{len(q_values):,} Q-values → Top recommendation: **{movie_catalog['names'][top_idx[0]]}**"
        )


# ── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown(
    "Built with [Streamlit](https://streamlit.io)  \n"
    "DQN Agent · TensorFlow/Keras  \n"
    "© 2025 RL Movie Recommender"
)
