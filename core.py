"""
Core classes for the DQN Movie Recommendation System.
Extracted from main.ipynb for use in Streamlit app.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import random
import pickle
import os
from collections import deque
from typing import List, Tuple, Dict, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Reward mapping
FEEDBACK_REWARDS = {
    'like': 1.5,
    'watch': 0.8,
    'click': 0.4,
    'skip': -0.2,
    'ignore': -0.5
}


class MovieFeatureProcessor:
    def __init__(self, embedding_dim: int = 100, n_tfidf_features: int = 5000):
        self.embedding_dim = embedding_dim
        self.n_tfidf_features = n_tfidf_features
        self.mlb = MultiLabelBinarizer()
        self.tfidf = TfidfVectorizer(max_features=n_tfidf_features, stop_words='english')
        self.svd = TruncatedSVD(n_components=embedding_dim, random_state=SEED)
        self.rating_scaler = MinMaxScaler()
        self.genre_dim = None
        self.feature_dim = None
        self.genre_list = None

    def fit(self, df: pd.DataFrame) -> 'MovieFeatureProcessor':
        genres_split = df['genre'].apply(lambda x: [g.strip() for g in x.split(',')])
        self.mlb.fit(genres_split)
        self.genre_list = list(self.mlb.classes_)
        self.genre_dim = len(self.genre_list)
        self.tfidf.fit(df['description'].fillna(''))
        tfidf_matrix = self.tfidf.transform(df['description'].fillna(''))
        self.svd.fit(tfidf_matrix)
        self.rating_scaler.fit(df[['rating']])
        self.feature_dim = self.genre_dim + self.embedding_dim + 1
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        genres_split = df['genre'].apply(lambda x: [g.strip() for g in x.split(',')])
        genre_features = self.mlb.transform(genres_split)
        tfidf_matrix = self.tfidf.transform(df['description'].fillna(''))
        desc_embeddings = self.svd.transform(tfidf_matrix)
        rating_features = self.rating_scaler.transform(df[['rating']])
        features = np.hstack([genre_features, desc_embeddings, rating_features])
        return features.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def get_genre_vector(self, genres: List[str]) -> np.ndarray:
        return self.mlb.transform([genres])[0]

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'MovieFeatureProcessor':
        import core as _core_module
        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == '__main__':
                    module = 'core'
                return super().find_class(module, name)
        with open(filepath, 'rb') as f:
            return _Unpickler(f).load()


def build_dqn_model(state_dim: int, action_dim: int, hidden_layers: List[int] = [512, 256, 128]) -> keras.Model:
    state_input = layers.Input(shape=(state_dim,), name='state_input')
    x = state_input
    for i, units in enumerate(hidden_layers[:-1]):
        x = layers.Dense(units, activation=layers.LeakyReLU(alpha=0.01), name=f'shared_dense_{i}')(x)
        x = layers.BatchNormalization(name=f'bn_{i}')(x)
        x = layers.Dropout(0.2, name=f'dropout_{i}')(x)
    value = layers.Dense(hidden_layers[-1], activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='value_dense')(x)
    value = layers.Dense(1, name='value_output')(value)
    advantage = layers.Dense(hidden_layers[-1], activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='advantage_dense')(x)
    advantage = layers.Dense(action_dim, name='advantage_output')(advantage)
    q_values = layers.Lambda(
        lambda x: x[0] + (x[1] - tf.reduce_mean(x[1], axis=1, keepdims=True)),
        name='q_values'
    )([value, advantage])
    model = keras.Model(inputs=state_input, outputs=q_values, name='DQN')
    return model


def build_simple_dqn_model(state_dim: int, action_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(state_dim,)),
        layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
        layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
        layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
        layers.Dense(action_dim, activation='linear')
    ], name='SimpleDQN')
    return model


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        use_dueling: bool = True
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate

        if use_dueling:
            self.q_network = build_dqn_model(state_dim, action_dim)
            self.target_network = build_dqn_model(state_dim, action_dim)
        else:
            self.q_network = build_simple_dqn_model(state_dim, action_dim)
            self.target_network = build_simple_dqn_model(state_dim, action_dim)

        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.q_network.compile(optimizer=self.optimizer, loss='mse')
        self.update_target_network()
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.train_step = 0
        self.loss_history = []

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def select_action(self, state: np.ndarray, valid_actions: List[int] = None, training: bool = True) -> int:
        if valid_actions is None:
            valid_actions = list(range(self.action_dim))
        if training and np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        state_tensor = np.expand_dims(state, axis=0)
        q_values = self.q_network.predict(state_tensor, verbose=0)[0]
        masked_q = np.full(self.action_dim, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked_q))

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        self.q_network.save(f"{filepath}_q_network.keras")
        self.target_network.save(f"{filepath}_target_network.keras")
        params = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'learning_rate': self.learning_rate
        }
        with open(f"{filepath}_params.pkl", 'wb') as f:
            pickle.dump(params, f)

    @classmethod
    def load(cls, filepath: str) -> 'DQNAgent':
        with open(f"{filepath}_params.pkl", 'rb') as f:
            params = pickle.load(f)
        agent = cls(
            state_dim=params['state_dim'],
            action_dim=params['action_dim'],
            gamma=params['gamma'],
            epsilon_start=params['epsilon'],
            epsilon_end=params['epsilon_end'],
            epsilon_decay=params['epsilon_decay'],
            batch_size=params['batch_size'],
            target_update_freq=params['target_update_freq'],
            learning_rate=params['learning_rate'],
            use_dueling=False
        )
        agent.q_network = keras.models.load_model(f"{filepath}_q_network.keras", compile=False)
        agent.q_network.compile(optimizer=optimizers.Adam(learning_rate=params['learning_rate']), loss='mse')
        agent.target_network = keras.models.load_model(f"{filepath}_target_network.keras", compile=False)
        return agent


class MovieRecommender:
    def __init__(self, agent: DQNAgent, movie_catalog: Dict, processor: MovieFeatureProcessor):
        self.agent = agent
        self.movie_catalog = movie_catalog
        self.processor = processor
        self.n_movies = len(movie_catalog['names'])

    def get_top_k_recommendations(
        self,
        user_state: np.ndarray = None,
        k: int = 10,
        excluded_movies: List[int] = None
    ) -> List[Dict]:
        if user_state is None:
            user_state = np.random.randn(self.processor.feature_dim).astype(np.float32) * 0.1
        if excluded_movies is None:
            excluded_movies = []
        state_tensor = np.expand_dims(user_state, axis=0)
        q_values = self.agent.q_network.predict(state_tensor, verbose=0)[0]
        valid_mask = np.ones(self.n_movies, dtype=bool)
        valid_mask[excluded_movies] = False
        masked_q = np.where(valid_mask, q_values, -np.inf)
        top_k_indices = np.argsort(masked_q)[-k:][::-1]
        recommendations = []
        for rank, movie_id in enumerate(top_k_indices, 1):
            recommendations.append({
                'rank': rank,
                'movie_id': int(movie_id),
                'movie_name': self.movie_catalog['names'][movie_id],
                'genre': self.movie_catalog['genres'][movie_id],
                'rating': self.movie_catalog['ratings'][movie_id],
                'description': self.movie_catalog['descriptions'][movie_id],
                'q_value': float(q_values[movie_id])
            })
        return recommendations

    def recommend_by_genre_preference(self, preferred_genres: List[str], k: int = 10) -> List[Dict]:
        genre_vector = self.processor.get_genre_vector(preferred_genres)
        embedding = np.random.randn(self.processor.embedding_dim).astype(np.float32) * 0.1
        rating = np.array([0.7])
        user_state = np.concatenate([genre_vector, embedding, rating]).astype(np.float32)
        user_state = user_state / (np.linalg.norm(user_state) + 1e-8)
        return self.get_top_k_recommendations(user_state, k)


class HybridPersonalizedRecommender:
    def __init__(self, agent, movie_catalog, processor):
        self.agent = agent
        self.movie_catalog = movie_catalog
        self.processor = processor
        self.n_movies = len(movie_catalog['names'])

    def personalized_recommend(self, user_state, k=10, alpha=0.5):
        q_values = self.agent.q_network.predict(np.expand_dims(user_state, 0), verbose=0)[0]
        q_norm = (q_values - q_values.min()) / (q_values.max() - q_values.min() + 1e-8)
        similarities = []
        for movie_features in self.movie_catalog['features']:
            sim = np.dot(user_state, movie_features) / (
                np.linalg.norm(user_state) * np.linalg.norm(movie_features) + 1e-8
            )
            similarities.append(sim)
        similarities = np.array(similarities)
        sim_norm = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)
        hybrid_scores = (1 - alpha) * q_norm + alpha * sim_norm
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        recommendations = []
        for rank, idx in enumerate(top_k_indices, 1):
            recommendations.append({
                'rank': rank,
                'movie_id': int(idx),
                'movie_name': self.movie_catalog['names'][idx],
                'genre': self.movie_catalog['genres'][idx],
                'rating': self.movie_catalog['ratings'][idx],
                'q_value': float(q_values[idx]),
                'similarity': float(similarities[idx]),
                'hybrid_score': float(hybrid_scores[idx])
            })
        return recommendations


class SimulatedUser:
    def __init__(self, feature_dim: int, preference_strength: float = 0.7):
        self.feature_dim = feature_dim
        self.preference_strength = preference_strength
        self.preference_vector = np.random.randn(feature_dim).astype(np.float32)
        self.preference_vector = self.preference_vector / (np.linalg.norm(self.preference_vector) + 1e-8)
        self.interaction_history = []

    def reset(self):
        self.preference_vector = np.random.randn(self.feature_dim).astype(np.float32)
        self.preference_vector = self.preference_vector / (np.linalg.norm(self.preference_vector) + 1e-8)
        self.interaction_history = []

    def get_feedback(self, movie_features: np.ndarray) -> Tuple[str, float]:
        similarity = np.dot(self.preference_vector, movie_features) / (
            np.linalg.norm(self.preference_vector) * np.linalg.norm(movie_features) + 1e-8
        )
        noise = np.random.normal(0, 0.1)
        score = similarity * self.preference_strength + noise
        if score > 0.3:
            feedback = 'like'
        elif score > 0.15:
            feedback = 'watch'
        elif score > 0.0:
            feedback = 'click'
        elif score > -0.15:
            feedback = 'skip'
        else:
            feedback = 'ignore'
        reward = FEEDBACK_REWARDS[feedback]
        self.interaction_history.append((movie_features, feedback, reward))
        return feedback, reward


class MovieRecommendationEnv:
    def __init__(
        self,
        movie_features: np.ndarray,
        movie_names: List[str],
        movie_genres: List[str],
        episode_length: int = 10,
        user_profile_update_rate: float = 0.3,
        mode: str = 'simulated'
    ):
        self.movie_features = movie_features
        self.movie_names = movie_names
        self.movie_genres = movie_genres
        self.n_movies = len(movie_names)
        self.feature_dim = movie_features.shape[1]
        self.episode_length = episode_length
        self.update_rate = user_profile_update_rate
        self.mode = mode
        self.state_dim = self.feature_dim
        self.action_dim = self.n_movies
        self.user_state = None
        self.simulated_user = None
        self.step_count = 0
        self.recommended_movies = set()
        self.episode_rewards = []

    def reset(self) -> np.ndarray:
        self.user_state = np.random.randn(self.feature_dim).astype(np.float32) * 0.1
        if self.mode == 'simulated':
            self.simulated_user = SimulatedUser(self.feature_dim)
        self.step_count = 0
        self.recommended_movies = set()
        self.episode_rewards = []
        return self.user_state.copy()

    def step(self, action: int, human_feedback: str = None) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        movie_features = self.movie_features[action]
        movie_name = self.movie_names[action]
        movie_genre = self.movie_genres[action]
        if self.mode == 'simulated':
            feedback, reward = self.simulated_user.get_feedback(movie_features)
        else:
            if human_feedback is None:
                raise ValueError("Human feedback required in human mode")
            feedback = human_feedback
            reward = FEEDBACK_REWARDS.get(feedback, 0.0)
        if action in self.recommended_movies:
            reward -= 0.3
        self.recommended_movies.add(action)
        if reward > 0:
            self.user_state = (1 - self.update_rate) * self.user_state + self.update_rate * movie_features
        else:
            self.user_state = (1 + self.update_rate * 0.5) * self.user_state - self.update_rate * 0.5 * movie_features
        norm = np.linalg.norm(self.user_state)
        if norm > 0:
            self.user_state = self.user_state / norm
        self.episode_rewards.append(reward)
        done = self.step_count >= self.episode_length
        info = {
            'movie_name': movie_name,
            'movie_genre': movie_genre,
            'feedback': feedback,
            'step': self.step_count,
            'cumulative_reward': sum(self.episode_rewards)
        }
        return self.user_state.copy(), reward, done, info

    def get_valid_actions(self) -> List[int]:
        return [i for i in range(self.n_movies) if i not in self.recommended_movies]
