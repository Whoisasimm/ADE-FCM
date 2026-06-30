"""
Preprocessing module for weblog data and general dataset preparation.
Implements TOH1 heuristic, session identification, and feature extraction.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from scipy.sparse import csr_matrix, lil_matrix


class Preprocessor:
    """Preprocess weblog data and prepare feature matrices."""

    def __init__(self):
        self.session_map = None
        self.feature_names = None

    def clean_weblog_data(self, df):
        """Remove image/stylesheet requests, robots, query strings, and static assets."""
        logger.info(f"Cleaning weblog data: {len(df)} rows before")
        n_before = len(df)

        image_exts = {'.gif', '.jpg', '.jpeg', '.png', '.bmp', '.svg', '.ico', '.tiff', '.webp'}
        static_exts = {'.css', '.js', '.woff', '.woff2', '.ttf', '.eot', '.map', '.xml'}
        robot_agents = {'bot', 'spider', 'crawler', 'scanner', 'archive', 'wget', 'curl', 'python-requests'}

        if 'url' in df.columns:
            df = df[~df['url'].str.lower().str.contains(r'\?', na=False)]

        if 'path' in df.columns:
            is_image = df['path'].apply(
                lambda p: Path(str(p)).suffix.lower() in image_exts if pd.notna(p) else False
            )
            is_static = df['path'].apply(
                lambda p: Path(str(p)).suffix.lower() in static_exts if pd.notna(p) else False
            )
            df = df[~is_image & ~is_static]

        if 'method' in df.columns:
            df = df[df['method'].str.upper().isin(['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])]

        if 'status' in df.columns:
            df = df[df['status'].apply(lambda s: str(s)[0] != '4' if pd.notna(s) else True)]

        if 'user_agent' in df.columns:
            is_robot = df['user_agent'].str.lower().apply(
                lambda a: any(r in a for r in robot_agents) if pd.notna(a) else False
            )
            df = df[~is_robot]

        logger.info(f"Cleaned data: {len(df)} rows (removed {n_before - len(df)})")
        return df

    def identify_users(self, df):
        """Group sessions by IP address. Returns DataFrame with user_id column."""
        logger.info("Identifying users by IP address")
        if 'ip' in df.columns:
            df['user_id'] = df['ip'].astype('category').cat.codes
        elif 'user_id' not in df.columns:
            df['user_id'] = 0
        logger.info(f"Identified {df['user_id'].nunique()} unique users")
        return df

    def identify_sessions(self, df, timeout_minutes=30):
        """TOH1 heuristic: split user activity into sessions by inactivity timeout.

        A new session starts if time since last request exceeds timeout_minutes.
        """
        logger.info(f"Identifying sessions with {timeout_minutes}min timeout")
        timeout_seconds = timeout_minutes * 60

        if 'timestamp' not in df.columns:
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'timestamp'})
            elif 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            else:
                logger.warning("No timestamp column found; assigning single session")
                df['session_id'] = 0
                return df

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['user_id', 'timestamp'])
        df['session_id'] = 0

        for uid in df['user_id'].unique():
            mask = df['user_id'] == uid
            user_idx = df[mask].index
            times = df.loc[user_idx, 'timestamp'].values
            diffs = np.zeros(len(times))
            if len(times) > 1:
                diffs[1:] = (times[1:] - times[:-1]).astype('timedelta64[s]').astype(float)
            session_starts = np.where(diffs > timeout_seconds)[0]
            session_ids = np.zeros(len(times), dtype=int)
            sid = 0
            for i in range(len(times)):
                if i in session_starts:
                    sid += 1
                session_ids[i] = sid
            df.loc[user_idx, 'session_id'] = session_ids

        n_sessions = df['session_id'].nunique()
        logger.info(f"Identified {n_sessions} sessions")
        self.session_map = df[['user_id', 'session_id', 'timestamp']].copy()
        return df

    def reduce_dimensions(self, df, min_support=1):
        """Remove URLs / pages with support below min_support."""
        logger.info(f"Reducing dimensions with min_support={min_support}")
        if 'url' in df.columns:
            url_counts = df['url'].value_counts()
            valid_urls = url_counts[url_counts >= min_support].index
            df = df[df['url'].isin(valid_urls)]
        elif 'path' in df.columns:
            path_counts = df['path'].value_counts()
            valid_paths = path_counts[path_counts >= min_support].index
            df = df[df['path'].isin(valid_paths)]
        logger.info(f"After dimension reduction: {len(df)} rows")
        return df

    def assign_session_weights(self, df):
        """Assign weight to each session inversely proportional to session length."""
        logger.info("Assigning session weights")
        session_lengths = df.groupby('session_id').size()
        max_len = session_lengths.max()
        if max_len > 0:
            weights = 1.0 / (session_lengths / max_len + 0.1)
        else:
            weights = pd.Series(1.0, index=session_lengths.index)
        df['session_weight'] = df['session_id'].map(weights)
        return df

    def build_session_matrix(self, df):
        """Build sparse row-major session-page matrix.

        Rows = sessions, Columns = pages/URLs, Values = visit counts (weighted).
        Returns sparse CSR matrix and feature names.
        """
        logger.info("Building session-page matrix")

        if 'url' in df.columns:
            page_col = 'url'
        elif 'path' in df.columns:
            page_col = 'path'
        else:
            logger.warning("No URL/path column found; using dummy features")
            n = len(df)
            return csr_matrix(np.ones((n, 1))), ['dummy']

        pages = df[page_col].unique()
        page_to_idx = {p: i for i, p in enumerate(pages)}
        n_sessions = df['session_id'].nunique()
        n_pages = len(pages)

        mat = lil_matrix((n_sessions, n_pages), dtype=np.float64)
        weight_col = 'session_weight' if 'session_weight' in df.columns else None

        for sid, group in df.groupby('session_id'):
            for _, row in group.iterrows():
                pidx = page_to_idx[row[page_col]]
                w = row[weight_col] if weight_col else 1.0
                mat[sid, pidx] += w

        logger.info(f"Built session matrix: {mat.shape} with {mat.nnz} non-zero entries")
        self.feature_names = list(pages)
        return mat.tocsr(), self.feature_names

    def normalize(self, matrix):
        """Divide each column by its max value (L_infinity norm per feature)."""
        logger.info("Normalizing matrix columns by max value")
        if isinstance(matrix, np.ndarray):
            col_max = np.max(matrix, axis=0)
            col_max = np.maximum(col_max, 1e-10)
            return matrix / col_max
        else:
            col_max = matrix.max(axis=0).toarray().ravel()
            col_max = np.maximum(col_max, 1e-10)
            return matrix.multiply(1.0 / col_max).tocsr()

    def standardize(self, X):
        """Z-score standardization: (x - mean) / std."""
        logger.info("Applying z-score standardization")
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.maximum(std, 1e-10)
        return (X - mean) / std

    def min_max_scale(self, X):
        """Min-max scaling to [0, 1]."""
        logger.info("Applying min-max scaling to [0, 1]")
        x_min = np.min(X, axis=0)
        x_max = np.max(X, axis=0)
        denom = np.maximum(x_max - x_min, 1e-10)
        return (X - x_min) / denom
