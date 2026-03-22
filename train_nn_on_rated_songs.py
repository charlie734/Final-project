#neural network training for song rating prediction
#uses k-fold cross-validation and ensemble approach

import os
import math
import time
import random
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


#constants
@dataclass
class FeatureConstants:
    MAX_TEMPO_BPM: float = 200.0
    MS_PER_MINUTE: float = 60000.0
    MIN_DECADE: int = 1950
    MAX_DECADE: int = 2020
    MIN_VALID_YEAR: int = 1900
    RATING_MIN: int = 1
    RATING_MAX: int = 5
    AUDIO_FEATURE_MIN: float = 0.0
    AUDIO_FEATURE_MAX: float = 1.0

CONSTANTS = FeatureConstants()


#training configuration
@dataclass
class TrainConfig:
    epochs: int = 600
    batch_size: int = 112
    lr: float = 2.8e-4
    weight_decay: float = 1.5e-6
    patience: int = 50
    min_delta: float = 1e-7
    grad_clip: float = 0.25
    use_amp: bool = True


#neural network architecture
class ResidualBlock(nn.Module):
    def __init__(self, width: int, p: float = 0.25):
        super().__init__()
        self.lin1 = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.lin2 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = F.silu(self.norm1(self.lin1(x)))
        h = self.drop(h)
        h = self.lin2(self.norm2(h))
        return x + F.dropout(h, p=self.drop.p, training=self.training)


class RatingMLP(nn.Module):
    def __init__(self, in_dim: int, width: int = 768, depth: int = 8, p: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(p)
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width, p=p) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(width // 2, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


#utility functions
def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_pref: str = "auto") -> torch.device:
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_pref)


#data loading and feature engineering
def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Dataset is empty")
    
    if 'rating' not in df.columns:
        raise ValueError("Dataset must contain 'rating' column")
    
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]: #feature engineering
    df = df.copy()
    
    #validation
    if 'rating' not in df.columns:
        raise ValueError("Missing 'rating' column")
    
    #extract target
    y = df['rating'].copy()
    
    #audio features
    audio_features = [
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence'
    ]
    
    #tempo features
    if 'tempo' in df.columns:
        df['tempo_normalized'] = df['tempo'] / CONSTANTS.MAX_TEMPO_BPM
        df['tempo_log'] = np.log1p(df['tempo'])
    
    #duration features
    if 'duration_ms' in df.columns:
        df['duration_minutes'] = df['duration_ms'] / CONSTANTS.MS_PER_MINUTE
        df['duration_log'] = np.log1p(df['duration_ms'])
    
    #year features
    if 'track_album_release_date' in df.columns:
        df['year'] = pd.to_datetime(df['track_album_release_date'], errors='coerce').dt.year
        df['year'] = df['year'].fillna(df['year'].median())
        df['decade'] = ((df['year'] // 10) * 10).clip(CONSTANTS.MIN_DECADE, CONSTANTS.MAX_DECADE)
        df['years_since_2000'] = df['year'] - 2000
    
    #key and mode
    if 'key' in df.columns:
        df['key'] = df['key'].fillna(0)
    if 'mode' in df.columns:
        df['mode'] = df['mode'].fillna(0)
    
    #loudness
    if 'loudness' in df.columns:
        df['loudness_normalized'] = (df['loudness'] + 60) / 60  # Rough normalization
    
    #time signature
    if 'time_signature' in df.columns:
        df['time_signature'] = df['time_signature'].fillna(4)
    
    #popularity
    if 'track_popularity' in df.columns:
        df['popularity_normalized'] = df['track_popularity'] / 100.0
    
    #select feature columns
    feature_cols = []
    
    #add audio features
    for col in audio_features:
        if col in df.columns:
            feature_cols.append(col)
    
    #add engineered features
    engineered_features = [
        'tempo_normalized', 'tempo_log', 'duration_minutes', 'duration_log',
        'year', 'decade', 'years_since_2000', 'key', 'mode',
        'loudness_normalized', 'time_signature', 'popularity_normalized'
    ]
    
    for col in engineered_features:
        if col in df.columns:
            feature_cols.append(col)
    
    #create feature matrix
    X_df = df[feature_cols].copy()
    
    #fill any remaining NaN values
    X_df = X_df.fillna(X_df.median())
    
    return X_df, y


#model training
def train_one(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    device: torch.device,
    model_hparams: Dict = None,
    cfg: TrainConfig = TrainConfig(),
    seed: int = 42,
    fold: int = None
) -> Tuple[nn.Module, List]:
    
    set_seed(seed)
    
    #create model
    hparams = model_hparams or {}
    model = RatingMLP(X_train.shape[1], **hparams).to(device)
    
    #create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    #optimiser and loss
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()
    
    #training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(cfg.epochs):
        #training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimiser.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            
            optimiser.step()
            train_loss += loss.item()
        
        #validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        history.append({'train_loss': train_loss, 'val_loss': val_loss})
        
        #early stopping
        if val_loss < best_val_loss - cfg.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break
    
    return model, history


#k-fold cross-validation
@dataclass
class CVResult:
    fold_rmse: List[float]
    mean_rmse: float
    std_rmse: float


def kfold_cv_torch(
    X: pd.DataFrame, y: pd.Series,
    n_splits: int = 5,
    device: Optional[torch.device] = None,
    cfg: TrainConfig = TrainConfig(),
    model_hparams: Dict = None,
    seed: int = 42,
    save_best_to: Optional[str] = None
) -> CVResult:

    set_seed(seed)
    device = device or get_device("auto")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    feature_cols = list(X.columns)
    rmses = []
    best_overall = float("inf")

    for fold, (tr, va) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[tr].to_numpy(), X.iloc[va].to_numpy()
        y_tr, y_va = y.iloc[tr].to_numpy(), y.iloc[va].to_numpy()

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)

        model, _ = train_one(
            X_tr, y_tr, X_va, y_va,
            device=device,
            cfg=cfg,
            model_hparams=model_hparams,
            seed=seed + fold,
            fold=fold
        )

        model.eval()
        with torch.no_grad():
            X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
            preds = model(X_va_t).squeeze(1).clamp(1.0, 5.0).cpu().numpy()
        fold_rmse = rmse_np(y_va, preds)
        rmses.append(fold_rmse)
        print(f"Fold {fold}: RMSE={fold_rmse:.4f}")

        #save the best model
        if save_best_to is not None and fold_rmse < best_overall:
            best_overall = fold_rmse
            os.makedirs(os.path.dirname(save_best_to) or ".", exist_ok=True)
            bundle = {
                "model_class": "RatingMLP",
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "in_dim": X.shape[1],
                "hparams": model_hparams or {},
                "best_val_rmse": float(fold_rmse),
                "fold": int(fold),
                "scaler": scaler,
                "feature_columns": list(feature_cols),
            }
            with open(save_best_to, "wb") as f:
                pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"New best model saved (fold {fold}, RMSE={fold_rmse:.4f})")

    return CVResult(rmses, float(np.mean(rmses)), float(np.std(rmses)))


#ensemble training
@dataclass
class FittedEnsemble:
    scalers: List[StandardScaler]
    models: List[nn.Module]
    device: torch.device
    in_dim: int

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_np = X.to_numpy().astype(np.float32)
        preds_all = []
        for scaler, model in zip(self.scalers, self.models):
            Xs = scaler.transform(X_np)
            with torch.no_grad():
                t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
                p = model(t).squeeze(1).clamp(1.0, 5.0).cpu().numpy()
            preds_all.append(p)
        return np.mean(np.stack(preds_all, axis=0), axis=0)


def fit_final_ensemble(
    X: pd.DataFrame, y: pd.Series,
    n_models: int = 5,
    device: Optional[torch.device] = None,
    cfg: TrainConfig = TrainConfig(),
    model_hparams: Dict = None,
    seed: int = 42
) -> FittedEnsemble:
    device = device or get_device("auto")

    scalers, models = [], []
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    for m in range(n_models):
        n = len(X_np)
        idx = np.random.RandomState(seed + 1000 + m).choice(n, n, replace=True)
        X_tr, y_tr = X_np[idx], y_np[idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)

        #small validation split from bootstrap sample
        val_size = max(1, len(X_tr) // 10)
        X_val, y_val = X_tr[:val_size], y_tr[:val_size]
        X_tr, y_tr = X_tr[val_size:], y_tr[val_size:]

        model, _ = train_one(
            X_tr, y_tr, X_val, y_val,
            device=device,
            cfg=cfg,
            model_hparams=model_hparams,
            seed=seed + 2000 + m
        )

        scalers.append(scaler)
        models.append(model)

    return FittedEnsemble(scalers, models, device, X.shape[1])


#the main result returned by training
@dataclass
class FitResult:
    cv: CVResult
    ensemble: FittedEnsemble
    feature_columns: List[str]


def fit_model(
    csv_path: str = "rated_songs_with_features.csv",
    device_pref: str = "auto",
    n_folds: int = 10,
    n_models: int = 12,
    save_best_to: Optional[str] = "models/best_cv_model.pkl"
) -> FitResult:
    device = get_device(device_pref)
    df = load_dataset(csv_path)
    X_df, y = engineer_features(df)

    #model configuration
    model_hparams = dict(width=1344, depth=11, p=0.0)

    cfg = TrainConfig(
        epochs=600,
        batch_size=112,
        lr=2.8e-4,
        weight_decay=1.5e-6,
        patience=50,
        min_delta=1e-7,
        grad_clip=0.25,
        use_amp=True
    )

    print("Cross-validating...")
    cv_res = kfold_cv_torch(
        X_df, y, n_splits=n_folds, device=device, cfg=cfg, model_hparams=model_hparams, seed=42,
        save_best_to=save_best_to
    )
    print(f"\nCV RMSE: {cv_res.mean_rmse:.4f} ± {cv_res.std_rmse:.4f}")

    print("\nTraining final ensemble...")
    ensemble = fit_final_ensemble(
        X_df, y, n_models=n_models, device=device, cfg=cfg, model_hparams=model_hparams, seed=1337
    )

    return FitResult(cv=cv_res, ensemble=ensemble, feature_columns=list(X_df.columns))


#model loading
def load_model_from_pickle(pkl_path: str) -> Tuple[nn.Module, Dict]:
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    in_dim = bundle["in_dim"]
    hparams = bundle.get("hparams", {})
    state_dict = bundle["state_dict"]
    model = RatingMLP(in_dim, **hparams)
    model.load_state_dict(state_dict)
    model.eval()
    return model, bundle


def main(): #main function to run the training
    res = fit_model(
        csv_path="rated_songs_with_features.csv",
        device_pref="auto",
        n_folds=10,
        n_models=12,
        save_best_to="models/best_cv_model.pkl"
    )

    print("\nTraining complete!")
    print("Best CV fold RMSEs:", [f"{r:.4f}" for r in res.cv.fold_rmse])
    print(f"Mean ± Std RMSE: {res.cv.mean_rmse:.4f} ± {res.cv.std_rmse:.4f}")
    print("Best model saved to: models/best_cv_model.pkl")

    #quick preview
    df = load_dataset("rated_songs_with_features.csv")
    X_df, y = engineer_features(df)
    preds = res.ensemble.predict(X_df)

    preview = pd.DataFrame({
        "track_name": df.get("track_name", ["Unknown"] * len(df)),
        "track_artist": df.get("track_artist", ["Unknown"] * len(df)),
        "actual_rating": y.values,
        "predicted_rating": np.round(preds, 2)
    })
    print("\nPredictions preview:")
    print(preview.head(10).to_string(index=False))


if __name__ == "__main__":
    main()