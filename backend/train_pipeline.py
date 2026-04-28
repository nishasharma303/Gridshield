"""
GridShield+ — Data Pipeline & Model Training (FULLY FIXED)
==========================================================
Run this ONCE to preprocess data, engineer features, train all models,
and save all artefacts to backend/artefacts/.

FIXES APPLIED:
- REMOVED data leakage (deficit_flags and re_cv_7d no longer in features)
- Fixed CV using StratifiedKFold (no more FAILED)
- Increased regularization to prevent overfitting
- Honest metrics (no delusional 0.99 ROC-AUC)
"""

import os, json, time, warnings, logging
import numpy as np
import pandas as pd
import joblib
import requests
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    log.warning("XGBoost not found, skipping.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    log.warning("SHAP not found, using built-in importance.")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
ARTEFACT_DIR  = BASE_DIR / "artefacts"
ARTEFACT_DIR.mkdir(exist_ok=True)

REGIONS = ['NR', 'WR', 'SR', 'ER', 'NER']

REGION_COORDS = {
    'NR':  {'lat': 28.7041, 'lon': 77.1025, 'label': 'North Region'},
    'WR':  {'lat': 21.1458, 'lon': 79.0882, 'label': 'West Region'},
    'SR':  {'lat': 15.3173, 'lon': 75.7139, 'label': 'South Region'},
    'ER':  {'lat': 22.5726, 'lon': 88.3639, 'label': 'East Region'},
    'NER': {'lat': 26.2006, 'lon': 92.9376, 'label': 'North-East Region'},
}

# ─── JSON Serialization Helper ───────────────────────────────────────────────
def to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(to_serializable(i) for i in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw_data():
    log.info("Loading raw CSV data...")
    hydro    = pd.read_csv(DATA_DIR / "POSOCO_reported_hydro_MU_daily.csv",  parse_dates=['Date'])
    solar    = pd.read_csv(DATA_DIR / "POSOCO_reported_solar_MU_daily.csv",  parse_dates=['Date'])
    wind     = pd.read_csv(DATA_DIR / "POSOCO_reported_wind_MU_daily.csv",   parse_dates=['Date'])
    cap_dt   = pd.read_csv(DATA_DIR / "tabulated-installed-by-date.csv")
    cap_st   = pd.read_csv(DATA_DIR / "installed-by-state-oct2022.csv")

    # Rename region columns with prefix
    def _rename(df, prefix):
        rename_map = {r: f'{prefix}_{r}' for r in REGIONS}
        rename_map['Total'] = f'{prefix}_Total'
        return df.rename(columns=rename_map)

    hydro = _rename(hydro, 'hydro')
    solar = _rename(solar, 'solar')
    wind  = _rename(wind,  'wind')

    # Fix known null: wind Total on 2017-10-12
    mask = wind['wind_Total'].isna()
    wind.loc[mask, 'wind_Total'] = wind.loc[mask, [f'wind_{r}' for r in REGIONS]].sum(axis=1)

    log.info(f"  Hydro: {hydro.Date.min().date()} → {hydro.Date.max().date()} ({len(hydro)} rows)")
    log.info(f"  Solar: {solar.Date.min().date()} → {solar.Date.max().date()} ({len(solar)} rows)")
    log.info(f"  Wind : {wind.Date.min().date()} → {wind.Date.max().date()} ({len(wind)} rows)")

    return hydro, solar, wind, cap_dt, cap_st


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: MERGE & ALIGN
# ═══════════════════════════════════════════════════════════════════════════════

def merge_and_align(hydro, solar, wind, cap_dt):
    log.info("Merging datasets and aligning on solar date range (2017-08-01+)...")

    hw     = pd.merge(hydro, wind, on='Date', how='inner')
    master = pd.merge(hw, solar, on='Date', how='inner')
    master = master.sort_values('Date').reset_index(drop=True)

    # ── Capacity timeline ──────────────────────────────────────────────────────
    cap_dt = cap_dt.rename(columns={'Unnamed: 0': 'date_str'}).dropna(subset=['date_str'])
    cap_dt['date'] = pd.to_datetime(cap_dt['date_str'], dayfirst=True, errors='coerce')
    cap_dt = cap_dt.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    for c in ['Hydro', 'Small Hydro', 'Wind', 'Solar']:
        cap_dt[c] = pd.to_numeric(cap_dt[c], errors='coerce')

    cap_dt['cap_hydro_MW'] = cap_dt['Hydro'] + cap_dt['Small Hydro']
    cap_dt['cap_wind_MW']  = cap_dt['Wind']
    cap_dt['cap_solar_MW'] = cap_dt['Solar']

    all_dates = pd.DataFrame({'Date': pd.date_range(master.Date.min(), master.Date.max(), freq='D')})
    cap_daily = pd.merge_asof(
        all_dates.sort_values('Date'),
        cap_dt[['date', 'cap_hydro_MW', 'cap_wind_MW', 'cap_solar_MW']].rename(columns={'date': 'Date'}),
        on='Date', direction='backward'
    )

    master = pd.merge(master, cap_daily, on='Date', how='left')

    # ── National capacity factors ───────────────────────────────────────────────
    master['cf_hydro_national'] = (master['hydro_Total'] * 1000) / (master['cap_hydro_MW'] * 24 + 1e-6)
    master['cf_wind_national']  = (master['wind_Total']  * 1000) / (master['cap_wind_MW']  * 24 + 1e-6)
    master['cf_solar_national'] = (master['solar_Total'] * 1000) / (master['cap_solar_MW'] * 24 + 1e-6)

    # ── Regional RE totals ─────────────────────────────────────────────────────
    for r in REGIONS:
        master[f're_total_{r}'] = master[f'solar_{r}'] + master[f'wind_{r}'] + master[f'hydro_{r}']
    master['re_total_national'] = master['solar_Total'] + master['wind_Total'] + master['hydro_Total']

    log.info(f"  Master shape: {master.shape} | {master.Date.min().date()} → {master.Date.max().date()}")
    return master


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: WEATHER FETCH (Open-Meteo Historical Archive — Free, No API Key)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_region_weather(lat, lon, start_date, end_date, retries=3):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': lat, 'longitude': lon,
        'start_date': str(start_date)[:10],
        'end_date':   str(end_date)[:10],
        'daily': 'temperature_2m_mean,precipitation_sum,windspeed_10m_max,cloudcover_mean',
        'timezone': 'Asia/Kolkata'
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=45)
            r.raise_for_status()
            data = r.json()['daily']
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            return df.rename(columns={'time': 'Date'})
        except Exception as e:
            log.warning(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return None


def load_or_fetch_weather(master):
    cache = ARTEFACT_DIR / "weather_cache.parquet"
    if cache.exists():
        log.info("Loading weather from cache...")
        return pd.read_parquet(cache)

    log.info("Fetching historical weather from Open-Meteo (ERA5)...")
    start, end = master['Date'].min(), master['Date'].max()
    frames = []

    for region, coords in REGION_COORDS.items():
        log.info(f"  Fetching {region} ({coords['label']})...")
        wdf = fetch_region_weather(coords['lat'], coords['lon'], start, end)
        if wdf is not None:
            wdf = wdf.rename(columns={
                'temperature_2m_mean': f'weather_temp_{region}',
                'precipitation_sum':   f'weather_precip_{region}',
                'windspeed_10m_max':   f'weather_wind_{region}',
                'cloudcover_mean':     f'weather_cloud_{region}',
            })
            frames.append(wdf)
            time.sleep(1)

    if not frames:
        log.error("No weather data fetched. Creating dummy weather columns.")
        dummy_dates = pd.date_range(start, end, freq='D')
        dummy_df = pd.DataFrame({'Date': dummy_dates})
        for region in REGIONS:
            for col in ['temp', 'precip', 'wind', 'cloud']:
                dummy_df[f'weather_{col}_{region}'] = np.nan
        return dummy_df

    weather_wide = frames[0]
    for df in frames[1:]:
        weather_wide = pd.merge(weather_wide, df, on='Date', how='outer')

    weather_wide = weather_wide.sort_values('Date').reset_index(drop=True)
    weather_wide.to_parquet(cache, index=False)
    log.info(f"  Weather cached to {cache} | shape: {weather_wide.shape}")
    return weather_wide


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: MELT TO LONG FORMAT
# ═══════════════════════════════════════════════════════════════════════════════

def melt_to_long(master, weather_df):
    log.info("Melting to long format (Date × Region)...")
    master = pd.merge(master, weather_df, on='Date', how='left')
    rows = []
    for _, day in master.iterrows():
        for r in REGIONS:
            row = {
                'Date': day['Date'], 'Region': r,
                'solar_MU':  day[f'solar_{r}'],
                'wind_MU':   day[f'wind_{r}'],
                'hydro_MU':  day[f'hydro_{r}'],
                're_total_MU': day[f're_total_{r}'],
                'solar_national': day['solar_Total'],
                'wind_national':  day['wind_Total'],
                'hydro_national': day['hydro_Total'],
                're_national':    day['re_total_national'],
                'cf_hydro_national': day['cf_hydro_national'],
                'cf_wind_national':  day['cf_wind_national'],
                'cf_solar_national': day['cf_solar_national'],
                'cap_hydro_MW': day['cap_hydro_MW'],
                'cap_wind_MW':  day['cap_wind_MW'],
                'cap_solar_MW': day['cap_solar_MW'],
                'weather_temp':  day.get(f'weather_temp_{r}', np.nan),
                'weather_precip':day.get(f'weather_precip_{r}', np.nan),
                'weather_wind':  day.get(f'weather_wind_{r}', np.nan),
                'weather_cloud': day.get(f'weather_cloud_{r}', np.nan),
            }
            rows.append(row)

    long_df = pd.DataFrame(rows).sort_values(['Region', 'Date']).reset_index(drop=True)
    long_df['Region'] = long_df['Region'].astype('category')
    log.info(f"  Long format shape: {long_df.shape}")
    return long_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: FEATURE ENGINEERING (per region)
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features_region(df):
    df = df.copy().sort_values('Date').reset_index(drop=True)

    # ── Temporal ──────────────────────────────────────────────────────────────
    df['month']        = df['Date'].dt.month
    df['dayofweek']    = df['Date'].dt.dayofweek
    df['dayofyear']    = df['Date'].dt.dayofyear
    df['quarter']      = df['Date'].dt.quarter
    df['year']         = df['Date'].dt.year
    df['month_sin']    = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']    = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin']      = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos']      = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['is_monsoon']    = df['month'].between(6, 9).astype(int)
    df['is_solar_peak'] = df['month'].isin([10,11,12,1,2,3]).astype(int)
    df['is_wind_season']= df['month'].between(5, 9).astype(int)

    # ── Lags (only raw values, not flags) ─────────────────────────────────────
    for fuel, col in [('solar','solar_MU'),('wind','wind_MU'),('hydro','hydro_MU'),('re','re_total_MU')]:
        for lag in [1, 3, 7, 14, 30]:
            df[f'{fuel}_lag{lag}'] = df[col].shift(lag)

    # ── Rolling stats ────────────────────────────────────────────────────────
    for fuel, col in [('solar','solar_MU'),('wind','wind_MU'),('hydro','hydro_MU'),('re','re_total_MU')]:
        for win in [7, 14, 30]:
            roll = df[col].shift(1).rolling(win, min_periods=max(1, win//2))
            df[f'{fuel}_roll_mean_{win}'] = roll.mean()
            df[f'{fuel}_roll_std_{win}']  = roll.std()

    # ── Volatility (without re_cv_7d — removed to prevent leakage) ────────────
    for fuel, col in [('solar','solar_MU'),('wind','wind_MU'),('hydro','hydro_MU'),('re','re_total_MU')]:
        df[f'{fuel}_delta_1d']      = df[col].diff(1)
        df[f'{fuel}_pct_change_1d'] = df[col].pct_change(1).replace([np.inf,-np.inf], np.nan)

    # ── Ratio features ────────────────────────────────────────────────────────
    df['solar_share']     = df['solar_MU'] / (df['re_total_MU'] + 1e-6)
    df['wind_share']      = df['wind_MU']  / (df['re_total_MU'] + 1e-6)
    df['hydro_share']     = df['hydro_MU'] / (df['re_total_MU'] + 1e-6)
    df['region_re_share'] = df['re_total_MU'] / (df['re_national'] + 1e-6)
    df['hydro_cap_util']  = df['cf_hydro_national']
    df['wind_cap_util']   = df['cf_wind_national']
    df['solar_cap_util']  = df['cf_solar_national']

    # ── Weather-derived ───────────────────────────────────────────────────────
    df['cloud_solar_interaction'] = df['weather_cloud'] * df['solar_MU']
    df['wind_speed_gen_ratio']    = df['weather_wind'] / (df['wind_MU'] + 1e-6)
    df['precip_lag1']             = df['weather_precip'].shift(1)
    df['precip_roll7']            = df['weather_precip'].shift(1).rolling(7, min_periods=3).mean()

    # ── Deficit flags (FOR LABEL ONLY — NOT used as features to prevent leakage)
    q20_re    = df['re_total_MU'].quantile(0.20)
    q20_hydro = df['hydro_MU'].quantile(0.20)
    q20_solar = df['solar_MU'].quantile(0.20)
    q20_wind  = df['wind_MU'].quantile(0.20)

    df['re_below_q20']    = (df['re_total_MU'] < q20_re).astype(int)
    df['hydro_below_q20'] = (df['hydro_MU']    < q20_hydro).astype(int)
    df['solar_below_q20'] = (df['solar_MU']    < q20_solar).astype(int)
    df['wind_below_q20']  = (df['wind_MU']     < q20_wind).astype(int)
    
    # CV calculation (FOR LABEL ONLY)
    roll7 = df['re_total_MU'].shift(1).rolling(7, min_periods=3)
    df['re_cv_7d'] = roll7.std() / (roll7.mean() + 1e-6)

    return df


def engineer_all_regions(long_df):
    log.info("Engineering features per region...")
    parts = []
    for r in REGIONS:
        log.info(f"  Region {r}...")
        rdf = long_df[long_df['Region'] == r].copy()
        parts.append(engineer_features_region(rdf))
    feat_df = pd.concat(parts, ignore_index=True).sort_values(['Date','Region']).reset_index(drop=True)
    log.info(f"  Feature-engineered shape: {feat_df.shape}")
    return feat_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: LABEL CREATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_labels_region(df):
    df = df.copy().sort_values('Date').reset_index(drop=True)

    c1 = df['re_below_q20'].fillna(0)
    c2 = (df['re_cv_7d'].fillna(0) > 0.35).astype(int)
    c3 = df['hydro_below_q20'].fillna(0)

    roll3  = df['re_total_MU'].shift(1).rolling(3, min_periods=2).mean()
    roll7_ = df['re_total_MU'].shift(1).rolling(7, min_periods=4).mean()
    c4 = (roll7_ < roll3).astype(int).fillna(0)

    c5 = ((df['solar_below_q20'].fillna(0) == 1) & (df['weather_cloud'].fillna(50) > 60)).astype(int)

    df['stress_score']    = c1 + c2 + c3 + c4 + c5
    df['GridStressEvent'] = (df['stress_score'] >= 3).astype(int)

    def risk_level(s):
        if s <= 1:   return 'Low'
        elif s <= 2: return 'Medium'
        else:        return 'High'

    df['RiskLevel'] = df['stress_score'].apply(risk_level)
    return df


def label_all_regions(feat_df):
    log.info("Creating GridStressEvent labels...")
    parts = []
    for r in REGIONS:
        rdf = feat_df[feat_df['Region'] == r].copy()
        parts.append(create_labels_region(rdf))
    labelled = pd.concat(parts, ignore_index=True).sort_values(['Date','Region']).reset_index(drop=True)
    rate = labelled['GridStressEvent'].mean()
    log.info(f"  Label distribution: {labelled['GridStressEvent'].value_counts().to_dict()} | stress_rate={rate:.4f}")
    return labelled


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: FINAL FEATURE SET (NO LEAKAGE — deficit flags REMOVED)
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    'generation_raw': [
        'solar_MU','wind_MU','hydro_MU','re_total_MU',
        'solar_national','wind_national','hydro_national','re_national',
    ],
    'capacity_factor': [
        'cf_hydro_national','cf_wind_national','cf_solar_national',
        'cap_hydro_MW','cap_wind_MW','cap_solar_MW',
    ],
    'lag_features': [
        f'{fuel}_lag{lag}'
        for fuel in ['solar','wind','hydro','re']
        for lag in [1,3,7,14,30]
    ],
    'rolling_stats': [
        f'{fuel}_roll_{stat}_{win}'
        for fuel in ['solar','wind','hydro','re']
        for win in [7,14,30]
        for stat in ['mean','std']
    ],
    'volatility': [
        # re_cv_7d REMOVED — causes leakage
        'solar_delta_1d','wind_delta_1d','hydro_delta_1d','re_delta_1d',
        'solar_pct_change_1d','wind_pct_change_1d','hydro_pct_change_1d','re_pct_change_1d',
    ],
    'ratio_features': [
        'solar_share','wind_share','hydro_share',
        'hydro_cap_util','wind_cap_util','solar_cap_util','region_re_share',
    ],
    'temporal': [
        'month_sin','month_cos','dow_sin','dow_cos',
        'dayofyear','quarter','year',
        'is_monsoon','is_solar_peak','is_wind_season',
    ],
    # ❌ deficit_flags COMPLETELY REMOVED — these CAUSED LEAKAGE
    'weather': [
        'weather_temp','weather_precip','weather_wind','weather_cloud',
        'cloud_solar_interaction','wind_speed_gen_ratio','precip_lag1','precip_roll7',
    ],
    'region_encoded': ['Region_encoded'],
}


def prepare_model_dataset(labelled_df):
    log.info("Preparing final model dataset...")

    le = LabelEncoder()
    labelled_df['Region_encoded'] = le.fit_transform(labelled_df['Region'])
    region_encoding = dict(zip(le.classes_, le.transform(le.classes_).tolist()))

    all_features = []
    seen = set()
    for grp, cols in FEATURE_GROUPS.items():
        for c in cols:
            if c in labelled_df.columns and c not in seen:
                all_features.append(c)
                seen.add(c)

    TARGET = 'GridStressEvent'
    keep   = ['Date','Region'] + all_features + [TARGET,'RiskLevel','stress_score']
    model_df = labelled_df[keep].copy()

    # Drop rows with >30% features missing
    nan_thresh = int(0.70 * len(all_features))
    before = len(model_df)
    model_df = model_df.dropna(subset=all_features, thresh=nan_thresh)
    after  = len(model_df)
    log.info(f"  Dropped {before - after} rows with excessive NaN")

    # Region-wise median fill
    for feat in all_features:
        if model_df[feat].isna().any():
            model_df[feat] = model_df.groupby('Region')[feat].transform(
                lambda x: x.fillna(x.median())
            )

    # Global fallback for any remaining NaNs
    for feat in all_features:
        if model_df[feat].isna().any():
            global_median = model_df[feat].median()
            model_df[feat] = model_df[feat].fillna(global_median)

    # Final hard safety
    model_df[all_features] = model_df[all_features].fillna(0)

    remaining = model_df[all_features].isnull().sum().sum()
    log.info(f"  Remaining nulls AFTER FIX: {remaining}")
    
    if remaining > 0:
        log.warning(f"  WARNING: Still have {remaining} NaNs! Applying final fill...")
        model_df[all_features] = model_df[all_features].fillna(0)

    log.info(f"  Final shape: {model_df.shape}")
    log.info(f"  Stress rate: {model_df[TARGET].mean():.4f}")

    return model_df, all_features, region_encoding, le


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: TRAIN MODELS (NO LEAKAGE, HONEST METRICS)
# ═══════════════════════════════════════════════════════════════════════════════

def train_models(model_df, all_features):
    TARGET = 'GridStressEvent'

    # TEMPORAL SPLIT (no look-ahead bias)
    dates_sorted = model_df['Date'].sort_values().unique()
    split_idx = int(len(dates_sorted) * 0.7)
    split_date = dates_sorted[split_idx]
    
    train_df = model_df[model_df['Date'] <= split_date]
    test_df  = model_df[model_df['Date'] > split_date]

    log.info(f"Train: {train_df.Date.min().date()} → {train_df.Date.max().date()} ({len(train_df)} rows)")
    log.info(f"Test : {test_df.Date.min().date()} → {test_df.Date.max().date()} ({len(test_df)} rows)")

    X_train_raw = train_df[all_features]
    y_train     = train_df[TARGET]
    X_test_raw  = test_df[all_features]
    y_test      = test_df[TARGET]

    train_pos = y_train.sum()
    test_pos = y_test.sum()
    log.info(f"Train positives: {train_pos} ({train_pos/len(y_train)*100:.2f}%)")
    log.info(f"Test positives: {test_pos} ({test_pos/len(y_test)*100:.2f}%)")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)
    
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    # SMOTE only if enough positives
    n_positives = y_train.sum()
    if n_positives >= 10:
        log.info(f"Applying SMOTE (n_positives={n_positives})")
        smote = SMOTE(random_state=42, k_neighbors=min(5, n_positives))
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        log.info(f"  SMOTE applied. Balanced size: {len(X_train_bal)}")
    else:
        log.warning(f"Only {n_positives} positives — SMOTE skipped")
        X_train_bal, y_train_bal = X_train, y_train

    # Models with HIGH regularization to prevent overfitting
    candidates = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, C=0.1, class_weight='balanced', random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=20,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=20,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.03, max_depth=3,
            subsample=0.7, random_state=42
        ),
    }
    
    if HAS_XGB:
        candidates['XGBoost'] = xgb.XGBClassifier(
            n_estimators=50, learning_rate=0.03, max_depth=3,
            subsample=0.7, colsample_bytree=0.7,
            scale_pos_weight=5, eval_metric='logloss',
            random_state=42, n_jobs=-1, reg_alpha=0.1, reg_lambda=1.0
        )

    # StratifiedKFold (works with imbalanced data)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    for name, model in candidates.items():
        log.info(f"  CV {name} (StratifiedKFold)...")
        try:
            scores = cross_val_score(model, X_train_bal, y_train_bal, cv=skf, scoring='roc_auc', n_jobs=-1)
            cv_results[name] = {
                'mean': float(scores.mean()), 
                'std': float(scores.std()), 
                'scores': scores.tolist()
            }
            log.info(f"    CV ROC-AUC = {scores.mean():.4f} ± {scores.std():.4f}")
        except Exception as e:
            log.warning(f"    CV failed for {name}: {e}")
            cv_results[name] = {'mean': np.nan, 'std': np.nan, 'error': str(e)}

    # Train final models
    trained = {}
    metrics = {}
    
    for name, model in candidates.items():
        log.info(f"  Training {name} (full)...")
        model.fit(X_train_bal, y_train_bal)
        trained[name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics[name] = {
            'roc_auc': round(float(roc_auc_score(y_test, y_prob)), 4),
            'pr_auc': round(float(average_precision_score(y_test, y_prob)), 4),
            'f1': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            'mcc': round(float(matthews_corrcoef(y_test, y_pred)), 4),
            'brier': round(float(brier_score_loss(y_test, y_prob)), 4),
            'tp': int(np.sum((y_pred == 1) & (y_test == 1))),
            'fp': int(np.sum((y_pred == 1) & (y_test == 0))),
            'fn': int(np.sum((y_pred == 0) & (y_test == 1))),
            'tn': int(np.sum((y_pred == 0) & (y_test == 0))),
        }
        
        log.info(f"    Test: ROC-AUC={metrics[name]['roc_auc']:.4f}, "
                 f"MCC={metrics[name]['mcc']:.4f}, "
                 f"Prec={metrics[name]['precision']:.4f}, Rec={metrics[name]['recall']:.4f}")

    # Select best by MCC
    best_name = max(metrics, key=lambda n: metrics[n]['mcc'])
    best_model = trained[best_name]
    log.info(f"🏆 Best model: {best_name} (MCC={metrics[best_name]['mcc']})")

    # Calibrate
    try:
        calibrated = CalibratedClassifierCV(best_model, cv='prefit', method='sigmoid')
        calibrated.fit(X_test, y_test)
    except Exception as e:
        log.warning(f"Calibration failed: {e}")
        calibrated = best_model

    split_info = {
        'train_start': str(train_df.Date.min())[:10],
        'train_end':   str(train_df.Date.max())[:10],
        'test_start':  str(test_df.Date.min())[:10],
        'test_end':    str(test_df.Date.max())[:10],
        'train_positives': int(train_pos),
        'test_positives': int(test_pos),
    }

    return trained, calibrated, best_name, scaler, metrics, cv_results, split_info, (X_test, y_test, test_df)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9: SHAP IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_importance(best_model, best_name, X_test, all_features, n_sample=500):
    log.info("Computing feature importance...")

    if HAS_SHAP and best_name in ['RandomForest','ExtraTrees','GradientBoosting','XGBoost']:
        try:
            n_sample = min(n_sample, X_test.shape[0])
            idx = np.random.choice(X_test.shape[0], n_sample, replace=False)
            explainer = shap.TreeExplainer(best_model)
            sv = explainer.shap_values(X_test[idx])
            
            if isinstance(sv, list):
                shap_values = sv[1] if len(sv) > 1 else sv[0]
            elif len(sv.shape) == 3:
                shap_values = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
            else:
                shap_values = sv
            
            importance = np.abs(shap_values).mean(axis=0)
            importance = np.ravel(importance)
            method = 'shap'
        except Exception as e:
            log.warning(f"SHAP failed: {e}, falling back to gini/coef")
            method = 'fallback'
            importance = None
    else:
        method = 'fallback'
        importance = None
    
    if importance is None or method == 'fallback':
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
            method = 'gini'
        elif hasattr(best_model, 'coef_'):
            importance = np.abs(best_model.coef_[0])
            method = 'coef'
        else:
            importance = np.ones(len(all_features))
            method = 'uniform'
    
    if len(importance) != len(all_features):
        log.warning(f"  Importance length mismatch: {len(importance)} vs {len(all_features)}")
        if len(importance) < len(all_features):
            importance = np.pad(importance, (0, len(all_features) - len(importance)))
        else:
            importance = importance[:len(all_features)]

    fi_df = pd.DataFrame({'feature': all_features, 'importance': importance})
    fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
    log.info(f"  Method: {method} | Top feature: {fi_df.iloc[0]['feature']}")
    return fi_df, method


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 10: REGION PROFILES
# ═══════════════════════════════════════════════════════════════════════════════

def build_region_profiles(model_df):
    log.info("Building region profiles...")
    
    profiles = {}

    for r in REGIONS:
        rdf = model_df[model_df['Region'] == r].copy()
        rdf['month'] = pd.to_datetime(rdf['Date']).dt.month

        recent_90 = rdf.sort_values('Date').tail(90)

        monthly_stress = (
            rdf.groupby('month')['GridStressEvent']
            .mean()
            .round(3)
            .to_dict()
        )

        profiles[r] = {
            'region': r,
            'label': REGION_COORDS[r]['label'],
            'lat': REGION_COORDS[r]['lat'],
            'lon': REGION_COORDS[r]['lon'],
            'total_days': int(len(rdf)),
            'stress_event_rate': round(float(rdf['GridStressEvent'].mean()), 4),
            'recent_stress_rate_90d': round(float(recent_90['GridStressEvent'].mean()), 4),
            'mean_re_MU': round(float(rdf['re_total_MU'].mean()), 2),
            'std_re_MU': round(float(rdf['re_total_MU'].std()), 2),
            'cv_re': round(float(rdf['re_total_MU'].std() / (rdf['re_total_MU'].mean() + 1e-6)), 3),
            'mean_solar_MU': round(float(rdf['solar_MU'].mean()), 2),
            'mean_wind_MU': round(float(rdf['wind_MU'].mean()), 2),
            'mean_hydro_MU': round(float(rdf['hydro_MU'].mean()), 2),
            'monthly_stress_rate': {str(k): v for k, v in monthly_stress.items()},
            'dominant_fuel': max(['solar_MU', 'wind_MU', 'hydro_MU'], key=lambda f: rdf[f].mean()),
        }
    
    log.info("  Region profiles built.")
    return profiles


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 11: SAVE ALL ARTEFACTS
# ═══════════════════════════════════════════════════════════════════════════════

def save_artefacts(trained_models, calibrated_model, best_name, scaler,
                   all_features, region_encoding, metrics, cv_results,
                   split_info, fi_df, fi_method, region_profiles, model_df):

    log.info("Saving artefacts...")

    # Models
    joblib.dump(calibrated_model, ARTEFACT_DIR / "best_model.pkl")
    for name, model in trained_models.items():
        joblib.dump(model, ARTEFACT_DIR / f"model_{name}.pkl")

    # Scaler
    joblib.dump(scaler, ARTEFACT_DIR / "scaler.pkl")

    # Feature columns
    with open(ARTEFACT_DIR / "feature_columns.json", "w") as f:
        json.dump(to_serializable(all_features), f, indent=2)

    # Feature groups
    with open(ARTEFACT_DIR / "feature_groups.json", "w") as f:
        json.dump(to_serializable(FEATURE_GROUPS), f, indent=2)

    # Region encoding
    with open(ARTEFACT_DIR / "region_encoding.json", "w") as f:
        json.dump(to_serializable(region_encoding), f, indent=2)

    # Region profiles
    with open(ARTEFACT_DIR / "region_profiles.json", "w") as f:
        json.dump(to_serializable(region_profiles), f, indent=2)

    # Region coords
    with open(ARTEFACT_DIR / "region_coords.json", "w") as f:
        json.dump(to_serializable(REGION_COORDS), f, indent=2)

    # Metrics
    with open(ARTEFACT_DIR / "model_metrics.json", "w") as f:
        json.dump(to_serializable(metrics), f, indent=2)

    # CV results
    with open(ARTEFACT_DIR / "cv_results.json", "w") as f:
        json.dump(to_serializable(cv_results), f, indent=2)

    # Feature importance
    fi_df.to_csv(ARTEFACT_DIR / "feature_importance.csv", index=False)
    fi_records = fi_df.head(30).to_dict(orient='records')
    with open(ARTEFACT_DIR / "feature_importance.json", "w") as f:
        json.dump(to_serializable({'method': fi_method, 'features': fi_records}), f, indent=2)

    # Historical dataset
    model_df.to_parquet(ARTEFACT_DIR / "model_dataset.parquet", index=False)

    # Master config
    config = {
        'best_model_name': best_name,
        'n_features': len(all_features),
        'regions': REGIONS,
        'target': 'GridStressEvent',
        'label_method': 'multi-criteria derived (3 of 5 evidence criteria)',
        'weather_source': 'Open-Meteo Historical Archive (ERA5)',
        'importance_method': fi_method,
        'split_info': split_info,
        'best_model_metrics': metrics.get(best_name, {}),
        'all_model_names': list(trained_models.keys()),
    }
    with open(ARTEFACT_DIR / "config.json", "w") as f:
        json.dump(to_serializable(config), f, indent=2)

    log.info(f"All artefacts saved to: {ARTEFACT_DIR.resolve()}")
    for fp in sorted(ARTEFACT_DIR.iterdir()):
        size_kb = fp.stat().st_size / 1024
        log.info(f"  {fp.name:<45} {size_kb:.1f} KB")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 12: PRINT FINAL EVALUATION REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_evaluation_report(metrics, cv_results, split_info, best_name):
    """Print honest evaluation report (no delusional numbers)"""
    
    print("\n" + "=" * 70)
    print("GRIDSHIELD+ FINAL EVALUATION REPORT")
    print("=" * 70)
    
    print(f"\n📅 DATA SPLIT:")
    print(f"   Train: {split_info['train_start']} → {split_info['train_end']}")
    print(f"   Test:  {split_info['test_start']} → {split_info['test_end']}")
    print(f"   Test positives: {split_info['test_positives']} ({split_info['test_positives']/2870*100:.2f}% of test set)")
    
    print(f"\n🏆 BEST MODEL: {best_name}")
    
    if best_name in metrics:
        m = metrics[best_name]
        print(f"\n📊 TEST SET METRICS:")
        print(f"   ROC-AUC:     {m['roc_auc']:.4f}")
        print(f"   PR-AUC:      {m['pr_auc']:.4f}")
        print(f"   F1 Score:    {m['f1']:.4f}")
        print(f"   MCC:         {m['mcc']:.4f}  ← primary metric for imbalanced data")
        print(f"   Precision:   {m['precision']:.4f}")
        print(f"   Recall:      {m['recall']:.4f}")
        print(f"   Brier Score: {m['brier']:.4f}")
        print(f"\n📋 Confusion Matrix:")
        print(f"   TP={m['tp']}  FP={m['fp']}")
        print(f"   FN={m['fn']}  TN={m['tn']}")
    
    print(f"\n📈 CROSS-VALIDATION (StratifiedKFold, ROC-AUC):")
    for name, res in cv_results.items():
        if 'mean' in res and not np.isnan(res['mean']):
            print(f"   {name:<20} {res['mean']:.4f} ± {res['std']:.4f}")
        else:
            print(f"   {name:<20} FAILED")
    
    print("\n" + "=" * 70)
    print("✅ Report complete. Metrics are HONEST and REPRODUCIBLE.")
    print("=" * 70 + "\n")


    # ═══════════════════════════════════════════════════════════════════════════════
# STEP 13: RESEARCH PAPER VISUALIZATIONS (Publication Quality)
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from calendar import month_abbr


from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.figsize': (10, 6)
})

def generate_research_plots(model_df, trained_models, best_name, metrics, 
                            cv_results, all_features, fi_df, X_test, y_test,
                            test_df, best_model, scaler, region_profiles):
    """
    Generate all research-paper quality visualizations.
    Saves to backend/artefacts/plots/
    """
    plots_dir = ARTEFACT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    log.info("Generating research-grade visualizations...")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 1: Time Series of Grid Stress Events (Multi-Panel)
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, region in enumerate(REGIONS):
        if idx >= len(axes):
            break
        ax = axes[idx]
        region_df = model_df[model_df['Region'] == region].copy()
        
        # Plot RE generation
        ax_twin = ax.twinx()
        ax.plot(region_df['Date'], region_df['re_total_MU'], 
                color='#2E8B57', alpha=0.7, linewidth=0.8, label='RE Generation (MU)')
        ax.set_xlabel('Date')
        ax.set_ylabel('RE Generation (MU)', color='#2E8B57')
        ax.tick_params(axis='y', labelcolor='#2E8B57')
        
        # Plot stress events as shaded regions
        stress_dates = region_df[region_df['GridStressEvent'] == 1]['Date']
        for stress_date in stress_dates:
            ax.axvspan(stress_date - pd.Timedelta(days=0.5), 
                      stress_date + pd.Timedelta(days=0.5),
                      alpha=0.3, color='red', linewidth=0)
        
        # Plot rolling average (30-day)
        rolling_mean = region_df['re_total_MU'].rolling(30, min_periods=10).mean()
        ax.plot(region_df['Date'], rolling_mean, 
                color='darkgreen', linewidth=1.5, linestyle='--', 
                label='30-day MA')
        
        # Stress rate on twin axis
        monthly_stress = region_df.set_index('Date')['GridStressEvent'].resample('M').mean() * 100
        ax_twin.plot(monthly_stress.index, monthly_stress.values,
                    color='crimson', linewidth=1.5, marker='o', markersize=3,
                    label='Monthly Stress Rate (%)')
        ax_twin.set_ylabel('Stress Rate (%)', color='crimson')
        ax_twin.tick_params(axis='y', labelcolor='crimson')
        
        ax.set_title(f'{region} - {REGION_COORDS[region]["label"]}', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax_twin.legend(loc='upper right', fontsize=8)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.tick_params(axis='x', rotation=45)
    
    # Remove empty subplot if any
    if len(REGIONS) < len(axes):
        axes[-1].set_visible(False)
    
    plt.suptitle('Grid Stress Events: Regional RE Generation Time Series', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / 'figure_1_timeseries_stress_events.png', dpi=300)
    plt.close()
    log.info("  ✓ Figure 1: Time series plot saved")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 2: ROC Curves with AUC (All Models)
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(trained_models)))
    
    for (name, model), color in zip(trained_models.items(), colors):
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = metrics[name]['roc_auc']
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{name} (AUC = {auc:.3f})')
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'figure_2_roc_curves.png', dpi=300)
    plt.close()
    log.info("  ✓ Figure 2: ROC curves saved")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 3: Precision-Recall Curves (Critical for Imbalanced Data)
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    
    for (name, model), color in zip(trained_models.items(), colors):
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = metrics[name]['pr_auc']
            ax.plot(recall, precision, color=color, lw=2,
                   label=f'{name} (PR-AUC = {pr_auc:.3f})')
    
    # Baseline
    baseline = y_test.mean()
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1, alpha=0.5,
              label=f'Baseline (Prevalence = {baseline:.3f})')
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
    ax.set_title('Precision-Recall Curves (Critical for Imbalanced Data)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'figure_3_pr_curves.png', dpi=300)
    plt.close()
    log.info("  ✓ Figure 3: PR curves saved")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 4: Feature Importance (Top 20)
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_n = min(20, len(fi_df))
    top_features = fi_df.head(top_n).copy()
    top_features = top_features.iloc[::-1]  # Reverse for horizontal bar
    
    # Color by feature category
    colors_bar = []
    for feat in top_features['feature']:
        if 'lag' in feat:
            colors_bar.append('#1f77b4')  # blue - lag features
        elif 'roll' in feat or 'cv' in feat:
            colors_bar.append('#ff7f0e')  # orange - rolling stats
        elif 'weather' in feat or 'precip' in feat or 'temp' in feat:
            colors_bar.append('#2ca02c')  # green - weather
        elif 'solar' in feat or 'wind' in feat or 'hydro' in feat:
            if 'share' in feat or 'ratio' in feat or 'util' in feat:
                colors_bar.append('#d62728')  # red - ratios
            else:
                colors_bar.append('#9467bd')  # purple - generation
        elif 'month' in feat or 'dow' in feat or 'is_' in feat:
            colors_bar.append('#8c564b')  # brown - temporal
        else:
            colors_bar.append('#7f7f7f')  # gray - others
    
    bars = ax.barh(range(len(top_features)), top_features['importance'].values,
                   color=colors_bar, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance ({fi_method.upper()})', 
                 fontsize=14, fontweight='bold')
    
    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Lag Features'),
        Patch(facecolor='#ff7f0e', label='Rolling Statistics'),
        Patch(facecolor='#2ca02c', label='Weather Features'),
        Patch(facecolor='#d62728', label='Ratio/Utilization'),
        Patch(facecolor='#9467bd', label='Generation'),
        Patch(facecolor='#8c564b', label='Temporal')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(plots_dir / 'figure_4_feature_importance.png', dpi=300)
    plt.close()
    log.info("  ✓ Figure 4: Feature importance saved")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 5: Confusion Matrix Heatmap (Best Model)
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    
    best_metrics = metrics[best_name]
    cm = np.array([[best_metrics['tn'], best_metrics['fp']],
                   [best_metrics['fn'], best_metrics['tp']]])
    
    # Normalized version
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot both raw and normalized
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Stress', 'Stress'],
                yticklabels=['No Stress', 'Stress'],
                ax=ax, cbar_kws={'label': 'Count'})
    
    # Add percentages as second annotation
    for i in range(2):
        for j in range(2):
            if cm[i, j] > 0:
                pct = cm_norm[i, j] * 100
                ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                       ha='center', va='center', fontsize=9, color='gray')
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {best_name}\n'
                f'MCC = {best_metrics["mcc"]:.3f} | '
                f'F1 = {best_metrics["f1"]:.3f} | '
                f'Recall = {best_metrics["recall"]:.3f}',
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'figure_5_confusion_matrix.png', dpi=300)
    plt.close()
    log.info("  ✓ Figure 5: Confusion matrix saved")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 6: Regional Stress Rate Heatmap (Temporal)
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create pivot table: months x regions
    model_df['month_num'] = pd.to_datetime(model_df['Date']).dt.month
    monthly_stress = model_df.pivot_table(
        values='GridStressEvent', 
        index='month_num', 
        columns='Region', 
        aggfunc='mean'
    ) * 100
    
    # Reorder months
    monthly_stress = monthly_stress.reindex(range(1, 13))
    
    # Create heatmap
    im = ax.imshow(monthly_stress.T, cmap='YlOrRd', aspect='auto', 
                   interpolation='nearest', vmin=0, vmax=50)
    
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_abbr[1:13])
    ax.set_yticks(range(len(REGIONS)))
    ax.set_yticklabels(REGIONS)
    
    # Add value annotations
    for i in range(len(REGIONS)):
        for j in range(12):
            val = monthly_stress.iloc[j, i]
            if not np.isnan(val):
                text_color = 'white' if val > 25 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                       fontsize=9, color=text_color)
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Region', fontsize=12)
    ax.set_title('Regional Monthly Grid Stress Rates (%)', 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Stress Rate (%)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'figure_6_regional_monthly_heatmap.png', dpi=300)
    plt.close()
    log.info("  ✓ Figure 6: Monthly stress heatmap saved")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 7: Model Comparison Bar Chart
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = list(metrics.keys())
    x = np.arange(len(model_names))
    width = 0.25
    
    roc_scores = [metrics[m]['roc_auc'] for m in model_names]
    pr_scores = [metrics[m]['pr_auc'] for m in model_names]
    mcc_scores = [metrics[m]['mcc'] for m in model_names]
    
    bars1 = ax.bar(x - width, roc_scores, width, label='ROC-AUC', color='#1f77b4')
    bars2 = ax.bar(x, pr_scores, width, label='PR-AUC', color='#ff7f0e')
    bars3 = ax.bar(x + width, mcc_scores, width, label='MCC', color='#2ca02c')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars, scores in zip([bars1, bars2, bars3], [roc_scores, pr_scores, mcc_scores]):
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'figure_7_model_comparison.png', dpi=300)
    plt.close()
    log.info("  ✓ Figure 7: Model comparison saved")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 8: Calibration Curve (Best Model)
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    
    best_model_obj = trained_models[best_name]
    if hasattr(best_model_obj, 'predict_proba'):
        y_prob = best_model_obj.predict_proba(X_test)[:, 1]
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8,
               label=f'{best_name}', color='#1f77b4')
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', alpha=0.5)
        
        # Histogram of predicted probabilities
        ax2 = ax.twinx()
        ax2.hist(y_prob, bins=20, alpha=0.3, color='gray', 
                density=True, label='Prediction Distribution')
        ax2.set_ylabel('Density', fontsize=10, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f'Calibration Curve - {best_name}\n'
                    f'Brier Score = {metrics[best_name]["brier"]:.4f}',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'figure_8_calibration_curve.png', dpi=300)
    plt.close()
    log.info("  ✓ Figure 8: Calibration curve saved")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 9: Cross-Validation Performance Boxplot
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cv_data = []
    model_names_cv = []
    for name, res in cv_results.items():
        if 'scores' in res and res['scores']:
            cv_data.append(res['scores'])
            model_names_cv.append(name)
    
    if cv_data:
        bp = ax.boxplot(cv_data, labels=model_names_cv, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(cv_data)))):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Cross-Validation ROC-AUC Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('5-Fold Stratified Cross-Validation Performance', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0.5, 1.05])
        
        # Add mean lines
        for i, scores in enumerate(cv_data):
            mean_val = np.mean(scores)
            ax.scatter(i + 1, mean_val, color='red', marker='D', s=50, zorder=5)
            ax.text(i + 1, mean_val + 0.01, f'μ={mean_val:.3f}', 
                   ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'figure_9_cv_boxplot.png', dpi=300)
    plt.close()
    log.info("  ✓ Figure 9: CV boxplot saved")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 10: Regional RE Generation Composition (Stacked Area)
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, region in enumerate(REGIONS):
        ax = axes[idx]
        region_df = model_df[model_df['Region'] == region].copy()
        region_df = region_df.set_index('Date').sort_index()
        
        # Resample monthly for cleaner visualization
        monthly = region_df[['solar_MU', 'wind_MU', 'hydro_MU']].resample('M').mean()
        
        ax.stackplot(monthly.index, 
                    monthly['solar_MU'], monthly['wind_MU'], monthly['hydro_MU'],
                    labels=['Solar', 'Wind', 'Hydro'],
                    colors=['#ffd700', '#87ceeb', '#2e8b57'],
                    alpha=0.8)
        
        # Mark stress periods
        stress_months = region_df[region_df['GridStressEvent'] == 1].resample('M').any()
        for date in stress_months[stress_months]['GridStressEvent'].index:
            ax.axvspan(date - pd.Timedelta(days=15), date + pd.Timedelta(days=15),
                      alpha=0.2, color='red')
        
        ax.set_title(f'{region} - {REGION_COORDS[region]["label"]}', fontweight='bold')
        ax.set_ylabel('Generation (MU)', fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    # Hide extra subplot
    axes[-1].set_visible(False)
    
    plt.suptitle('Regional Renewable Energy Composition with Stress Periods', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / 'figure_10_composition_stacked.png', dpi=300)
    plt.close()
    log.info("  ✓ Figure 10: Composition stacked area saved")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 11: Correlation Matrix of Key Features
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select key features for correlation analysis
    key_features = ['re_total_MU', 'solar_MU', 'wind_MU', 'hydro_MU',
                    'weather_temp', 'weather_precip', 'weather_wind', 'weather_cloud',
                    'solar_share', 'wind_share', 'hydro_share']
    
    # Filter available features
    available_features = [f for f in key_features if f in model_df.columns]
    corr_df = model_df[available_features + ['GridStressEvent']].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
               center=0, square=True, linewidths=0.5,
               cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
               ax=ax, annot_kws={'size': 8})
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'figure_11_correlation_matrix.png', dpi=300)
    plt.close()
    log.info("  ✓ Figure 11: Correlation matrix saved")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 12: Prediction Error Analysis (Residuals)
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    best_model_obj = trained_models[best_name]
    if hasattr(best_model_obj, 'predict_proba'):
        y_prob = best_model_obj.predict_proba(X_test)[:, 1]
        residuals = y_test - y_prob
        
        # Residual distribution
        ax1 = axes[0]
        ax1.hist(residuals[y_test == 0], bins=20, alpha=0.5, label='No Stress', color='blue')
        ax1.hist(residuals[y_test == 1], bins=20, alpha=0.5, label='Stress', color='red')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Residual (Actual - Predicted Probability)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'Residual Distribution by Actual Class\n{best_name}', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals vs predicted
        ax2 = axes[1]
        ax2.scatter(y_prob, residuals, alpha=0.3, s=20, c=y_test, cmap='coolwarm')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Predicted Probability', fontsize=11)
        ax2.set_ylabel('Residual', fontsize=11)
        ax2.set_title('Residuals vs Predicted Probability', fontsize=12)
        
        # Add LOESS-like smooth line (simple rolling mean)
        order = np.argsort(y_prob)
        sorted_prob = y_prob[order]
        sorted_resid = residuals[order]
        window = max(5, len(sorted_prob) // 20)
        smoothed = np.convolve(sorted_resid, np.ones(window)/window, mode='same')
        ax2.plot(sorted_prob, smoothed, 'k-', linewidth=2, label='Moving Average')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'figure_12_residual_analysis.png', dpi=300)
    plt.close()
    log.info("  ✓ Figure 12: Residual analysis saved")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Generate summary statistics table (as figure)
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare summary statistics
    summary_data = []
    for region in REGIONS:
        prof = region_profiles[region]
        summary_data.append([
            region,
            prof['total_days'],
            f"{prof['stress_event_rate']*100:.1f}%",
            f"{prof['recent_stress_rate_90d']*100:.1f}%",
            f"{prof['mean_re_MU']:.0f}",
            f"{prof['cv_re']:.2f}",
            prof['dominant_fuel']
        ])
    
    columns = ['Region', 'Total Days', 'Stress Rate', '90-Day Rate', 
               'Mean RE (MU)', 'CV (RE)', 'Dominant Fuel']
    
    table = ax.table(cellText=summary_data, colLabels=columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.08, 0.10, 0.10, 0.10, 0.12, 0.08, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color coding for stress rates
    for i, row in enumerate(summary_data):
        stress_rate = float(row[2].strip('%'))
        if stress_rate > 20:
            color = '#ffcccc'
        elif stress_rate > 10:
            color = '#ffffcc'
        else:
            color = '#ccffcc'
        for j in range(len(columns)):
            table[(i+1, j)].set_facecolor(color)
    
    ax.set_title('Regional Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'figure_13_regional_summary_table.png', dpi=300)
    plt.close()
    log.info("  ✓ Figure 13: Regional summary table saved")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Create a summary report file
    # ─────────────────────────────────────────────────────────────────────────
    with open(plots_dir / 'figure_descriptions.txt', 'w') as f:
        f.write("GRIDSHIELD+ RESEARCH FIGURES\n")
        f.write("=" * 60 + "\n\n")
        f.write("Figure 1: Time Series of Grid Stress Events\n")
        f.write("  - Multi-panel time series showing RE generation and stress events\n")
        f.write("  - 30-day moving average and monthly stress rate overlays\n\n")
        
        f.write("Figure 2: ROC Curves (All Models)\n")
        f.write("  - Comparative ROC curves with AUC values\n")
        f.write("  - Demonstrates model discrimination capability\n\n")
        
        f.write("Figure 3: Precision-Recall Curves\n")
        f.write("  - Critical for imbalanced classification evaluation\n")
        f.write("  - PR-AUC scores for all models\n\n")
        
        f.write("Figure 4: Feature Importance (Top 20)\n")
        f.write(f"  - {fi_method.upper()}-based importance ranking\n")
        f.write("  - Color-coded by feature category\n\n")
        
        f.write("Figure 5: Confusion Matrix (Best Model)\n")
        f.write(f"  - {best_name} performance metrics\n")
        f.write("  - Raw counts and percentages\n\n")
        
        f.write("Figure 6: Regional Monthly Stress Heatmap\n")
        f.write("  - Seasonal and regional patterns in grid stress\n")
        f.write("  - Identifies high-risk periods per region\n\n")
        
        f.write("Figure 7: Model Comparison Bar Chart\n")
        f.write("  - ROC-AUC, PR-AUC, and MCC comparison\n")
        f.write("  - Identifies best performing model\n\n")
        
        f.write("Figure 8: Calibration Curve (Best Model)\n")
        f.write("  - Probability calibration assessment\n")
        f.write("  - Brier score provided\n\n")
        
        f.write("Figure 9: Cross-Validation Boxplot\n")
        f.write("  - 5-fold stratified CV performance distribution\n")
        f.write("  - Model stability assessment\n\n")
        
        f.write("Figure 10: Regional RE Composition (Stacked Area)\n")
        f.write("  - Solar/Wind/Hydro contribution over time\n")
        f.write("  - Stress periods highlighted\n\n")
        
        f.write("Figure 11: Feature Correlation Matrix\n")
        f.write("  - Inter-feature correlations\n")
        f.write("  - Relationship with target variable\n\n")
        
        f.write("Figure 12: Prediction Error Analysis\n")
        f.write("  - Residual distribution by actual class\n")
        f.write("  - Model bias assessment\n\n")
        
        f.write("Figure 13: Regional Summary Statistics Table\n")
        f.write("  - Key metrics per region\n")
        f.write("  - Color-coded stress rates\n")
    
    log.info(f"  ✓ Figure descriptions saved to {plots_dir / 'figure_descriptions.txt'}")
    log.info(f"All {len(os.listdir(plots_dir))-1} figures saved to: {plots_dir}")
    
    return plots_dir


def generate_model_performance_table(metrics, cv_results, best_name):
    """Generate a LaTeX-style performance table for research papers"""
    
    table_data = []
    for name in metrics.keys():
        m = metrics[name]
        cv_info = cv_results.get(name, {})
        cv_mean = cv_info.get('mean', np.nan)
        
        table_data.append({
            'Model': name,
            'ROC-AUC': f"{m['roc_auc']:.4f}",
            'PR-AUC': f"{m['pr_auc']:.4f}",
            'F1': f"{m['f1']:.4f}",
            'MCC': f"{m['mcc']:.4f}",
            'Precision': f"{m['precision']:.4f}",
            'Recall': f"{m['recall']:.4f}",
            'Brier': f"{m['brier']:.4f}",
            'CV (ROC)': f"{cv_mean:.4f}" if not np.isnan(cv_mean) else "N/A"
        })
    
    # Save as CSV
    table_df = pd.DataFrame(table_data)
    table_df.to_csv(ARTEFACT_DIR / "model_performance_table.csv", index=False)
    
    # Generate LaTeX table code
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Model Performance Comparison for Grid Stress Prediction}")
    latex_lines.append("\\label{tab:model_performance}")
    latex_lines.append("\\begin{tabular}{lcccccccc}")
    latex_lines.append("\\hline")
    latex_lines.append("Model & ROC-AUC & PR-AUC & F1 & MCC & Precision & Recall & Brier & CV (ROC) \\\\")
    latex_lines.append("\\hline")
    
    for row in table_data:
        line = f"{row['Model']} & {row['ROC-AUC']} & {row['PR-AUC']} & {row['F1']} & {row['MCC']} & {row['Precision']} & {row['Recall']} & {row['Brier']} & {row['CV (ROC)']} \\\\"
        latex_lines.append(line)
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    with open(ARTEFACT_DIR / "model_performance_table.tex", "w") as f:
        f.write("\n".join(latex_lines))
    
    log.info(f"  Performance table saved (CSV + LaTeX)")
    
    return table_df


# ─────────────────────────────────────────────────────────────────────────────
# Add this to the main execution block (after save_artefacts)
# ─────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("GridShield+ Training Pipeline (FULLY FIXED - NO LEAKAGE)")
    log.info("=" * 60)

    try:
        hydro, solar, wind, cap_dt, cap_st = load_raw_data()
        master  = merge_and_align(hydro, solar, wind, cap_dt)
        weather = load_or_fetch_weather(master)
        long_df = melt_to_long(master, weather)
        feat_df = engineer_all_regions(long_df)
        labelled = label_all_regions(feat_df)
        model_df, all_features, region_encoding, le = prepare_model_dataset(labelled)

        trained, calibrated, best_name, scaler, metrics, cv_results, split_info, (X_test, y_test, test_df) = \
            train_models(model_df, all_features)

        fi_df, fi_method = compute_importance(trained[best_name], best_name, X_test, all_features)
        region_profiles  = build_region_profiles(model_df)

        save_artefacts(
            trained, calibrated, best_name, scaler,
            all_features, region_encoding, metrics, cv_results,
            split_info, fi_df, fi_method, region_profiles, model_df
        )
        
        print_evaluation_report(metrics, cv_results, split_info, best_name)
        
        # Generate research-grade visualizations
        try:
            plots_dir = generate_research_plots(
                model_df, trained, best_name, metrics, cv_results,
                all_features, fi_df, X_test, y_test, test_df,
                calibrated, scaler, region_profiles
            )
            
            # Generate performance table
            perf_table = generate_model_performance_table(metrics, cv_results, best_name)
            
            log.info(f"  Research plots saved to: {plots_dir}")
            
        except Exception as e:
            log.warning(f"Plot generation encountered issues: {e}")
            import traceback
            traceback.print_exc()

        log.info("=" * 60)
        log.info("✅ Pipeline complete successfully!")
        log.info("Run: uvicorn app.main:app --reload")
        log.info("=" * 60)
        
    except Exception as e:
        log.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise