"""
ICT ML Model Trainer - Phase 3: AI/Machine Learning Module
============================================================

This module provides comprehensive machine learning training infrastructure
for ICT trading signals. It trains models on historical ICT signals to 
predict signal quality and optimal trade execution.

CORE CAPABILITIES:
1. Historical signal dataset management
2. Cross-validation training pipelines
3. Model persistence and versioning
4. Performance tracking and comparison
5. Automated hyperparameter tuning
6. Ensemble model training

ICT FEATURE CATEGORIES LEARNED:
- Order Block success rates by type and location
- FVG fill rates and timing patterns
- Liquidity sweep effectiveness
- Time-of-day performance patterns
- Confluence factor combinations
- Multi-timeframe alignment patterns

Author: ICT AI Engine
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable, Union
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import pickle
import logging
import os
from collections import defaultdict
from abc import ABC, abstractmethod
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Attention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Install with: pip install tensorflow")


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ModelType(Enum):
    """Types of ML models supported"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOST = "gradient_boost"
    LOGISTIC_REGRESSION = "logistic"
    SVM = "svm"
    KNN = "knn"
    DECISION_TREE = "decision_tree"
    ADABOOST = "adaboost"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"


class TrainingMode(Enum):
    """Training mode configurations"""
    QUICK = "quick"           # Fast training, fewer iterations
    STANDARD = "standard"     # Balanced training
    THOROUGH = "thorough"     # Extensive training, hyperparameter search
    PRODUCTION = "production" # Full training for deployment


class TargetVariable(Enum):
    """What the model predicts"""
    SIGNAL_QUALITY = "signal_quality"    # High/Medium/Low probability
    WIN_LOSS = "win_loss"                # Binary win/loss
    PROFIT_CATEGORY = "profit_category"  # Profit range buckets
    OPTIMAL_ENTRY = "optimal_entry"      # Entry timing quality
    OPTIMAL_EXIT = "optimal_exit"        # Exit timing quality


class FeatureSet(Enum):
    """Feature set categories"""
    ICT_CORE = "ict_core"           # Core ICT elements (OB, FVG, liquidity)
    CONFLUENCE = "confluence"        # Confluence factors
    TIMING = "timing"               # Time-based features
    STRUCTURE = "structure"         # Market structure features
    FULL = "full"                   # All features combined


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    model_type: ModelType = ModelType.RANDOM_FOREST
    training_mode: TrainingMode = TrainingMode.STANDARD
    target_variable: TargetVariable = TargetVariable.SIGNAL_QUALITY
    feature_set: FeatureSet = FeatureSet.FULL
    
    # Data splits
    test_size: float = 0.2
    validation_size: float = 0.1
    
    # Training parameters
    n_estimators: int = 100           # For ensemble methods
    max_depth: Optional[int] = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    learning_rate: float = 0.01
    
    # Cross-validation
    cv_folds: int = 5
    
    # Early stopping
    early_stopping_rounds: int = 50
    
    # Hyperparameter tuning
    tune_hyperparameters: bool = False
    n_trials: int = 50
    
    # Class balancing
    balance_classes: bool = True
    class_weight: Optional[Dict] = None
    
    # Feature selection
    feature_importance_threshold: float = 0.01
    max_features: Optional[int] = None
    
    # Model saving
    save_model: bool = True
    model_path: str = "./models"
    model_version: str = "v1.0"


@dataclass
class TrainingSample:
    """Single training sample with features and target"""
    sample_id: str
    timestamp: datetime
    
    # ICT Features
    ict_features: Dict[str, Any]
    
    # Target variables
    signal_quality: str           # 'high', 'medium', 'low'
    outcome: str                  # 'win', 'loss', 'breakeven'
    profit_pips: float            # Actual profit in pips
    profit_percent: float         # Profit as percentage
    
    # Timing metrics
    entry_timing_score: float     # How good was entry timing (0-100)
    exit_timing_score: float      # How good was exit timing (0-100)
    time_in_trade: int            # Minutes in trade
    
    # Risk metrics
    max_adverse_excursion: float  # Worst drawdown
    max_favorable_excursion: float # Best profit reached
    risk_reward_actual: float     # Actual R:R achieved
    
    # Context
    market_session: str           # London, NY, Asian
    day_of_week: int
    ict_model_used: str           # Model 2022, Silver Bullet, etc.
    
    # Metadata
    symbol: str = "EURUSD"
    timeframe: str = "15m"


@dataclass
class TrainingResult:
    """Results from model training"""
    model_type: ModelType
    training_mode: TrainingMode
    target_variable: TargetVariable
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    
    # Cross-validation scores
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    
    # Confusion matrix
    confusion_matrix: np.ndarray
    classification_report: str
    
    # Feature importance
    feature_importance: Dict[str, float]
    top_features: List[str]
    
    # Training details
    training_time: float          # Seconds
    n_samples: int
    n_features: int
    
    # Model metadata
    model_version: str
    trained_at: datetime
    model_path: Optional[str]


@dataclass
class ModelVersion:
    """Model version tracking"""
    version: str
    model_type: ModelType
    target_variable: TargetVariable
    
    # Performance
    accuracy: float
    f1_score: float
    
    # Metadata
    trained_at: datetime
    training_samples: int
    features_used: List[str]
    
    # File info
    model_path: str
    scaler_path: Optional[str]
    encoder_path: Optional[str]


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class ICTFeatureEngineer:
    """
    Extracts and engineers features from ICT trading data.
    
    FEATURE CATEGORIES:
    1. Order Block Features - OB type, location, age, touch count
    2. FVG Features - Fill rate, size, location, type
    3. Liquidity Features - Pool size, sweep type, distance
    4. Structure Features - Trend, break type, swing positions
    5. Confluence Features - Factor count, alignment scores
    6. Timing Features - Session, kill zone, macro time
    7. Multi-timeframe Features - HTF/LTF alignment
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_names = []
        self.categorical_features = [
            'session', 'day_of_week', 'ict_model', 'trend', 
            'structure_break_type', 'ob_type', 'fvg_type',
            'current_zone', 'htf_bias', 'ltf_bias'
        ]
        self.numeric_features = []
        
    def extract_features(
        self, 
        sample: Union[TrainingSample, Dict[str, Any]],
        include_categories: List[FeatureSet] = None
    ) -> Dict[str, Any]:
        """
        Extract all features from a training sample.
        
        Args:
            sample: Training sample or raw feature dict
            include_categories: Which feature categories to include
            
        Returns:
            Dictionary of extracted features
        """
        if include_categories is None:
            include_categories = [FeatureSet.FULL]
            
        features = {}
        
        if isinstance(sample, TrainingSample):
            ict_data = sample.ict_features
            context = {
                'session': sample.market_session,
                'day_of_week': sample.day_of_week,
                'ict_model': sample.ict_model_used,
                'symbol': sample.symbol,
                'timeframe': sample.timeframe
            }
        else:
            ict_data = sample
            context = sample.get('context', {})
            
        # Extract by category
        if FeatureSet.FULL in include_categories or FeatureSet.ICT_CORE in include_categories:
            features.update(self._extract_order_block_features(ict_data))
            features.update(self._extract_fvg_features(ict_data))
            features.update(self._extract_liquidity_features(ict_data))
            
        if FeatureSet.FULL in include_categories or FeatureSet.STRUCTURE in include_categories:
            features.update(self._extract_structure_features(ict_data))
            
        if FeatureSet.FULL in include_categories or FeatureSet.CONFLUENCE in include_categories:
            features.update(self._extract_confluence_features(ict_data))
            
        if FeatureSet.FULL in include_categories or FeatureSet.TIMING in include_categories:
            features.update(self._extract_timing_features(ict_data, context))
            
        return features
    
    def _extract_order_block_features(self, data: Dict) -> Dict[str, Any]:
        """Extract Order Block features"""
        features = {}
        
        # Check for OB data
        ob_data = data.get('order_block', data.get('ob', {}))
        
        # OB presence
        features['has_bullish_ob'] = 1 if ob_data.get('bullish_ob') else 0
        features['has_bearish_ob'] = 1 if ob_data.get('bearish_ob') else 0
        features['ob_count'] = ob_data.get('ob_count', 0)
        
        # Nearest OB metrics
        nearest_ob = ob_data.get('nearest_ob', {})
        features['ob_distance_pips'] = nearest_ob.get('distance', 999)
        features['ob_strength'] = nearest_ob.get('strength', 0)
        features['ob_touch_count'] = nearest_ob.get('touch_count', 0)
        features['ob_age_bars'] = nearest_ob.get('age_bars', 999)
        features['ob_breaker'] = 1 if nearest_ob.get('is_breaker', False) else 0
        features['ob_mitigation_block'] = 1 if nearest_ob.get('is_mitigation', False) else 0
        
        # OB type encoding
        features['ob_type'] = nearest_ob.get('type', 'none')
        
        # OB zone metrics
        features['ob_in_premium'] = 1 if ob_data.get('in_premium_zone', False) else 0
        features['ob_in_discount'] = 1 if ob_data.get('in_discount_zone', False) else 0
        
        # OB historical performance (if available)
        features['ob_historical_success_rate'] = ob_data.get('historical_success_rate', 0.5)
        
        return features
    
    def _extract_fvg_features(self, data: Dict) -> Dict[str, Any]:
        """Extract Fair Value Gap features"""
        features = {}
        
        fvg_data = data.get('fvg', data.get('fair_value_gap', {}))
        
        # FVG presence
        features['has_bullish_fvg'] = 1 if fvg_data.get('bullish_fvg') else 0
        features['has_bearish_fvg'] = 1 if fvg_data.get('bearish_fvg') else 0
        features['fvg_count'] = fvg_data.get('fvg_count', 0)
        
        # Nearest FVG metrics
        nearest_fvg = fvg_data.get('nearest_fvg', {})
        features['fvg_distance_pips'] = nearest_fvg.get('distance', 999)
        features['fvg_size_pips'] = nearest_fvg.get('size', 0)
        features['fvg_fill_percent'] = nearest_fvg.get('fill_percent', 0)
        features['fvg_age_bars'] = nearest_fvg.get('age_bars', 999)
        
        # FVG type
        features['fvg_type'] = nearest_fvg.get('type', 'none')
        features['fvg_ce_touched'] = 1 if nearest_fvg.get('ce_touched', False) else 0
        features['fvg_fully_filled'] = 1 if nearest_fvg.get('fully_filled', False) else 0
        
        # IFVG (Inverse FVG)
        features['has_ifvg'] = 1 if fvg_data.get('has_ifvg', False) else 0
        
        # FVG stacking
        features['fvg_stacked_count'] = fvg_data.get('stacked_count', 0)
        
        # Historical fill rates
        features['fvg_historical_fill_rate'] = fvg_data.get('historical_fill_rate', 0.65)
        features['fvg_avg_fill_time'] = fvg_data.get('avg_fill_time_bars', 10)
        
        return features
    
    def _extract_liquidity_features(self, data: Dict) -> Dict[str, Any]:
        """Extract Liquidity features"""
        features = {}
        
        liq_data = data.get('liquidity', {})
        
        # Liquidity pools
        features['has_buy_liquidity'] = 1 if liq_data.get('buy_side_pools') else 0
        features['has_sell_liquidity'] = 1 if liq_data.get('sell_side_pools') else 0
        features['total_liquidity_pools'] = liq_data.get('pool_count', 0)
        
        # Nearest liquidity
        nearest_liq = liq_data.get('nearest_pool', {})
        features['liquidity_distance_pips'] = nearest_liq.get('distance', 999)
        features['liquidity_size'] = nearest_liq.get('size', 0)
        features['liquidity_age_bars'] = nearest_liq.get('age_bars', 999)
        
        # Liquidity sweeps
        features['liquidity_swept'] = 1 if liq_data.get('recent_sweep', False) else 0
        features['sweep_type'] = liq_data.get('sweep_type', 'none')
        features['sweep_strength'] = liq_data.get('sweep_strength', 0)
        
        # Stop hunt detection
        features['stop_hunt_detected'] = 1 if liq_data.get('stop_hunt', False) else 0
        
        # Liquidity draw
        features['draw_on_liquidity_distance'] = liq_data.get('draw_distance', 999)
        features['clear_draw_present'] = 1 if liq_data.get('clear_draw', False) else 0
        
        # Equal highs/lows
        features['equal_highs'] = liq_data.get('equal_high_count', 0)
        features['equal_lows'] = liq_data.get('equal_low_count', 0)
        
        return features
    
    def _extract_structure_features(self, data: Dict) -> Dict[str, Any]:
        """Extract Market Structure features"""
        features = {}
        
        struct_data = data.get('structure', data.get('market_structure', {}))
        
        # Trend
        features['trend'] = struct_data.get('trend', 'ranging')
        features['trend_strength'] = struct_data.get('trend_strength', 0)
        
        # Structure breaks
        features['structure_break_type'] = struct_data.get('break_type', 'none')
        features['has_bos'] = 1 if struct_data.get('bos', False) else 0
        features['has_choch'] = 1 if struct_data.get('choch', False) else 0
        features['has_mss'] = 1 if struct_data.get('mss', False) else 0
        
        # Swing points
        features['swing_high_distance'] = struct_data.get('swing_high_distance', 999)
        features['swing_low_distance'] = struct_data.get('swing_low_distance', 999)
        
        # Premium/Discount zones
        features['current_zone'] = struct_data.get('zone', 'equilibrium')
        features['in_premium'] = 1 if struct_data.get('zone') == 'premium' else 0
        features['in_discount'] = 1 if struct_data.get('zone') == 'discount' else 0
        features['equilibrium_distance'] = struct_data.get('eq_distance', 0)
        
        # Displacement
        features['displacement_detected'] = 1 if struct_data.get('displacement', False) else 0
        features['displacement_strength'] = struct_data.get('displacement_strength', 0)
        
        # Range metrics
        features['adr_percent'] = struct_data.get('adr_percent', 100)
        features['range_position'] = struct_data.get('range_position', 50)
        
        return features
    
    def _extract_confluence_features(self, data: Dict) -> Dict[str, Any]:
        """Extract Confluence features"""
        features = {}
        
        conf_data = data.get('confluence', {})
        
        # Factor counts
        features['confluence_factor_count'] = conf_data.get('factor_count', 0)
        features['confluence_score'] = conf_data.get('score', 0)
        features['confluence_level'] = conf_data.get('level', 0)
        
        # Alignment scores
        features['structure_alignment'] = 1 if conf_data.get('structure_aligned', False) else 0
        features['pd_array_alignment'] = 1 if conf_data.get('pd_array_aligned', False) else 0
        features['time_alignment'] = 1 if conf_data.get('time_aligned', False) else 0
        features['liquidity_alignment'] = 1 if conf_data.get('liquidity_aligned', False) else 0
        
        # HTF/LTF bias alignment
        features['htf_bias'] = conf_data.get('htf_bias', 'neutral')
        features['ltf_bias'] = conf_data.get('ltf_bias', 'neutral')
        features['bias_aligned'] = 1 if conf_data.get('bias_aligned', False) else 0
        
        # Model-specific confluence
        features['model_2022_factors'] = conf_data.get('model_2022_count', 0)
        features['silver_bullet_factors'] = conf_data.get('silver_bullet_count', 0)
        
        # PD array stacking
        features['pd_arrays_stacked'] = conf_data.get('stacked_count', 0)
        
        return features
    
    def _extract_timing_features(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Extract Timing features"""
        features = {}
        
        timing_data = data.get('timing', {})
        
        # Session
        features['session'] = context.get('session', timing_data.get('session', 'unknown'))
        features['in_london'] = 1 if 'london' in features['session'].lower() else 0
        features['in_ny'] = 1 if 'ny' in features['session'].lower() or 'new_york' in features['session'].lower() else 0
        features['in_asian'] = 1 if 'asian' in features['session'].lower() else 0
        
        # Kill zones
        features['in_kill_zone'] = 1 if timing_data.get('kill_zone', False) else 0
        features['kill_zone_type'] = timing_data.get('kill_zone_type', 'none')
        
        # Day of week
        features['day_of_week'] = context.get('day_of_week', timing_data.get('day', 0))
        features['is_monday'] = 1 if features['day_of_week'] == 0 else 0
        features['is_friday'] = 1 if features['day_of_week'] == 4 else 0
        features['is_midweek'] = 1 if features['day_of_week'] in [1, 2, 3] else 0
        
        # Hour of day
        features['hour'] = timing_data.get('hour', 12)
        features['is_morning'] = 1 if 6 <= features['hour'] < 12 else 0
        features['is_afternoon'] = 1 if 12 <= features['hour'] < 18 else 0
        
        # Macro time
        features['in_macro_time'] = 1 if timing_data.get('macro_time', False) else 0
        
        # Time until events
        features['minutes_to_kill_zone_end'] = timing_data.get('minutes_to_kz_end', 999)
        features['minutes_from_session_open'] = timing_data.get('minutes_from_open', 0)
        
        # ICT model
        features['ict_model'] = context.get('ict_model', timing_data.get('model', 'unknown'))
        
        return features
    
    def prepare_training_data(
        self, 
        samples: List[TrainingSample],
        target_variable: TargetVariable = TargetVariable.SIGNAL_QUALITY,
        feature_set: FeatureSet = FeatureSet.FULL
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from samples.
        
        Args:
            samples: List of training samples
            target_variable: What to predict
            feature_set: Which features to include
            
        Returns:
            (X features, y targets, feature_names)
        """
        if not samples:
            raise ValueError("No samples provided")
            
        # Extract features from all samples
        feature_dicts = []
        targets = []
        
        for sample in samples:
            features = self.extract_features(sample, [feature_set])
            feature_dicts.append(features)
            
            # Get target
            if target_variable == TargetVariable.SIGNAL_QUALITY:
                targets.append(sample.signal_quality)
            elif target_variable == TargetVariable.WIN_LOSS:
                targets.append(sample.outcome)
            elif target_variable == TargetVariable.PROFIT_CATEGORY:
                targets.append(self._categorize_profit(sample.profit_percent))
            elif target_variable == TargetVariable.OPTIMAL_ENTRY:
                targets.append(self._categorize_score(sample.entry_timing_score))
            elif target_variable == TargetVariable.OPTIMAL_EXIT:
                targets.append(self._categorize_score(sample.exit_timing_score))
                
        # Convert to dataframe for easier processing
        df = pd.DataFrame(feature_dicts)
        
        # AGGRESSIVELY clean all string values
        # Replace all non-numeric strings with 0
        for col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.replace('none', '0').str.replace('unknown', '0').str.replace('', '0').str.replace('nan', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        # Ensure all values are numeric
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Encode categorical features - convert to sequential integers
        for col in self.categorical_features:
            if col in df.columns:
                unique_vals = sorted(df[col].unique())
                mapping = {val: i for i, val in enumerate(unique_vals)}
                df[col] = df[col].map(mapping).fillna(0)
        
        # Store feature names
        self.feature_names = list(df.columns)
        self.numeric_features = [f for f in self.feature_names if f not in self.categorical_features]
        
        # Convert to numpy
        X = df.values.astype(np.float32)
        
        # Encode targets
        if 'target_encoder' not in self.label_encoders:
            self.label_encoders['target_encoder'] = LabelEncoder()
            y = self.label_encoders['target_encoder'].fit_transform(targets)
        else:
            y = self.label_encoders['target_encoder'].transform(targets)
            
        return X, y, self.feature_names
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Scale numeric features"""
        if not SKLEARN_AVAILABLE or self.scaler is None:
            return X
            
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
            
    def _categorize_profit(self, profit_percent: float) -> str:
        """Categorize profit into buckets"""
        if profit_percent >= 2.0:
            return 'excellent'
        elif profit_percent >= 1.0:
            return 'good'
        elif profit_percent >= 0:
            return 'breakeven'
        elif profit_percent >= -1.0:
            return 'small_loss'
        else:
            return 'large_loss'
            
    def _categorize_score(self, score: float) -> str:
        """Categorize timing score into buckets"""
        if score >= 80:
            return 'excellent'
        elif score >= 60:
            return 'good'
        elif score >= 40:
            return 'average'
        else:
            return 'poor'
            
    def save(self, path: str):
        """Save feature engineer state"""
        state = {
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numeric_features': self.numeric_features
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
            
        if self.scaler is not None:
            scaler_path = path.replace('.pkl', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
    def load(self, path: str):
        """Load feature engineer state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        self.label_encoders = state['label_encoders']
        self.feature_names = state['feature_names']
        self.categorical_features = state['categorical_features']
        self.numeric_features = state.get('numeric_features', [])
        
        scaler_path = path.replace('.pkl', '_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)


# =============================================================================
# ML MODEL TRAINER
# =============================================================================

class ICTModelTrainer:
    """
    Comprehensive ML model trainer for ICT signals.
    
    TRAINING PIPELINE:
    1. Data preparation and validation
    2. Feature engineering and selection
    3. Model training with cross-validation
    4. Hyperparameter tuning (optional)
    5. Model evaluation and comparison
    6. Model persistence
    
    SUPPORTED MODELS:
    - Random Forest (primary)
    - Gradient Boosting
    - Logistic Regression
    - SVM
    - KNN
    - Decision Tree
    - AdaBoost
    - LSTM (for sequence prediction)
    - Ensemble (combination)
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.feature_engineer = ICTFeatureEngineer()
        self.models = {}
        self.training_history = []
        self.model_versions = []
        
    def train(
        self,
        samples: List[TrainingSample],
        config: Optional[TrainingConfig] = None
    ) -> TrainingResult:
        """
        Train a model on ICT signal samples.
        
        Args:
            samples: List of training samples
            config: Training configuration (optional)
            
        Returns:
            TrainingResult with performance metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for training")
            
        config = config or self.config
        start_time = datetime.now()
        
        logger.info(f"Starting training: {config.model_type.value} for {config.target_variable.value}")
        
        # Prepare data
        X, y, feature_names = self.feature_engineer.prepare_training_data(
            samples,
            config.target_variable,
            config.feature_set
        )
        
        # Scale features
        X_scaled = self.feature_engineer.scale_features(X, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=config.test_size,
            random_state=42,
            stratify=y if config.balance_classes else None
        )
        
        # Get model
        model = self._get_model(config)
        
        # Hyperparameter tuning if enabled
        if config.tune_hyperparameters:
            model = self._tune_hyperparameters(model, X_train, y_train, config)
            
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC (only for binary)
        try:
            if len(np.unique(y)) == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                y_proba = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        except:
            roc_auc = None
            
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=config.cv_folds)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred)
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, feature_names)
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save model
        model_path = None
        if config.save_model:
            model_path = self._save_model(model, config)
            
        # Store model
        self.models[config.model_type.value] = model
        
        # Create result
        result = TrainingResult(
            model_type=config.model_type,
            training_mode=config.training_mode,
            target_variable=config.target_variable,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            cv_scores=cv_scores.tolist(),
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            feature_importance=feature_importance,
            top_features=[f[0] for f in top_features],
            training_time=training_time,
            n_samples=len(samples),
            n_features=len(feature_names),
            model_version=config.model_version,
            trained_at=datetime.now(),
            model_path=model_path
        )
        
        self.training_history.append(result)
        
        logger.info(f"Training complete. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return result
    
    def train_ensemble(
        self,
        samples: List[TrainingSample],
        models: List[ModelType] = None,
        config: Optional[TrainingConfig] = None
    ) -> TrainingResult:
        """
        Train an ensemble of models.
        
        Args:
            samples: Training samples
            models: List of model types for ensemble
            config: Training configuration
            
        Returns:
            TrainingResult for ensemble
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")
            
        config = config or self.config
        models = models or [
            ModelType.RANDOM_FOREST,
            ModelType.GRADIENT_BOOST,
            ModelType.LOGISTIC_REGRESSION
        ]
        
        logger.info(f"Training ensemble with {len(models)} models")
        
        # Train individual models
        individual_results = []
        for model_type in models:
            model_config = TrainingConfig(
                model_type=model_type,
                training_mode=config.training_mode,
                target_variable=config.target_variable,
                feature_set=config.feature_set,
                save_model=False  # Don't save individual models
            )
            result = self.train(samples, model_config)
            individual_results.append(result)
            
        # Create voting ensemble
        from sklearn.ensemble import VotingClassifier
        
        estimators = []
        for model_type in models:
            if model_type.value in self.models:
                estimators.append((model_type.value, self.models[model_type.value]))
                
        if not estimators:
            raise ValueError("No models available for ensemble")
            
        # Prepare data again for ensemble training
        X, y, feature_names = self.feature_engineer.prepare_training_data(
            samples, config.target_variable, config.feature_set
        )
        X_scaled = self.feature_engineer.scale_features(X, fit=False)
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        # Cross-validation for ensemble
        cv_scores = cross_val_score(ensemble, X_scaled, y, cv=config.cv_folds)
        
        # Train ensemble on full data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=config.test_size, random_state=42
        )
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        self.models['ensemble'] = ensemble
        
        # Create combined feature importance
        combined_importance = defaultdict(float)
        for result in individual_results:
            for feat, imp in result.feature_importance.items():
                combined_importance[feat] += imp / len(individual_results)
                
        result = TrainingResult(
            model_type=ModelType.ENSEMBLE,
            training_mode=config.training_mode,
            target_variable=config.target_variable,
            accuracy=accuracy,
            precision=precision_score(y_test, y_pred, average='weighted', zero_division=0),
            recall=recall_score(y_test, y_pred, average='weighted', zero_division=0),
            f1_score=f1,
            roc_auc=None,
            cv_scores=cv_scores.tolist(),
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            confusion_matrix=confusion_matrix(y_test, y_pred),
            classification_report=classification_report(y_test, y_pred),
            feature_importance=dict(combined_importance),
            top_features=sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)[:10],
            training_time=sum(r.training_time for r in individual_results),
            n_samples=len(samples),
            n_features=len(feature_names),
            model_version=config.model_version,
            trained_at=datetime.now(),
            model_path=None
        )
        
        return result
    
    def _get_model(self, config: TrainingConfig):
        """Get sklearn model based on config"""
        class_weight = 'balanced' if config.balance_classes else None
        
        if config.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_samples_split=config.min_samples_split,
                min_samples_leaf=config.min_samples_leaf,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            )
        elif config.model_type == ModelType.GRADIENT_BOOST:
            return GradientBoostingClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                min_samples_split=config.min_samples_split,
                min_samples_leaf=config.min_samples_leaf,
                random_state=42
            )
        elif config.model_type == ModelType.LOGISTIC_REGRESSION:
            return LogisticRegression(
                class_weight=class_weight,
                max_iter=1000,
                random_state=42
            )
        elif config.model_type == ModelType.SVM:
            return SVC(
                kernel='rbf',
                class_weight=class_weight,
                probability=True,
                random_state=42
            )
        elif config.model_type == ModelType.KNN:
            return KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            )
        elif config.model_type == ModelType.DECISION_TREE:
            return DecisionTreeClassifier(
                max_depth=config.max_depth,
                min_samples_split=config.min_samples_split,
                class_weight=class_weight,
                random_state=42
            )
        elif config.model_type == ModelType.ADABOOST:
            return AdaBoostClassifier(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
            
    def _tune_hyperparameters(self, model, X, y, config: TrainingConfig):
        """Tune hyperparameters using GridSearchCV"""
        param_grid = self._get_param_grid(config.model_type)
        
        if not param_grid:
            return model
            
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=config.cv_folds,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _get_param_grid(self, model_type: ModelType) -> Dict:
        """Get parameter grid for hyperparameter tuning"""
        grids = {
            ModelType.RANDOM_FOREST: {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            ModelType.GRADIENT_BOOST: {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'min_samples_split': [2, 5]
            },
            ModelType.LOGISTIC_REGRESSION: {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            ModelType.SVM: {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        return grids.get(model_type, {})
    
    def _get_feature_importance(
        self, 
        model, 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Extract feature importance from model"""
        importance = {}
        
        if hasattr(model, 'feature_importances_'):
            for name, imp in zip(feature_names, model.feature_importances_):
                importance[name] = float(imp)
        elif hasattr(model, 'coef_'):
            coef = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            for name, imp in zip(feature_names, coef):
                importance[name] = float(imp)
        else:
            # Assign equal importance
            for name in feature_names:
                importance[name] = 1.0 / len(feature_names)
                
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
            
        return importance
    
    def _save_model(self, model, config: TrainingConfig) -> str:
        """Save trained model to disk"""
        os.makedirs(config.model_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.model_type.value}_{config.target_variable.value}_{timestamp}.pkl"
        filepath = os.path.join(config.model_path, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
            
        # Save feature engineer
        fe_path = os.path.join(config.model_path, f"feature_engineer_{timestamp}.pkl")
        self.feature_engineer.save(fe_path)
        
        # Track version
        version = ModelVersion(
            version=config.model_version,
            model_type=config.model_type,
            target_variable=config.target_variable,
            accuracy=0,  # Will be updated
            f1_score=0,
            trained_at=datetime.now(),
            training_samples=0,
            features_used=self.feature_engineer.feature_names,
            model_path=filepath,
            scaler_path=fe_path.replace('.pkl', '_scaler.pkl'),
            encoder_path=fe_path
        )
        self.model_versions.append(version)
        
        logger.info(f"Model saved to: {filepath}")
        return filepath
    
    def load_model(self, model_path: str, feature_engineer_path: str = None):
        """Load a trained model from disk"""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        if feature_engineer_path:
            self.feature_engineer.load(feature_engineer_path)
            
        # Determine model type from filename
        model_name = os.path.basename(model_path)
        for mt in ModelType:
            if mt.value in model_name:
                self.models[mt.value] = model
                break
        else:
            self.models['loaded'] = model
            
        return model
    
    def predict(
        self, 
        features: Dict[str, Any],
        model_type: ModelType = ModelType.RANDOM_FOREST
    ) -> Dict[str, Any]:
        """
        Make prediction for new sample.
        
        Args:
            features: ICT feature dictionary
            model_type: Which model to use
            
        Returns:
            Prediction with probabilities
        """
        model_key = model_type.value
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not trained")
            
        model = self.models[model_key]
        
        # Extract and prepare features
        extracted = self.feature_engineer.extract_features(features)
        
        # Clean extracted features - convert all to numeric
        for key in extracted:
            if isinstance(extracted[key], str):
                val = extracted[key].lower()
                if val in ['none', 'unknown', '', 'nan', 'true', 'false']:
                    extracted[key] = 0
                elif val == 'true':
                    extracted[key] = 1
                elif val == 'false':
                    extracted[key] = 0
                else:
                    try:
                        extracted[key] = float(val)
                    except:
                        extracted[key] = 0
        
        # Convert to array
        feature_values = [extracted.get(f, 0) for f in self.feature_engineer.feature_names]
        X = np.array([feature_values], dtype=np.float32)
        
        # Scale
        X_scaled = self.feature_engineer.scale_features(X, fit=False)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Decode prediction
        target_encoder = self.feature_engineer.label_encoders.get('target_encoder')
        if target_encoder:
            prediction_label = target_encoder.inverse_transform([prediction])[0]
            classes = target_encoder.classes_
        else:
            prediction_label = str(prediction)
            classes = ['class_' + str(i) for i in range(len(probabilities))]
            
        return {
            'prediction': prediction_label,
            'confidence': float(max(probabilities)),
            'probabilities': {c: float(p) for c, p in zip(classes, probabilities)}
        }
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
        if not self.training_history:
            return pd.DataFrame()
            
        data = []
        for result in self.training_history:
            data.append({
                'model': result.model_type.value,
                'target': result.target_variable.value,
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'cv_mean': result.cv_mean,
                'cv_std': result.cv_std,
                'training_time': result.training_time,
                'n_samples': result.n_samples
            })
            
        return pd.DataFrame(data).sort_values('f1_score', ascending=False)
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Get the best performing model"""
        if not self.training_history:
            return None, None
            
        best_result = max(self.training_history, key=lambda x: x.f1_score)
        model_key = best_result.model_type.value
        
        return model_key, self.models.get(model_key)


# =============================================================================
# DATA GENERATOR FOR TESTING
# =============================================================================

class SyntheticDataGenerator:
    """Generate synthetic ICT training data for testing"""
    
    @staticmethod
    def generate_samples(n_samples: int = 1000) -> List[TrainingSample]:
        """Generate synthetic training samples"""
        samples = []
        
        sessions = ['london', 'new_york', 'asian']
        models = ['model_2022', 'silver_bullet', 'optimal_trade_entry', 'standard']
        outcomes = ['win', 'loss', 'breakeven']
        qualities = ['high', 'medium', 'low']
        
        for i in range(n_samples):
            # Generate realistic ICT features
            has_ob = np.random.random() > 0.3
            has_fvg = np.random.random() > 0.4
            has_liquidity = np.random.random() > 0.35
            
            confluence_count = (
                (1 if has_ob else 0) +
                (1 if has_fvg else 0) +
                (1 if has_liquidity else 0) +
                np.random.randint(0, 4)
            )
            
            # Higher confluence = higher win rate
            base_win_rate = 0.3 + (confluence_count * 0.1)
            is_win = np.random.random() < base_win_rate
            
            if is_win:
                outcome = 'win'
                profit_pips = np.random.uniform(10, 100)
                profit_percent = np.random.uniform(0.5, 3.0)
                quality = np.random.choice(['high', 'medium'], p=[0.6, 0.4])
            elif np.random.random() < 0.2:
                outcome = 'breakeven'
                profit_pips = np.random.uniform(-5, 5)
                profit_percent = np.random.uniform(-0.2, 0.2)
                quality = 'medium'
            else:
                outcome = 'loss'
                profit_pips = np.random.uniform(-50, -5)
                profit_percent = np.random.uniform(-1.5, -0.3)
                quality = np.random.choice(['medium', 'low'], p=[0.3, 0.7])
                
            ict_features = {
                'order_block': {
                    'bullish_ob': has_ob and np.random.random() > 0.5,
                    'bearish_ob': has_ob and np.random.random() > 0.5,
                    'ob_count': np.random.randint(0, 4) if has_ob else 0,
                    'nearest_ob': {
                        'distance': np.random.uniform(5, 50) if has_ob else 999,
                        'strength': np.random.uniform(0.5, 1.0) if has_ob else 0,
                        'touch_count': np.random.randint(0, 3),
                        'age_bars': np.random.randint(5, 50),
                        'type': np.random.choice(['bullish', 'bearish', 'none'])
                    },
                    'historical_success_rate': np.random.uniform(0.4, 0.7)
                },
                'fvg': {
                    'bullish_fvg': has_fvg and np.random.random() > 0.5,
                    'bearish_fvg': has_fvg and np.random.random() > 0.5,
                    'fvg_count': np.random.randint(0, 5) if has_fvg else 0,
                    'nearest_fvg': {
                        'distance': np.random.uniform(3, 30) if has_fvg else 999,
                        'size': np.random.uniform(5, 25) if has_fvg else 0,
                        'fill_percent': np.random.uniform(0, 100),
                        'age_bars': np.random.randint(3, 40),
                        'type': np.random.choice(['bullish', 'bearish', 'none'])
                    },
                    'historical_fill_rate': np.random.uniform(0.5, 0.8)
                },
                'liquidity': {
                    'buy_side_pools': has_liquidity,
                    'sell_side_pools': has_liquidity,
                    'pool_count': np.random.randint(0, 3) if has_liquidity else 0,
                    'nearest_pool': {
                        'distance': np.random.uniform(10, 80) if has_liquidity else 999,
                        'size': np.random.uniform(100, 1000),
                        'age_bars': np.random.randint(10, 100)
                    },
                    'recent_sweep': np.random.random() > 0.6 if has_liquidity else False,
                    'sweep_strength': np.random.uniform(0, 1)
                },
                'structure': {
                    'trend': np.random.choice(['bullish', 'bearish', 'ranging']),
                    'trend_strength': np.random.uniform(0, 1),
                    'bos': np.random.random() > 0.7,
                    'choch': np.random.random() > 0.8,
                    'zone': np.random.choice(['premium', 'discount', 'equilibrium']),
                    'displacement': np.random.random() > 0.6
                },
                'confluence': {
                    'factor_count': confluence_count,
                    'score': confluence_count * 15,
                    'structure_aligned': np.random.random() > 0.4,
                    'pd_array_aligned': np.random.random() > 0.5,
                    'time_aligned': np.random.random() > 0.5,
                    'htf_bias': np.random.choice(['bullish', 'bearish', 'neutral']),
                    'ltf_bias': np.random.choice(['bullish', 'bearish', 'neutral']),
                    'bias_aligned': np.random.random() > 0.5
                },
                'timing': {
                    'kill_zone': np.random.random() > 0.4,
                    'hour': np.random.randint(0, 24),
                    'macro_time': np.random.random() > 0.7
                }
            }
            
            sample = TrainingSample(
                sample_id=f"sample_{i}",
                timestamp=datetime.now() - timedelta(days=np.random.randint(1, 365)),
                ict_features=ict_features,
                signal_quality=quality,
                outcome=outcome,
                profit_pips=profit_pips,
                profit_percent=profit_percent,
                entry_timing_score=np.random.uniform(30, 95),
                exit_timing_score=np.random.uniform(30, 95),
                time_in_trade=np.random.randint(15, 480),
                max_adverse_excursion=np.random.uniform(5, 50),
                max_favorable_excursion=np.random.uniform(10, 150),
                risk_reward_actual=profit_pips / 20 if profit_pips > 0 else profit_pips / 30,
                market_session=np.random.choice(sessions),
                day_of_week=np.random.randint(0, 5),
                ict_model_used=np.random.choice(models)
            )
            
            samples.append(sample)
            
        return samples


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ICT ML Model Trainer - Phase 3 AI/ML Module")
    print("=" * 60)
    
    if not SKLEARN_AVAILABLE:
        print("\n⚠️  scikit-learn not installed. Install with:")
        print("   pip install scikit-learn")
        print("\nShowing module structure only...")
    else:
        print("\n✓ scikit-learn available")
        
        # Generate synthetic data
        print("\n📊 Generating synthetic training data...")
        generator = SyntheticDataGenerator()
        samples = generator.generate_samples(n_samples=500)
        print(f"   Generated {len(samples)} samples")
        
        # Initialize trainer
        config = TrainingConfig(
            model_type=ModelType.RANDOM_FOREST,
            training_mode=TrainingMode.STANDARD,
            target_variable=TargetVariable.SIGNAL_QUALITY,
            tune_hyperparameters=False,
            save_model=False
        )
        
        trainer = ICTModelTrainer(config)
        
        # Train Random Forest
        print("\n🌲 Training Random Forest...")
        rf_result = trainer.train(samples, config)
        print(f"   Accuracy: {rf_result.accuracy:.4f}")
        print(f"   F1 Score: {rf_result.f1_score:.4f}")
        print(f"   CV Mean: {rf_result.cv_mean:.4f} (±{rf_result.cv_std:.4f})")
        
        # Train Gradient Boosting
        print("\n🚀 Training Gradient Boosting...")
        gb_config = TrainingConfig(
            model_type=ModelType.GRADIENT_BOOST,
            target_variable=TargetVariable.SIGNAL_QUALITY,
            save_model=False
        )
        gb_result = trainer.train(samples, gb_config)
        print(f"   Accuracy: {gb_result.accuracy:.4f}")
        print(f"   F1 Score: {gb_result.f1_score:.4f}")
        
        # Compare models
        print("\n📈 Model Comparison:")
        comparison = trainer.compare_models()
        print(comparison.to_string())
        
        # Show top features
        print("\n🔍 Top 10 Features (Random Forest):")
        for i, (feat, imp) in enumerate(sorted(rf_result.feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True)[:10]):
            print(f"   {i+1}. {feat}: {imp:.4f}")
        
        # Make prediction
        print("\n🎯 Sample Prediction:")
        test_features = samples[0].ict_features
        prediction = trainer.predict(test_features, ModelType.RANDOM_FOREST)
        print(f"   Prediction: {prediction['prediction']}")
        print(f"   Confidence: {prediction['confidence']:.2%}")
        
    print("\n" + "=" * 60)
    print("Module ready for integration")
    print("=" * 60)
