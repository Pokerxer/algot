"""
ICT LSTM Trend Predictor - Phase 3: Deep Learning Module
=========================================================

This module implements LSTM (Long Short-Term Memory) neural networks
for price direction and trend prediction in ICT trading.

CORE CAPABILITIES:
1. Sequence-based price prediction
2. Multi-step ahead forecasting
3. Directional bias prediction
4. Trend strength estimation
5. Optimal entry timing prediction
6. Market regime classification

ICT-SPECIFIC APPLICATIONS:
- Predicting draw on liquidity direction
- Estimating time to liquidity sweep
- Forecasting displacement probability
- Predicting structure break direction
- Timing optimal entries within kill zones

Author: ICT AI Engine
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import pickle
import logging
import os
from collections import deque
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Input, Bidirectional, 
        Attention, Conv1D, MaxPooling1D, Flatten,
        BatchNormalization, GRU, Layer, Concatenate
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
        TensorBoard, Callback
    )
    from tensorflow.keras.regularizers import l1_l2
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Install with: pip install tensorflow")

try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# ENUMERATIONS
# =============================================================================

class LSTMArchitecture(Enum):
    """LSTM architecture variants"""
    SIMPLE = "simple"              # Basic LSTM
    STACKED = "stacked"            # Multiple LSTM layers
    BIDIRECTIONAL = "bidirectional" # Bidirectional LSTM
    CNN_LSTM = "cnn_lstm"          # CNN feature extraction + LSTM
    ATTENTION = "attention"         # LSTM with attention mechanism
    MULTI_HEAD = "multi_head"       # Multiple parallel LSTM heads


class PredictionTarget(Enum):
    """What the LSTM predicts"""
    DIRECTION = "direction"         # Up/Down/Sideways
    PRICE = "price"                 # Next price value
    RETURN = "return"               # Price return
    VOLATILITY = "volatility"       # Future volatility
    TREND_STRENGTH = "trend_strength" # Trend strength (0-100)
    TIME_TO_EVENT = "time_to_event"  # Bars until event


class TimeHorizon(Enum):
    """Prediction time horizons"""
    SCALP = 5               # 5 bars ahead
    SHORT = 12              # 12 bars ahead
    MEDIUM = 24             # 24 bars ahead
    SWING = 48              # 48 bars ahead
    POSITION = 96           # 96 bars ahead


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LSTMConfig:
    """Configuration for LSTM model"""
    architecture: LSTMArchitecture = LSTMArchitecture.STACKED
    prediction_target: PredictionTarget = PredictionTarget.DIRECTION
    time_horizon: TimeHorizon = TimeHorizon.SHORT
    
    # Sequence parameters
    sequence_length: int = 60        # Input sequence length
    prediction_steps: int = 12       # How many steps to predict
    
    # Model architecture
    lstm_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    dense_units: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    
    # Regularization
    l1_reg: float = 0.0001
    l2_reg: float = 0.0001
    
    # Features
    use_technical_features: bool = True
    use_ict_features: bool = True
    
    # Model saving
    save_model: bool = True
    model_path: str = "./models/lstm"
    model_name: str = "ict_lstm"


@dataclass
class SequenceData:
    """Prepared sequence data for LSTM"""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scalers: Dict[str, Any]
    sequence_length: int
    prediction_steps: int


@dataclass
class LSTMPrediction:
    """LSTM model prediction result"""
    direction: str                   # 'up', 'down', 'sideways'
    confidence: float                # 0-100
    predicted_values: List[float]    # Predicted sequence
    trend_strength: float            # 0-100
    volatility_forecast: float       # Expected volatility
    time_to_reversal: Optional[int]  # Estimated bars to reversal
    support_levels: List[float]      # Predicted support
    resistance_levels: List[float]   # Predicted resistance
    entry_zones: List[Dict]          # Optimal entry zones
    exit_targets: List[float]        # Target prices
    risk_assessment: str             # 'low', 'medium', 'high'


@dataclass
class TrainingHistory:
    """Training history and metrics"""
    epochs_trained: int
    train_loss: List[float]
    val_loss: List[float]
    train_accuracy: List[float]
    val_accuracy: List[float]
    best_val_loss: float
    best_epoch: int
    training_time: float
    final_metrics: Dict[str, float]


# =============================================================================
# FEATURE BUILDER FOR LSTM
# =============================================================================

class LSTMFeatureBuilder:
    """
    Build sequential features for LSTM from OHLCV and ICT data.
    
    FEATURE CATEGORIES:
    1. Price Features - OHLCV normalized
    2. Technical Features - RSI, ATR, MA crossovers
    3. ICT Features - OB distance, FVG fill, liquidity levels
    4. Time Features - Session, day encoding
    5. Derived Features - Returns, volatility, momentum
    """
    
    def __init__(self, config: LSTMConfig):
        self.config = config
        self.price_scaler = MinMaxScaler() if SKLEARN_AVAILABLE else None
        self.feature_scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_names = []
        
    def build_features(
        self,
        ohlcv: pd.DataFrame,
        ict_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Build all features from OHLCV and ICT data.
        
        Args:
            ohlcv: OHLCV DataFrame with columns [open, high, low, close, volume]
            ict_data: Optional dictionary with ICT analysis DataFrames
            
        Returns:
            DataFrame with all features
        """
        df = ohlcv.copy()
        
        # Ensure column names are lowercase
        df.columns = [c.lower() for c in df.columns]
        
        # Build feature categories
        df = self._add_price_features(df)
        
        if self.config.use_technical_features:
            df = self._add_technical_features(df)
            
        if self.config.use_ict_features and ict_data:
            df = self._add_ict_features(df, ict_data)
            
        df = self._add_time_features(df)
        df = self._add_derived_features(df)
        
        # Drop NaN rows (from indicator calculations)
        df = df.dropna()
        
        # Store feature names
        self.feature_names = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add normalized price features"""
        # Log returns
        df['return'] = np.log(df['close'] / df['close'].shift(1))
        df['return_2'] = df['return'].rolling(2).sum()
        df['return_5'] = df['return'].rolling(5).sum()
        df['return_10'] = df['return'].rolling(10).sum()
        
        # Price relative to range
        df['hl_range'] = df['high'] - df['low']
        df['close_position'] = (df['close'] - df['low']) / (df['hl_range'] + 1e-8)
        
        # Candle patterns
        df['body'] = df['close'] - df['open']
        df['body_ratio'] = df['body'].abs() / (df['hl_range'] + 1e-8)
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Gap detection
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_filled'] = ((df['low'] <= df['close'].shift(1)) & (df['high'] >= df['close'].shift(1))).astype(int)
        
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        # Simple Moving Averages
        for period in [10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff(5) / 5
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
            
        # EMA
        for period in [12, 26]:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
            df['volume_trend'] = df['volume'].rolling(10).mean() / (df['volume'].rolling(30).mean() + 1e-8)
            
        # Volatility
        df['volatility_10'] = df['return'].rolling(10).std()
        df['volatility_20'] = df['return'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_10'] / (df['volatility_20'] + 1e-8)
        
        return df
    
    def _add_ict_features(self, df: pd.DataFrame, ict_data: Dict) -> pd.DataFrame:
        """Add ICT-specific features"""
        n = len(df)
        
        # Order Block features
        if 'order_blocks' in ict_data:
            ob_df = ict_data['order_blocks']
            df['nearest_bullish_ob_dist'] = np.nan
            df['nearest_bearish_ob_dist'] = np.nan
            df['ob_strength'] = 0
            df['in_bullish_ob'] = 0
            df['in_bearish_ob'] = 0
            
        # FVG features
        if 'fvgs' in ict_data:
            fvg_df = ict_data['fvgs']
            df['nearest_fvg_dist'] = np.nan
            df['fvg_size'] = 0
            df['fvg_fill_progress'] = 0
            df['unfilled_fvg_count'] = 0
            
        # Liquidity features
        if 'liquidity' in ict_data:
            liq_df = ict_data['liquidity']
            df['buy_liq_distance'] = np.nan
            df['sell_liq_distance'] = np.nan
            df['recent_sweep'] = 0
            df['liq_cluster_strength'] = 0
            
        # Structure features
        if 'structure' in ict_data:
            struct_df = ict_data['structure']
            df['trend_direction'] = 0  # -1, 0, 1
            df['structure_break'] = 0
            df['displacement'] = 0
            df['swing_high_dist'] = np.nan
            df['swing_low_dist'] = np.nan
            
        # Premium/Discount zone
        df['premium_discount'] = 0  # -1 = discount, 0 = eq, 1 = premium
        
        # If no ICT data provided, calculate basic versions
        if not ict_data:
            # Calculate swing points
            df['swing_high'] = df['high'].rolling(10, center=True).max()
            df['swing_low'] = df['low'].rolling(10, center=True).min()
            
            # Range for premium/discount
            range_high = df['high'].rolling(50).max()
            range_low = df['low'].rolling(50).min()
            range_mid = (range_high + range_low) / 2
            
            df['premium_discount'] = np.where(
                df['close'] > range_mid + (range_high - range_mid) * 0.5, 1,
                np.where(df['close'] < range_mid - (range_mid - range_low) * 0.5, -1, 0)
            )
            
            # Simple FVG detection
            df['potential_fvg'] = ((df['low'] > df['high'].shift(2)) | 
                                   (df['high'] < df['low'].shift(2))).astype(int)
                                   
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'timestamp' in df.columns or df.index.dtype == 'datetime64[ns]':
            if 'timestamp' in df.columns:
                ts = pd.to_datetime(df['timestamp'])
            else:
                ts = df.index
                
            # Hour encoding (cyclical)
            df['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24)
            
            # Day of week encoding
            df['dow_sin'] = np.sin(2 * np.pi * ts.dayofweek / 7)
            df['dow_cos'] = np.cos(2 * np.pi * ts.dayofweek / 7)
            
            # Session indicators
            hour = ts.hour
            df['london_session'] = ((hour >= 8) & (hour < 16)).astype(int)
            df['ny_session'] = ((hour >= 13) & (hour < 22)).astype(int)
            df['asian_session'] = ((hour >= 0) & (hour < 8)).astype(int)
            
            # Kill zones
            df['london_kz'] = ((hour >= 8) & (hour < 11)).astype(int)
            df['ny_kz'] = ((hour >= 14) & (hour < 17)).astype(int)
            
            # Trading day features
            df['is_monday'] = (ts.dayofweek == 0).astype(int)
            df['is_friday'] = (ts.dayofweek == 4).astype(int)
            df['is_weekend'] = (ts.dayofweek >= 5).astype(int)
            
        else:
            # Add dummy time features if no timestamp
            df['hour_sin'] = 0
            df['hour_cos'] = 0
            df['dow_sin'] = 0
            df['dow_cos'] = 0
            df['london_session'] = 0
            df['ny_session'] = 0
            df['asian_session'] = 0
            
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived/combined features"""
        # Trend indicators combined
        if all(col in df.columns for col in ['sma_10', 'sma_20', 'sma_50']):
            df['trend_alignment'] = (
                (df['sma_10'] > df['sma_20']).astype(int) +
                (df['sma_20'] > df['sma_50']).astype(int)
            ) - 1  # -1, 0, 1
            
        # Momentum divergence
        if 'rsi' in df.columns and 'momentum_10' in df.columns:
            df['rsi_momentum_divergence'] = (
                (df['rsi'].diff() > 0).astype(int) != 
                (df['momentum_10'].diff() > 0).astype(int)
            ).astype(int)
            
        # Volatility regime
        if 'volatility_20' in df.columns:
            vol_percentile = df['volatility_20'].rolling(100).rank(pct=True)
            df['high_vol_regime'] = (vol_percentile > 0.8).astype(int)
            df['low_vol_regime'] = (vol_percentile < 0.2).astype(int)
            
        # Price action patterns
        df['inside_bar'] = (
            (df['high'] < df['high'].shift(1)) & 
            (df['low'] > df['low'].shift(1))
        ).astype(int)
        
        df['outside_bar'] = (
            (df['high'] > df['high'].shift(1)) & 
            (df['low'] < df['low'].shift(1))
        ).astype(int)
        
        # Consecutive moves
        df['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['consecutive_up'] = df['consecutive_up'].groupby(
            (df['consecutive_up'] != df['consecutive_up'].shift()).cumsum()
        ).cumsum()
        
        return df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        target_col: str = 'direction',
        sequence_length: int = None,
        prediction_steps: int = None
    ) -> SequenceData:
        """
        Create sequences for LSTM training.
        
        Args:
            df: Feature DataFrame
            target_col: Target column name
            sequence_length: Length of input sequence
            prediction_steps: Steps ahead to predict
            
        Returns:
            SequenceData with prepared arrays
        """
        seq_len = sequence_length or self.config.sequence_length
        pred_steps = prediction_steps or self.config.prediction_steps
        
        # Prepare target
        if target_col == 'direction':
            # Create direction labels: 0=down, 1=sideways, 2=up
            future_return = df['close'].shift(-pred_steps) / df['close'] - 1
            threshold = df['atr'].mean() * 0.5 / df['close'].mean() if 'atr' in df.columns else 0.001
            
            df['target'] = np.where(
                future_return > threshold, 2,
                np.where(future_return < -threshold, 0, 1)
            )
        elif target_col == 'return':
            df['target'] = df['close'].shift(-pred_steps) / df['close'] - 1
        else:
            df['target'] = df[target_col].shift(-pred_steps)
            
        # Remove NaN targets
        df = df.dropna(subset=['target'])
        
        # Select features
        feature_cols = [c for c in df.columns if c not in ['target', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Scale features
        features = df[feature_cols].values
        if self.feature_scaler:
            features = self.feature_scaler.fit_transform(features)
            
        targets = df['target'].values
        
        # Create sequences
        X, y = [], []
        for i in range(seq_len, len(features)):
            X.append(features[i-seq_len:i])
            y.append(targets[i])
            
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, shuffle=False
        )
        
        return SequenceData(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=feature_cols,
            scalers={'features': self.feature_scaler},
            sequence_length=seq_len,
            prediction_steps=pred_steps
        )


# =============================================================================
# LSTM MODEL BUILDER
# =============================================================================

class LSTMModelBuilder:
    """
    Build various LSTM architectures for ICT prediction.
    """
    
    def __init__(self, config: LSTMConfig):
        self.config = config
        
    def build(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 3
    ) -> Model:
        """
        Build LSTM model based on configuration.
        
        Args:
            input_shape: (sequence_length, n_features)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM models")
            
        architecture = self.config.architecture
        
        if architecture == LSTMArchitecture.SIMPLE:
            model = self._build_simple(input_shape, num_classes)
        elif architecture == LSTMArchitecture.STACKED:
            model = self._build_stacked(input_shape, num_classes)
        elif architecture == LSTMArchitecture.BIDIRECTIONAL:
            model = self._build_bidirectional(input_shape, num_classes)
        elif architecture == LSTMArchitecture.CNN_LSTM:
            model = self._build_cnn_lstm(input_shape, num_classes)
        elif architecture == LSTMArchitecture.ATTENTION:
            model = self._build_attention(input_shape, num_classes)
        elif architecture == LSTMArchitecture.MULTI_HEAD:
            model = self._build_multi_head(input_shape, num_classes)
        else:
            model = self._build_stacked(input_shape, num_classes)
            
        # Compile model
        if num_classes > 1:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
            
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def _build_simple(self, input_shape: Tuple, num_classes: int) -> Model:
        """Simple single LSTM layer model"""
        model = Sequential([
            LSTM(
                self.config.lstm_units[0],
                input_shape=input_shape,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout
            ),
            Dense(self.config.dense_units[0], activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(num_classes, activation='softmax' if num_classes > 1 else 'linear')
        ])
        return model
    
    def _build_stacked(self, input_shape: Tuple, num_classes: int) -> Model:
        """Stacked LSTM layers"""
        model = Sequential()
        
        # LSTM layers
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1
            
            if i == 0:
                model.add(LSTM(
                    units,
                    input_shape=input_shape,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.recurrent_dropout,
                    kernel_regularizer=l1_l2(self.config.l1_reg, self.config.l2_reg)
                ))
            else:
                model.add(LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.recurrent_dropout
                ))
                
            model.add(BatchNormalization())
            
        # Dense layers
        for units in self.config.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.config.dropout_rate))
            
        model.add(Dense(num_classes, activation='softmax' if num_classes > 1 else 'linear'))
        
        return model
    
    def _build_bidirectional(self, input_shape: Tuple, num_classes: int) -> Model:
        """Bidirectional LSTM"""
        model = Sequential()
        
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1
            
            if i == 0:
                model.add(Bidirectional(
                    LSTM(units, return_sequences=return_sequences,
                         dropout=self.config.dropout_rate),
                    input_shape=input_shape
                ))
            else:
                model.add(Bidirectional(
                    LSTM(units, return_sequences=return_sequences,
                         dropout=self.config.dropout_rate)
                ))
                
            model.add(BatchNormalization())
            
        for units in self.config.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.config.dropout_rate))
            
        model.add(Dense(num_classes, activation='softmax' if num_classes > 1 else 'linear'))
        
        return model
    
    def _build_cnn_lstm(self, input_shape: Tuple, num_classes: int) -> Model:
        """CNN for feature extraction + LSTM for sequence"""
        model = Sequential([
            # CNN feature extraction
            Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # LSTM sequence processing
            LSTM(self.config.lstm_units[0], return_sequences=True,
                 dropout=self.config.dropout_rate),
            LSTM(self.config.lstm_units[1] if len(self.config.lstm_units) > 1 else 32,
                 dropout=self.config.dropout_rate),
            
            # Dense output
            Dense(self.config.dense_units[0], activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(num_classes, activation='softmax' if num_classes > 1 else 'linear')
        ])
        
        return model
    
    def _build_attention(self, input_shape: Tuple, num_classes: int) -> Model:
        """LSTM with attention mechanism"""
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        x = LSTM(self.config.lstm_units[0], return_sequences=True,
                 dropout=self.config.dropout_rate)(inputs)
        x = BatchNormalization()(x)
        
        # Simple attention (using Dense for attention weights)
        attention_weights = Dense(1, activation='tanh')(x)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        x = tf.reduce_sum(x * attention_weights, axis=1)
        
        # Dense layers
        for units in self.config.dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.config.dropout_rate)(x)
            
        outputs = Dense(num_classes, activation='softmax' if num_classes > 1 else 'linear')(x)
        
        return Model(inputs=inputs, outputs=outputs)
    
    def _build_multi_head(self, input_shape: Tuple, num_classes: int) -> Model:
        """Multiple parallel LSTM heads for different aspects"""
        inputs = Input(shape=input_shape)
        
        # Head 1: Trend detection
        trend_lstm = LSTM(64, dropout=self.config.dropout_rate)(inputs)
        trend_dense = Dense(32, activation='relu')(trend_lstm)
        
        # Head 2: Volatility analysis
        vol_lstm = LSTM(64, dropout=self.config.dropout_rate)(inputs)
        vol_dense = Dense(32, activation='relu')(vol_lstm)
        
        # Head 3: Momentum
        momentum_lstm = LSTM(64, dropout=self.config.dropout_rate)(inputs)
        momentum_dense = Dense(32, activation='relu')(momentum_lstm)
        
        # Concatenate heads
        concat = Concatenate()([trend_dense, vol_dense, momentum_dense])
        
        # Final dense layers
        x = Dense(64, activation='relu')(concat)
        x = Dropout(self.config.dropout_rate)(x)
        x = Dense(32, activation='relu')(x)
        
        outputs = Dense(num_classes, activation='softmax' if num_classes > 1 else 'linear')(x)
        
        return Model(inputs=inputs, outputs=outputs)


# =============================================================================
# LSTM TRAINER
# =============================================================================

class ICTLSTMTrainer:
    """
    Train and evaluate LSTM models for ICT prediction.
    """
    
    def __init__(self, config: Optional[LSTMConfig] = None):
        self.config = config or LSTMConfig()
        self.feature_builder = LSTMFeatureBuilder(self.config)
        self.model_builder = LSTMModelBuilder(self.config)
        self.model = None
        self.training_history = None
        
    def train(
        self,
        ohlcv: pd.DataFrame,
        ict_data: Optional[Dict] = None,
        config: Optional[LSTMConfig] = None
    ) -> TrainingHistory:
        """
        Train LSTM model on OHLCV data.
        
        Args:
            ohlcv: OHLCV DataFrame
            ict_data: Optional ICT analysis data
            config: Training configuration
            
        Returns:
            Training history
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM training")
            
        config = config or self.config
        start_time = datetime.now()
        
        logger.info(f"Building features for LSTM training...")
        
        # Build features
        df = self.feature_builder.build_features(ohlcv, ict_data)
        
        logger.info(f"Created {len(df)} samples with {len(self.feature_builder.feature_names)} features")
        
        # Create sequences
        seq_data = self.feature_builder.create_sequences(
            df,
            target_col='direction',
            sequence_length=config.sequence_length,
            prediction_steps=config.prediction_steps
        )
        
        logger.info(f"Created sequences - Train: {len(seq_data.X_train)}, Val: {len(seq_data.X_val)}, Test: {len(seq_data.X_test)}")
        
        # Determine number of classes
        num_classes = len(np.unique(seq_data.y_train))
        
        # Build model
        input_shape = (seq_data.X_train.shape[1], seq_data.X_train.shape[2])
        self.model = self.model_builder.build(input_shape, num_classes)
        
        logger.info(f"Built {config.architecture.value} model with {self.model.count_params()} parameters")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if config.save_model:
            os.makedirs(config.model_path, exist_ok=True)
            checkpoint_path = os.path.join(config.model_path, f"{config.model_name}_best.keras")
            callbacks.append(ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ))
            
        # Train model
        logger.info("Starting LSTM training...")
        
        history = self.model.fit(
            seq_data.X_train,
            seq_data.y_train,
            validation_data=(seq_data.X_val, seq_data.y_val),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        test_results = self.model.evaluate(seq_data.X_test, seq_data.y_test, verbose=0)
        
        # Create predictions for metrics
        y_pred = np.argmax(self.model.predict(seq_data.X_test, verbose=0), axis=1)
        test_accuracy = accuracy_score(seq_data.y_test, y_pred)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Build training history
        self.training_history = TrainingHistory(
            epochs_trained=len(history.history['loss']),
            train_loss=history.history['loss'],
            val_loss=history.history['val_loss'],
            train_accuracy=history.history.get('accuracy', []),
            val_accuracy=history.history.get('val_accuracy', []),
            best_val_loss=min(history.history['val_loss']),
            best_epoch=np.argmin(history.history['val_loss']) + 1,
            training_time=training_time,
            final_metrics={
                'test_loss': test_results[0],
                'test_accuracy': test_accuracy,
                'n_train_samples': len(seq_data.X_train),
                'n_features': seq_data.X_train.shape[2]
            }
        )
        
        logger.info(f"Training complete. Test accuracy: {test_accuracy:.4f}")
        
        return self.training_history
    
    def predict(
        self,
        ohlcv: pd.DataFrame,
        ict_data: Optional[Dict] = None
    ) -> LSTMPrediction:
        """
        Make prediction using trained model.
        
        Args:
            ohlcv: Recent OHLCV data (at least sequence_length bars)
            ict_data: Optional ICT analysis data
            
        Returns:
            LSTMPrediction with direction and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Build features
        df = self.feature_builder.build_features(ohlcv, ict_data)
        
        # Get last sequence
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        features = df[feature_cols].values[-self.config.sequence_length:]
        
        if self.feature_builder.feature_scaler:
            features = self.feature_builder.feature_scaler.transform(features)
            
        # Reshape for model
        X = np.array([features])
        
        # Predict
        proba = self.model.predict(X, verbose=0)[0]
        predicted_class = np.argmax(proba)
        confidence = float(proba[predicted_class]) * 100
        
        # Map to direction
        direction_map = {0: 'down', 1: 'sideways', 2: 'up'}
        direction = direction_map.get(predicted_class, 'sideways')
        
        # Calculate support/resistance from recent data
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current_price = df['close'].iloc[-1]
        
        # Estimate trend strength from probabilities
        if direction == 'up':
            trend_strength = (proba[2] - proba[0]) * 100
        elif direction == 'down':
            trend_strength = (proba[0] - proba[2]) * 100
        else:
            trend_strength = 0
            
        # Risk assessment
        if confidence > 70:
            risk_assessment = 'low'
        elif confidence > 50:
            risk_assessment = 'medium'
        else:
            risk_assessment = 'high'
            
        return LSTMPrediction(
            direction=direction,
            confidence=confidence,
            predicted_values=[current_price * (1 + (0.01 if direction == 'up' else -0.01 if direction == 'down' else 0))],
            trend_strength=trend_strength,
            volatility_forecast=df['atr'].iloc[-1] if 'atr' in df.columns else 0,
            time_to_reversal=None,
            support_levels=[recent_low, recent_low - (recent_high - recent_low) * 0.236],
            resistance_levels=[recent_high, recent_high + (recent_high - recent_low) * 0.236],
            entry_zones=[{
                'price': current_price,
                'zone_type': 'current',
                'confidence': confidence
            }],
            exit_targets=[
                recent_high if direction == 'up' else recent_low,
                recent_high * 1.01 if direction == 'up' else recent_low * 0.99
            ],
            risk_assessment=risk_assessment
        )
    
    def save_model(self, path: str = None):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
            
        path = path or os.path.join(self.config.model_path, f"{self.config.model_name}.keras")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
        # Save feature builder state
        fb_path = path.replace('.keras', '_features.pkl')
        with open(fb_path, 'wb') as f:
            pickle.dump({
                'feature_names': self.feature_builder.feature_names,
                'feature_scaler': self.feature_builder.feature_scaler,
                'config': self.config
            }, f)
            
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load trained model"""
        self.model = load_model(path)
        
        fb_path = path.replace('.keras', '_features.pkl')
        if os.path.exists(fb_path):
            with open(fb_path, 'rb') as f:
                state = pickle.load(f)
                self.feature_builder.feature_names = state['feature_names']
                self.feature_builder.feature_scaler = state['feature_scaler']
                self.config = state.get('config', self.config)
                
        logger.info(f"Model loaded from {path}")


# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================

class SyntheticOHLCVGenerator:
    """Generate synthetic OHLCV data for testing"""
    
    @staticmethod
    def generate(
        n_bars: int = 5000,
        initial_price: float = 1.1000,
        volatility: float = 0.0005,
        trend_strength: float = 0.0001
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data with trend cycles"""
        np.random.seed(42)
        
        dates = pd.date_range(
            start='2024-01-01',
            periods=n_bars,
            freq='15min'
        )
        
        # Generate price with trend cycles
        prices = [initial_price]
        trend = 0
        
        for i in range(1, n_bars):
            # Cycle trend direction every ~200 bars
            if i % 200 == 0:
                trend = np.random.choice([-1, 0, 1])
                
            # Random walk with trend
            change = np.random.normal(
                trend * trend_strength,
                volatility
            )
            prices.append(prices[-1] * (1 + change))
            
        prices = np.array(prices)
        
        # Generate OHLC from prices
        ohlcv = []
        for i, close in enumerate(prices):
            volatility_mult = np.random.uniform(0.8, 1.2)
            high_offset = np.random.uniform(0, volatility * volatility_mult)
            low_offset = np.random.uniform(0, volatility * volatility_mult)
            
            if i == 0:
                open_price = close
            else:
                open_price = prices[i-1]
                
            high = max(open_price, close) + high_offset * close
            low = min(open_price, close) - low_offset * close
            volume = np.random.randint(1000, 10000)
            
            ohlcv.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
            
        return pd.DataFrame(ohlcv)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ICT LSTM Trend Predictor - Phase 3 Deep Learning Module")
    print("=" * 60)
    
    if not TF_AVAILABLE:
        print("\n⚠️  TensorFlow not installed. Install with:")
        print("   pip install tensorflow")
        print("\nShowing module structure only...")
    else:
        print(f"\n✓ TensorFlow {tf.__version__} available")
        
        # Generate synthetic data
        print("\n📊 Generating synthetic OHLCV data...")
        ohlcv = SyntheticOHLCVGenerator.generate(n_bars=3000)
        print(f"   Generated {len(ohlcv)} bars")
        
        # Configure LSTM
        config = LSTMConfig(
            architecture=LSTMArchitecture.STACKED,
            prediction_target=PredictionTarget.DIRECTION,
            sequence_length=60,
            prediction_steps=12,
            lstm_units=[64, 32],
            dense_units=[32, 16],
            epochs=10,  # Short for demo
            batch_size=32,
            save_model=False
        )
        
        # Train model
        print("\n🧠 Training LSTM model...")
        trainer = ICTLSTMTrainer(config)
        history = trainer.train(ohlcv)
        
        print(f"\n📈 Training Results:")
        print(f"   Epochs: {history.epochs_trained}")
        print(f"   Best Val Loss: {history.best_val_loss:.4f}")
        print(f"   Test Accuracy: {history.final_metrics['test_accuracy']:.4f}")
        print(f"   Training Time: {history.training_time:.1f}s")
        
        # Make prediction
        print("\n🎯 Sample Prediction:")
        recent_data = ohlcv.tail(100)
        prediction = trainer.predict(recent_data)
        print(f"   Direction: {prediction.direction}")
        print(f"   Confidence: {prediction.confidence:.1f}%")
        print(f"   Trend Strength: {prediction.trend_strength:.1f}")
        print(f"   Risk Assessment: {prediction.risk_assessment}")
        
    print("\n" + "=" * 60)
    print("Module ready for integration")
    print("=" * 60)
