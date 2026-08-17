import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization,
    Input, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDeepLearning:
    """
    Deep learning models for time series prediction including LSTM, GRU, and Transformer.
    """
    
    def __init__(self, sequence_length: int = 30, prediction_horizon: int = 1):
        """
        Initialize the deep learning time series predictor.
        
        Args:
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of steps ahead to predict
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.models = {}
        self.history = {}
        
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for time series deep learning.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X) - self.prediction_horizon + 1):
            X_sequences.append(X[i-self.sequence_length:i])
            y_sequences.append(y[i+self.prediction_horizon-1])
            
        return np.array(X_sequences), np.array(y_sequences)
    
    def create_lstm_model(self, input_shape: Tuple[int, int], 
                         lstm_units: List[int] = [128, 64],
                         dropout_rate: float = 0.2,
                         l1_reg: float = 0.01,
                         l2_reg: float = 0.01) -> Model:
        """
        Create LSTM model for time series prediction.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate for regularization
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            lstm_units[0],
            return_sequences=len(lstm_units) > 1,
            input_shape=input_shape,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:]):
            return_seq = i < len(lstm_units) - 2
            model.add(LSTM(
                units,
                return_sequences=return_seq,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_gru_model(self, input_shape: Tuple[int, int],
                        gru_units: List[int] = [128, 64],
                        dropout_rate: float = 0.2,
                        l1_reg: float = 0.01,
                        l2_reg: float = 0.01) -> Model:
        """
        Create GRU model for time series prediction.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            gru_units: List of GRU layer units
            dropout_rate: Dropout rate for regularization
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            
        Returns:
            Compiled GRU model
        """
        model = Sequential()
        
        # First GRU layer
        model.add(GRU(
            gru_units[0],
            return_sequences=len(gru_units) > 1,
            input_shape=input_shape,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Additional GRU layers
        for i, units in enumerate(gru_units[1:]):
            return_seq = i < len(gru_units) - 2
            model.add(GRU(
                units,
                return_sequences=return_seq,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_transformer_model(self, input_shape: Tuple[int, int],
                               num_heads: int = 8,
                               ff_dim: int = 128,
                               num_transformer_blocks: int = 2,
                               dropout_rate: float = 0.2) -> Model:
        """
        Create Transformer model for time series prediction.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_transformer_blocks: Number of transformer blocks
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Transformer model
        """
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Transformer blocks
        for _ in range(num_transformer_blocks):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=input_shape[1] // num_heads
            )(x, x)
            attention_output = Dropout(dropout_rate)(attention_output)
            x1 = LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed-forward network
            ffn_output = Dense(ff_dim, activation='relu')(x1)
            ffn_output = Dropout(dropout_rate)(ffn_output)
            ffn_output = Dense(input_shape[1])(ffn_output)
            x = LayerNormalization(epsilon=1e-6)(x1 + ffn_output)
        
        # Global average pooling and final layers
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   model_params: Optional[Dict[str, Any]] = None,
                   epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train a deep learning model.
        
        Args:
            model_name: Name of the model ('lstm', 'gru', 'transformer')
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_params: Model-specific parameters
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Dictionary with training results
        """
        if model_params is None:
            model_params = {}
        
        # Scale the data
        X_train_scaled = self.scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = self.scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        # Create model
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        if model_name == 'lstm':
            model = self.create_lstm_model(input_shape, **model_params)
        elif model_name == 'gru':
            model = self.create_gru_model(input_shape, **model_params)
        elif model_name == 'transformer':
            model = self.create_transformer_model(input_shape, **model_params)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # Store model and history
        self.models[model_name] = model
        self.history[model_name] = history.history
        
        # Make predictions
        y_pred_scaled = model.predict(X_val_scaled, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'model_name': model_name,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse,
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the trained model
            X: Input features
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train the model first.")
        
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_pred_scaled = self.models[model_name].predict(X_scaled, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        epochs: int = 100, batch_size: int = 32) -> Dict[str, Dict[str, Any]]:
        """
        Train all deep learning models with default parameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Dictionary with results for all models
        """
        results = {}
        
        # LSTM model
        print("Training LSTM model...")
        results['lstm'] = self.train_model(
            'lstm', X_train, y_train, X_val, y_val,
            model_params={'lstm_units': [128, 64], 'dropout_rate': 0.2},
            epochs=epochs, batch_size=batch_size
        )
        
        # GRU model
        print("Training GRU model...")
        results['gru'] = self.train_model(
            'gru', X_train, y_train, X_val, y_val,
            model_params={'gru_units': [128, 64], 'dropout_rate': 0.2},
            epochs=epochs, batch_size=batch_size
        )
        
        # Transformer model
        print("Training Transformer model...")
        results['transformer'] = self.train_model(
            'transformer', X_train, y_train, X_val, y_val,
            model_params={'num_heads': 8, 'ff_dim': 128, 'num_transformer_blocks': 2},
            epochs=epochs, batch_size=batch_size
        )
        
        return results
    
    def get_best_model(self, results: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model based on R² score.
        
        Args:
            results: Results dictionary from train_all_models
            
        Returns:
            Tuple of (best_model_name, best_model_results)
        """
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        return best_model[0], best_model[1]
    
    def save_models(self, filepath_prefix: str):
        """
        Save all trained models.
        
        Args:
            filepath_prefix: Prefix for model file paths
        """
        for model_name, model in self.models.items():
            model.save(f"{filepath_prefix}_{model_name}.h5")
    
    def load_model(self, model_name: str, filepath: str):
        """
        Load a trained model.
        
        Args:
            model_name: Name to assign to the loaded model
            filepath: Path to the model file
        """
        self.models[model_name] = tf.keras.models.load_model(filepath)