
import joblib
import numpy as np
import tensorflow as tf
from datetime import datetime
import json

class PickleBasedIrrigationPredictor:
    """Irrigation predictor that uses pickle files for all artifacts"""

    def __init__(self, model_path, scaler_path, encoder_path, config_path):
        self.model = self._load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path) # May be None if not used
        self.config = self._load_config(config_path)
        self.feature_columns = self.config.get('feature_columns', [])
        self.sequence_length = self.config.get('sequence_length')

        if self.sequence_length is None:
             raise ValueError("Sequence length not found in config.")

        print("✅ Predictor initialized from files!")
        print(f"   Model: {model_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Encoder: {encoder_path}")
        print(f"   Config: {config_path}")
        print(f"   Sequence length: {self.sequence_length}")


    def _load_model(self, model_path):
        # Custom objects might be needed if the model uses custom layers or functions
        # For this model, it seems standard Keras layers are used
        return tf.keras.models.load_model(model_path)

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def preprocess_input(self, raw_data):
        """Preprocess raw input data (e.g., pandas DataFrame) for prediction"""
        # Ensure data has required columns and is sorted by timestamp
        if not all(col in raw_data.columns for col in self.feature_columns):
             missing = [col for col in self.feature_columns if col not in raw_data.columns]
             raise ValueError(f"Input data is missing required columns: {missing}")

        processed_data = raw_data[self.feature_columns].values.astype(np.float32)

        # Scale the data using the loaded scaler
        n_timesteps, n_features = processed_data.shape
        processed_data = processed_data.reshape(-1, n_features)
        scaled_data = self.scaler.transform(processed_data)
        scaled_data = scaled_data.reshape(n_timesteps, n_features)

        return scaled_data

    def create_sequences(self, processed_data):
        """Create sequences from preprocessed data"""
        sequences = []
        for i in range(self.sequence_length, len(processed_data) + 1):
            sequence = processed_data[i-self.sequence_length:i]
            sequences.append(sequence)
        return np.array(sequences, dtype=np.float32)


    def predict_irrigation(self, sequence_data, return_details=True):
        """Make irrigation prediction with confidence assessment"""
        # Ensure sequence_data has the right shape
        if len(sequence_data.shape) == 2:
            sequence_data = sequence_data.reshape(1, *sequence_data.shape)

        # Validate input shape
        expected_shape = (self.sequence_length, len(self.feature_columns))
        if sequence_data.shape[1:] != expected_shape:
            raise ValueError(f"Expected sequence shape {expected_shape}, got {sequence_data.shape[1:]}")

        # Scale the input data (assuming it's already preprocessed but not sequenced/scaled)
        # If input is raw, use preprocess_input first
        n_samples, n_timesteps, n_features = sequence_data.shape
        sequence_reshaped = sequence_data.reshape(-1, n_features)
        sequence_scaled = self.scaler.transform(sequence_reshaped) # Re-scale each sequence
        sequence_scaled = sequence_scaled.reshape(n_samples, n_timesteps, n_features)


        # Make prediction
        prediction_prob = self.model.predict(sequence_scaled, verbose=0)[0][0]
        prediction = int(prediction_prob > 0.5)

        # Calculate confidence
        confidence = abs(prediction_prob - 0.5) * 2

        if return_details:
            recommendation = self._get_recommendation(prediction_prob, confidence)
            return {
                'prediction': prediction,
                'probability': float(prediction_prob),
                'confidence': float(confidence),
                'recommendation': recommendation,
                'timestamp': datetime.now().isoformat()
            }
        else:
            return prediction

    def _get_recommendation(self, probability, confidence):
        """Generate detailed irrigation recommendation"""
        if probability > 0.8 and confidence > 0.6:
            return "🚿 IRRIGATE NOW - High confidence irrigation needed"
        elif probability > 0.6 and confidence > 0.4:
            return "💧 Consider irrigation - Moderate confidence"
        elif probability < 0.3 and confidence > 0.5:
            return "🚫 No irrigation needed - High confidence"
        elif probability < 0.5 and confidence > 0.3:
            return "⏹️ Hold irrigation - Moderate confidence"
        else:
            return "⏳ Monitor conditions - Low confidence prediction"

    def predict_batch(self, sequences_data):
        """Make predictions for multiple sequences"""
        predictions = []
        for sequence in sequences_data:
            pred = self.predict_irrigation(sequence, return_details=True)
            predictions.append(pred)
        return predictions

# Example Usage (within the generated script)
# if __name__ == "__main__":
#     # Load the predictor
#     predictor = PickleBasedIrrigationPredictor(
#         model_path='advanced_irrigation_model.h5',
#         scaler_path='advanced_irrigation_model_scaler.pkl',
#         encoder_path='advanced_irrigation_model_encoder.pkl', # Use None if encoder was not saved
#         config_path='advanced_irrigation_model_config.json'
#     )

#     # Example prediction with a synthetic sequence
#     # You would replace this with actual data
#     synthetic_sequence = np.random.rand(predictor.sequence_length, len(predictor.feature_columns))

#     # Preprocess and create sequences if starting from raw data
#     # raw_df = pd.DataFrame(...) # Load your raw data
#     # processed_data = predictor.preprocess_input(raw_df)
#     # sequences = predictor.create_sequences(processed_data)
#     # if sequences.shape[0] > 0:
#     #    single_sequence_input = sequences[-1] # Use the last sequence for prediction
#     # else:
#     #    print("Not enough data to create a sequence of required length.")
#     #    single_sequence_input = None


#     # Make a prediction
#     # if single_sequence_input is not None:
#     #    prediction_result = predictor.predict_irrigation(single_sequence_input)
#     #    print("\nPrediction Result:")
#     #    print(prediction_result)

