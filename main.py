import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import joblib
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization, MultiHeadAttention, Flatten, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class EngineHealthHybridModel:
    def __init__(self, data_path, seq_length=30):
        self.data_path = data_path
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None
        self.numerical_columns = ['Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 
                                  'lub oil temp', 'Coolant temp', 'Engine rpm', 'Hour', 'Day', 'Month']
    
    def load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%Y-%m-%d %H:%M:%S")
        df = df.sort_values(by="Timestamp")
        df['Hour'] = df['Timestamp'].dt.hour
        df['Day'] = df['Timestamp'].dt.day
        df['Month'] = df['Timestamp'].dt.month
        df.drop(columns=['Timestamp'], inplace=True)
        
        df[self.numerical_columns] = self.scaler.fit_transform(df[self.numerical_columns])
        joblib.dump(self.scaler, "scaler.pkl")
        df.to_csv("preprocessed_engine_data.csv", index=False)
        print("Data Preprocessing Completed Successfully!")
    
    def create_sequences(self, data):
        sequences, labels = [], []
        for i in range(len(data) - self.seq_length):
            sequences.append(data[i:i + self.seq_length, :-1])  # Features
            labels.append(data[i + self.seq_length, -1])  # Target
        return np.array(sequences), np.array(labels)

    def prepare_data(self):
        df = pd.read_csv("preprocessed_engine_data.csv")
        data_array = df.values
        X, y = self.create_sequences(data_array)

        # ✅ Reshape target to (None, 1) to match model output
        y = y.reshape(-1, 1)

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def build_hybrid_model(self, input_shape, num_heads=4):
        inputs = Input(shape=input_shape)

        # **LSTM Encoder**
        lstm_encoder = LSTM(128, return_sequences=True)(inputs)
        lstm_encoder = Dropout(0.4)(lstm_encoder)

        # **TFT Transformer Block (Self-Attention)**
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(lstm_encoder, lstm_encoder)
        attn_output = LayerNormalization(epsilon=1e-6)(attn_output)

        # **LSTM Decoder**
        lstm_decoder = LSTM(64, return_sequences=False)(attn_output)
        lstm_decoder = Dropout(0.3)(lstm_decoder)

        # **Concatenate LSTM and Transformer Outputs**
        merged_features = Concatenate()([lstm_decoder, Flatten()(attn_output)])

        # Fully Connected Layer
        dense_output = Dense(64, activation='relu')(merged_features)
        dense_output = Dropout(0.3)(dense_output)
        dense_output = Dense(32, activation='relu')(dense_output)
        dense_output = Dropout(0.2)(dense_output)

        # Output Layer for Binary Classification
        output = Dense(1, activation='sigmoid')(dense_output)

        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Hybrid LSTM + TFT Model built successfully.")

    def train_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                      validation_data=(X_test, y_test))
        self.model.save("engine_health_hybrid_model.h5")
        print("Model training completed.")
    
    def evaluate_model(self, X_test, y_test):
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        final_results = {"Test Accuracy": test_acc * 100, "Test Loss": test_loss}
        with open("final_model_results_hybrid.json", "w") as f:
            json.dump(final_results, f)
        print("Model evaluation completed.")

    def plot_training_history(self):
        if self.history:
            plt.figure(figsize=(10, 5))
            plt.plot(self.history.history['accuracy'], label='Training Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Training and Validation Accuracy (Hybrid Model)')
            plt.savefig("training_validation_accuracy_hybrid.png")
            plt.close()
            
            plt.figure(figsize=(10, 5))
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss (Hybrid Model)')
            plt.savefig("training_validation_loss_hybrid.png")
            plt.close()
            print("Training history plots saved.")
        else:
            print("No training history found.")

# Usage example
if __name__ == "__main__":
    model_instance = EngineHealthHybridModel("engine_data.csv")
    model_instance.load_and_preprocess_data()
    X_train, X_test, y_train, y_test = model_instance.prepare_data()
    model_instance.build_hybrid_model((model_instance.seq_length, X_train.shape[2]))
    model_instance.train_model(X_train, y_train, X_test, y_test)
    model_instance.evaluate_model(X_test, y_test)
    model_instance.plot_training_history()
