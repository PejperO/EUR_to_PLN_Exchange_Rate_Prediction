import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import plot_model

model_name = "lstmv1EURtoPLN"

# Loading data from CSV file
train_data = 'eurpln_d_train.csv'
test_data = 'eurpln_d_test.csv'

df_train = pd.read_csv(train_data)
df_test = pd.read_csv(test_data, header=None)
df_test.columns = ['Data', 'Otwarcie', 'Najwyzszy', 'Najnizszy', 'Zamkniecie']

# Displaying column names
print("Columns in training data:", df_train.columns)
print("Columns in test data:", df_test.columns)

# Checking sample data
print("Sample training data:\n", df_train.head())
print("Sample test data:\n", df_test.head())

# Creating X (index) and y (open value) for training data
if 'Otwarcie' in df_train.columns and 'Otwarcie' in df_test.columns:
    y_train = df_train['Otwarcie'].values
    y_test = df_test['Otwarcie'].values
else:
    raise KeyError("The 'Open' column does not exist in the training or test data.")

X_train_dates = np.arange(len(df_train)).reshape(-1, 1)
X_test_dates = np.arange(len(df_test)).reshape(-1, 1)

# Plotting training data
plt.plot(X_train_dates, y_train, color='black', label='Data')
plt.xlabel('days from 2014-05-19 to 2024-05-18')
plt.ylabel('Open value in EUR to PLN')
plt.title('10 years - EUR to PLN - Training Data')
plt.legend()
plt.savefig(f'data_train_{model_name}.png')
plt.show()

# Plotting test data
plt.plot(X_test_dates, y_test, color='black', label='Data')
plt.xlabel('days from 2014-05-19 to 2024-05-18')
plt.ylabel('Open value in EUR to PLN')
plt.title('10 years - EUR to PLN - Test Data')
plt.legend()
plt.savefig(f'data_test_{model_name}.png')
plt.show()

# Scaling y values to the range between 0 and 1
min_y = np.min(np.concatenate((y_train, y_test)))
max_y = np.max(np.concatenate((y_train, y_test)))
y_train = (y_train - min_y) / (max_y - min_y)
y_test = (y_test - min_y) / (max_y - min_y)

# Reshaping data for LSTM model
def create_sequences(y_data, seq_length):
    X_sequences = []
    y_labels = []
    for i in range(len(y_data) - seq_length):
        X_sequences.append(y_data[i:i+seq_length])
        y_labels.append(y_data[i+seq_length])
    return np.array(X_sequences), np.array(y_labels)

# Sequence length
seq_length = 50

# Creating sequences for training and test data
X_train_seq, y_train_seq = create_sequences(y_train, seq_length)
X_test_seq, y_test_seq = create_sequences(y_test, seq_length)
print(f'X_train_seq Shape: {X_train_seq.shape}, y_train_seq Shape: {y_train_seq.shape}')
print(f'X_test_seq Shape: {X_test_seq.shape}, y_test_seq Shape: {y_test_seq.shape}')

# Reshape X to be 3D [samples, time steps, features] as expected by LSTM
X_train_seq = np.reshape(X_train_seq, (X_train_seq.shape[0], X_train_seq.shape[1], 1))
X_test_seq = np.reshape(X_test_seq, (X_test_seq.shape[0], X_test_seq.shape[1], 1))

# Model Definition
Dropout_rate = 0.2
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dropout(Dropout_rate))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(Dropout_rate))
model.add(Dense(1))  # Output layer with one neuron

# Model compilation
model.compile(optimizer='adam', loss='mean_squared_error')

# Model training
batch_size = round(len(X_train_seq) / 10)
print('batch_size =', batch_size)
history = model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=batch_size, verbose=1)

# Plotting Loss vs Epochs
plt.figure(figsize=(10, 7))
plt.plot(history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("RNN model, Loss MSE vs Epoch")
plt.ylim(0, 0.1)  # Setting the y-axis range from 0 to 0.1
mid_x = len(history.history["loss"]) / 2  # Center of x-axis (number of epochs divided by 2)
mid_y = 0.05    # Center of the y-axis (0.05 is half the range from 0 to 0.1)
plt.text(mid_x, mid_y, f' Sequence length = {seq_length} \n Batch size={batch_size} \n Dropout rate={Dropout_rate} \n Input data - {train_data} \n Final MSE = {round(history.history["loss"][-1], 5)}', fontsize=16, ha='left')  # Dodanie tekstu na środku obrazka
plt.savefig(f'history_{model_name}.png')
plt.show()

# Predictions on training data
predictions_train = model.predict(X_train_seq)
# chart
plt.plot(predictions_train.flatten(), color='red', label='Predykcje')
plt.plot(y_train_seq.flatten(), color='blue', label='Dane treningowe', alpha=0.5)
plt.legend()
mid_x = len(predictions_train) / 2  # Center of x-axis (number of epochs divided by 2)
mid_y = 1 / 10  # Center of the y-axis (0.05 is half the range from 0 to 0.1)
# Residual Sum of Squares
plt.text(mid_x, mid_y, f' RSS = {round(np.sum((predictions_train.flatten() - y_train_seq.flatten()) ** 2), 4)} \n MSE = {round(np.mean((predictions_train.flatten() - y_train_seq.flatten()) ** 2), 6)} ', fontsize=16, ha='left')  # Dodanie tekstu na środku obrazka
plt.savefig(f'pred_train_{model_name}.png')
plt.show()

# Predictions on test data
predictions_test = model.predict(X_test_seq)
# chart
plt.plot(predictions_test.flatten(), color='red', label='Predykcje', alpha=1)
plt.plot(y_test_seq.flatten(), color='black', label='Dane testowe', alpha=0.5)
plt.legend()
mid_x = len(predictions_test) / 8  # Center of x-axis (number of epochs divided by 2)
mid_y = 10 / 35  # Center of the y-axis (0.05 is half the range from 0 to 0.1)
# Residual Sum of Squares
plt.text(mid_x, mid_y, f' RSS = {round(np.sum((predictions_test.flatten() - y_test_seq.flatten()) ** 2), 4)} \n MSE = {round(np.mean((predictions_test.flatten() - y_test_seq.flatten()) ** 2), 6)} ', fontsize=16, ha='left')  # Dodanie tekstu na środku obrazka
plt.savefig(f'pred_test_{model_name}.png')
plt.show()

# Export RNN model
model.save(f"keras_{model_name}.keras")

# Model visualization
plot_model(model, to_file=f'architecture_{model_name}.png', show_shapes=True, show_layer_names=True)
