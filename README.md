# EUR to PLN Exchange Rate Prediction using LSTM
This project implements and tests a Long Short-Term Memory (LSTM) neural network for predicting the EUR to PLN exchange rate based on historical data.
The data consists of daily opening, high, low, and closing exchange rates, and the model aims to predict the next day's opening rate.

## Project Structure
- **main.py:** The main LSTM model script that processes historical data, trains the LSTM model, and predicts exchange rates.
- **eurpln_d_train.csv:** Training dataset containing historical exchange rates.
- **eurpln_d_test.csv:** Test dataset used to evaluate the model's performance.

## Model Architecture
The LSTM model is built using the following layers:

1. LSTM: Two LSTM layers to capture the temporal dependencies in the time-series data. The first layer returns sequences, while the second does not.
2. Dropout: Regularization layers to prevent overfitting by dropping some units during training.
3. Dense: Fully connected layer with a single output representing the predicted opening price.

## Model Summary
- **Input Layer:** Sequences of past exchange rate data (days).
- **LSTM Layer 1:** 50 units with return sequences enabled.
- **Dropout Layer 1:** 20% dropout to avoid overfitting.
- **LSTM Layer 2:** 50 units without return sequences.
- **Dropout Layer 2:** 20% dropout.
- **Dense Layer:** A single fully connected layer outputting the next day's predicted exchange rate.

![1](https://github.com/user-attachments/assets/9b9fc25b-9c54-44a6-838b-98b0a5e262ab)

## Data Preprocessing
1. **Scaling:** The exchange rate values are scaled between 0 and 1 using min-max scaling to ensure faster convergence during training.
2. **Reshaping:** The data is reshaped into sequences to serve as input for the LSTM network.

## Results
### Training Loss and Validation Loss
![2](https://github.com/user-attachments/assets/56d2e34e-9159-48a8-8cfd-0976be4a51ab)
![3)](https://github.com/user-attachments/assets/7bdabf6b-1296-4ea2-81d8-ffccc89a8bd9)
![4](https://github.com/user-attachments/assets/afc8fce1-5690-49f8-843d-0a2ad2ea8a12)
![5](https://github.com/user-attachments/assets/ba28be0b-24ed-4495-bf66-849cfe3196e2)
![6](https://github.com/user-attachments/assets/ec8e2709-1a76-4a8f-ab14-7e916e7a993b)


## How to Run the Project
1. **Download Historical Data:** Use the historical EUR to PLN exchange rate data in CSV format.
2. **Install Dependencies:** Install required libraries using the command:
``` 
pip install -r requirements.txt
```
3. **Run the Script:**
     - Place the ***eurpln_d_train.csv*** and ***eurpln_d_test.csv*** files in the same directory as main.py.
     - Run the script to train the model and generate predictions:
      ```
      python main.py
      ```
4. **View Results:** The results (plots and predictions) will be saved as .png files in the project directory.

## Key Takeaways

### Model Performance
Training Dataset: The model closely follows the actual trends in the training data, which suggests that it has learned the patterns in the time series.
Test Dataset: Predictions on the test data show the model's ability to generalize to unseen data, though there may be some divergence from actual values.

### What I Learned
Through this project, I gained insights into:

- **Time Series Forecasting with LSTM:** Understanding how LSTM models handle sequential data and why they're effective for time series predictions.
- **Data Preprocessing:** The importance of scaling and reshaping data for feeding into neural networks, especially for sequential data.
- **Model Regularization:** Using dropout layers to prevent overfitting in deep neural networks.
- **Evaluating Model Performance:** How to visualize model predictions and loss curves to understand overfitting and underfitting.
- **Model Training and Tuning:** Fine-tuning hyperparameters such as batch size and learning rate to improve model performance.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
