# **Stock Price Prediction Project**
This project aims to predict the closing price of Infosys stock using different deep learning models, including Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and 1D Convolutional Neural Network (Conv1D).

## **Project Overview**
The project is focused on using historical stock price data of Infosys to train and evaluate deep learning models for predicting the closing price of Infosys stock on the 10th day. The project uses Python programming language and popular deep learning libraries such as TensorFlow and Keras.

The project includes the following main components:

### **Data Preprocessing**: 
The historical stock price data is preprocessed to prepare it for training the deep learning models. This includes data normalization, feature engineering, and data splitting into training and testing sets.
### **Model Training**:
The deep learning models, including RNN, LSTM, GRU, and Conv1D, are trained on the preprocessed data using Keras API in TensorFlow. Hyperparameter tuning and model evaluation are performed during the training process.
### **Model Evaluation**:
The trained models are evaluated using various metrics, including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE), to assess their performance and accuracy in predicting the closing price of Infosys stock.
### **Results and Visualization**:
The predictions made by the trained models are compared with the actual closing price on the 10th day. The results are visualized using plots to provide insights and interpretations of the model performance.

## **Project Files**
The project includes the following main files:

**Stock Price Prediction Deep Learning.ipynb**: Jupyter notebook containing the main code for data preprocessing, model training, and evaluation.

**README.md**: This file providing project overview, instructions, and details.

**INFY.csv**: CSV file containing the historical stock price data of Infosys.

**requirements.txt**: Text file containing the list of required Python packages for running the project.

## **Getting Started**
To run the project locally, follow these steps:

1.Clone the repository to your local machine using the git clone command or by downloading the ZIP file from the GitHub repository.

2.Navigate to the project directory and create a virtual environment using python -m venv venv (optional but recommended).

3.Activate the virtual environment (if created) using the appropriate command based on your operating system.

4.Install the required Python packages using pip install -r requirements.txt command.

5.Open and run the stock_price_prediction.ipynb Jupyter notebook in your preferred environment (e.g. Jupyter Notebook, JupyterLab, Google Colab).

6.Follow the instructions in the notebook to preprocess the data, train and evaluate the models, and visualize the results.

## **Results**
The trained models are evaluated using various metrics, and the performance results are displayed in the notebook. The predictions made by the models are compared with the actual closing price on the 10th day, and the results are visualized using plots. The model with the best performance in terms of accuracy and prediction quality can be selected for further analysis or deployment in real-world scenarios.
