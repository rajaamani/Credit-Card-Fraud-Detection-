# Credit Card Fraud Detection using Autoencoders

![Credit Card Fraud Detection](https://github.com/rajaamani/Credit-Card-Fraud-Detection-/assets/101103515/dd5facaf-607c-4da9-8252-99c97336df23)

**Table of Contents:**
- [Project Overview](#project-overview)
- [Autoencoder Architecture](#autoencoder-architecture)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Transformation](#data-transformation)
- [Model Building](#model-building)
- [Encoding Data](#encoding-data)
- [Linear Classifier](#linear-classifier)
- [Non-linear Classifier](#non-linear-classifier)
- [Results](#results)
- [Conclusion](#conclusion)

## Project Overview

This project utilizes autoencoder architecture for credit card fraud detection. Autoencoders are deep learning models that can extract hidden data points from a given dataset. The goal is to identify fraudulent credit card transactions (Class 1) from genuine ones (Class 0) using a linear classifier.

## Autoencoder Architecture

Autoencoders consist of two components: an encoder and a decoder, both implemented as neural networks. The encoder extracts latent (hidden) representations from the input data, while the decoder reconstructs the original data from these representations.

![Autoencoder Architecture](https://github.com/rajaamani/Credit-Card-Fraud-Detection-/assets/101103515/da83865a-4105-4385-b09b-073b7406855c)

![Autoencoder--Architecture](https://github.com/rajaamani/Credit-Card-Fraud-Detection-/assets/101103515/62d5f871-d4be-4b04-bf29-68bbede1425b)

## Dataset

The dataset used in this project is `creditcard.csv`, which contains historical credit card transactions. The objective is to train an autoencoder to find hidden patterns and then apply a linear classifier to distinguish between fraudulent and genuine transactions.

## Dependencies

The following libraries are required to run this project:

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Keras

## Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/rajaamani/Credit-Card-Fraud-Detection-using-Autoencoders.git
   cd Credit-Card-Fraud-Detection-using-Autoencoders
   ```

2. Install the required dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn keras
   ```

3. Dataset can be found in https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

4. Open the Jupyter Notebook `Credit Card Fraud Detection - Autoencoder.ipynb` to run the project.

## Exploratory Data Analysis

- Explore statistics and class distribution in the dataset.
- Visualize data points using t-SNE for dimensionality reduction.

## Data Transformation

- Transform the "Time" attribute to represent time in hours.
- Perform data sampling to balance the dataset.

## Model Building

- Split the dataset into training and testing sets.
- Normalize and scale the features using a robust scaler.
- Create an autoencoder model for feature extraction.
- Train the autoencoder model on the scaled data.

## Encoding Data

- Use the trained autoencoder to encode the data.
- Visualize the encoded data using t-SNE.

## Linear Classifier

- Implement a linear classifier (Logistic Regression) on the encoded data.
- Evaluate the model's performance using classification metrics.

## Non-linear Classifier

- Implement a non-linear classifier (Support Vector Machine) on the original data.
- Evaluate the model's performance using classification metrics.

## Results

- Logistic Regression accuracy: 88%
- Support Vector Machine accuracy: 80%

This project demonstrates the effectiveness of using autoencoders for feature extraction in credit card fraud detection.

## Conclusion

In conclusion, this project showcases the power of autoencoders in improving the accuracy of credit card fraud detection. By encoding and transforming the data, we were able to achieve an 88% accuracy rate with a linear classifier, highlighting the potential benefits of this approach in real-world applications.


