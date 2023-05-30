# Task 8 - Next Word Prediction

This repository contains the implementation of a Next Word Prediction model. The model aims to predict the most probable word that follows a given context. The project was developed as part of an internship task assigned by Lets Grow More.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Next Word Prediction model utilizes natural language processing and machine learning techniques to anticipate the next word in a sentence or phrase. The model is trained on a large dataset of text, allowing it to learn the statistical patterns and relationships between words.

## Model Architecture

The Next Word Prediction model is based on a recurrent neural network (RNN) architecture. Specifically, it utilizes a Long Short-Term Memory (LSTM) network, a type of RNN that excels at capturing long-term dependencies in sequential data. The LSTM network is trained on input sequences of words and learns to predict the next word in the sequence.

## Dataset

To train the Next Word Prediction model, a large corpus of text data is required. The dataset used in this project consists of diverse sources, such as books, articles, and online text. It is important to note that the dataset should be representative of the language patterns and contexts relevant to the task.

## Data Preprocessing

Before training the model, the dataset undergoes several preprocessing steps. This includes tokenization, where each word is separated into individual tokens, and creating input-output pairs by sliding a window over the tokenized text. Additionally, the dataset is encoded to numerical representations to facilitate training.

## Model Training

The Next Word Prediction model is trained using the tokenized and encoded dataset. During training, the LSTM network learns to predict the next word in a sequence given the previous words as input. The model's parameters are adjusted iteratively using backpropagation and gradient descent, optimizing the network's ability to predict the next word accurately.

## Evaluation

To evaluate the performance of the Next Word Prediction model, various metrics are considered. Common evaluation metrics include perplexity, which measures how well the model predicts the test data, and accuracy, which calculates the percentage of correctly predicted next words. The model's performance can be further analyzed by examining the predictions on specific examples and assessing the coherence and relevance of the generated text.

## Usage

To use the Next Word Prediction model, follow these steps:

1. Clone the repository
2. Install the required dependencies
3. Prepare your own dataset or use the existing dataset provided in the repository.
4. Preprocess the dataset by tokenizing and encoding the text.
5. Train the Next Word Prediction model by running `train.py`.
6. Evaluate the trained model by running `evaluate.py`.
7. Use the trained model to predict the next word given a context by running `predict.py`.

## Contributing

Contributions to the Next Word Prediction model are welcome! If you have any ideas, suggestions, or bug fixes, feel free to submit a pull request. Please ensure that your contributions align with the repository's code of conduct


## Contact
For any questions, feedback, or collaborations, feel free to reach out to me. Connect with me on LinkedIn - jash thakar or email at - jash.thakar99@gmail.com 


