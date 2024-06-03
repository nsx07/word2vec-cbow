# CBOW Model for Word Embeddings

This repository contains a Python implementation of the Continuous Bag of Words (CBOW) model for learning word embeddings. This implementation is intended for a college homework assignment.

## Table of Contents
- [Introduction](#introduction)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Training Results](#training-results)
- [References](#references)

## Introduction

The Continuous Bag of Words (CBOW) model is a neural network-based approach for learning word embeddings. In CBOW, the context words are used to predict the target word. This implementation allows you to train a CBOW model on a given text corpus and learn embeddings for each word in the vocabulary.

## Implementation Details

The CBOW model in this repository is implemented in Python using NumPy. The key components of the implementation include:

- **Sentence Separation:** The input text is split into sentences based on punctuation.
- **Vocabulary Creation:** A vocabulary is created from the training data, mapping each word to a unique index.
- **One-Hot Encoding:** Words are converted into one-hot encoded vectors.
- **Training Data Encoding:** The training data is encoded into pairs of target words and their context words.
- **Weight Initialization:** The weight matrices (embeddings and output weights) are initialized with random values.
- **Forward Pass:** The average context vector is computed, and softmax is applied to obtain predicted probabilities.
- **Loss Calculation:** The cross-entropy loss is computed.
- **Backward Pass and Weight Update:** The gradients are calculated and the weights are updated using gradient descent.

## Usage

To use the CBOW model, follow these steps:

1. **Initialize the Model:**
   ```python
   cbow = CBOW(embedding_dim=10, window_size=2, epochs=100, learning_rate=0.01)
   ```
  
  2. ***Train the model***
      ```python
      text = "This is an example text for training. This text will be used to train the CBOW model."
      cbow.train(text)
      ```

  3. ***Print the learned embeddings***
      ```python
      cbow.print_embeddings()
      ```

## Training Results
  After training the model, you can visualize the training loss over epochs and the learned word embeddings in a 3D space. Below are placeholders for these visualizations.

### Loss Over Epochs
  ![Loss](https://storage.agendahub.app/wwwroot/w2v/loss.png)

### 3D Visualization of Word Embeddings (picture, 3D only on Colab)
  ![Emebddings](https://storage.agendahub.app/wwwroot/w2v/plot.png)
  
    
