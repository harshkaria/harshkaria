---
title: Notes on Loss Functions
date: 2023-04-23
description: "Some blurbs on important evaluators in ML"
draft: false
---
The following is an overview of two common loss functions we commonly use in machine learning modeling. Machine learning is all about optimizing models to make accurate predictions on unseen data. One key aspect of this is defining a loss function, which measures the difference between the model's predictions and the true values. In this context, we often talk about two types of losses: classification loss and regression loss.

## Loss for Classification

### What is entropy?

Background on information theory:

- we can define information as the minimum number of bits to represent a message
- The measure of entropy is defined as I(p) = -(p1* log p1)+p2 * log(p2)+ … +p(n) log n

### Cross Entropy Loss

- Cross entropy loss is the measurement of the similarity of two distributions, commonly used for classification tasks

```python
import torch
import torch.nn.functional as F

def cross_entropy_loss(y_pred, y_true):
    """
    Computes the cross entropy loss between predicted logits and true labels.
    """
    log_softmax = F.log_softmax(y_pred, dim=1)
    loss = -torch.sum(log_softmax * F.one_hot(y_true, num_classes=y_pred.shape[1]))
    return loss / y_pred.shape[0]
```

We then take the negative of the softmax and one_hot the classes to select only the loss for the class at hand, and optimize against this.

### Use case: But how does this pertain to predicting the next word?

Let’s say we have a phrase “I like to study at Starbucks.” 

P(Starbucks | I like to study at) = P(I | Start) * P(Like | I, Start) * P(To | Like, I study) * P(Study | To, Like, I) * P(At | Study, To, Like, I) * P(Starbucks | At, Study, To, Like, I)

Each term in this expression represents the probability of the corresponding word given the previous context, and these probabilities can be estimated using a language model trained on a large corpus of text data. The cross-entropy loss is then used to evaluate how well the language model is able to predict the actual next word given the previous context.

In one case, a HMM would model the underlying state sequence, which determines the probabilities of the observed words. The cross-entropy loss would then be used to evaluate how well the HMM is able to predict the actual next word given the current state and the previous words in the sequence.

In another case, we would use masked language modeling to learn the distribution directly from the data. We often use a variant of the cross-entropy loss called the masked language model (MLM) loss. 

The MLM loss only considers the predictions for the masked words, ignoring the predictions for the non-masked words in the input sequence. The MLM loss can be computed using the negative log-likelihood of the correct target word given the predicted probability distribution over the vocabulary.

Each word in the vocabulary can be considered as a separate class that the model needs to predict the probability distribution over. The goal of the model is to predict the correct word (or the masked token) at each position in the sequence, given the context of the surrounding words.

## Losses for Regression:

Common losses for regression include MSE, RMSE, MAV. These predict continuous values, wheras cross entropy losses predict discrete values. 

### MSE

The mean squared error (MSE) is a loss function used to measure the average squared difference between the predicted and true values in a regression problem. Mathematically, the MSE can be expressed as:

MSE = (1/n) * sum((y_pred - y_true)^2)

where n is the number of samples in the dataset, y_pred is the predicted value, and y_true is the true value.

### Use case: Stock Value Prediction

Given a dataset of stock market data over a year how would we use MSE as the objective function to predict ending monthly price?

- First, we would prep the dataset. Assuming we have independent features like moving average, volume, starting price, etc, we would split this into a 12 batches of approx. 29 days each, and predicting the 30th day.
- We would then create a LSTM network tuning the learning rate, number of layers, number of hidden dimensions and after the forward pass, use MSE to calculate the difference between the predicted price and the actual price on the 30th day.
- We would then perform back propagation until convergence.

The main purpose is to show that unlike cross-entropy loss, where we were predicting a discrete variable such as a drink or a word, with regressive tasks, we use functions such as MSE to get fidelity over continuous values. 

## Summary

- We use objective loss functions to optimize our networks against ground truth values.
- These loss functions are differentiable by definition, and we go in the rate of steepest descent.
- For classification, we use logarithmic loss functions such as cross entropy, which measures the difference between two distributions
- For regression, we use quadratic loss functions such as MSE when predicting a continuous values.