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

The above calculates the entropy of a discrete probability distribution, where p1, p2, ..., p(n) are the probabilities of each of the n possible outcomes. The entropy is maximized when all the outcomes are equally likely (i.e., when the probability distribution is uniform), and it is minimized when one outcome has a probability of 1 and all the others have a probability of 0 (i.e., when the distribution is deterministic).


### Cross Entropy Loss

Cross-entropy loss (CE loss) is a measure of the difference between two probability distributions. Specifically, it measures the amount of information needed to represent the true probability distribution of the data (which is often unknown) using a model's predicted probability distribution. The CE loss is defined as the negative log-likelihood of the true distribution given the predicted distribution.

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

The code snippet provided above is a PyTorch implementation of the cross-entropy loss function. The function takes in two parameters: `y_pred`, which is a tensor representing the predicted class probabilities, and `y_true`, which is a tensor representing the true labels. 

We apply a logistic softmax operation to the predicted probabilities using the `F.log_softmax function`, which is a PyTorch implementation of the softmax function using the natural logarithm (`ln`).

The `F.one_hot` function is then used to convert the true labels into a one-hot encoding format, which is a binary vector where only the index corresponding to the true label is set to 1, and all other indices are set to 0. This allows us to select the probability of the correct class from the predicted probabilities by performing an element-wise multiplication of the one-hot encoded ground truth labels and the log of the predicted probabilities we got above.

The negative of the resulting tensor is summed across all elements, since we want to go in the direction of descent.

The mean of the resulting tensor is returned as the cross-entropy loss. The division by `y_pred.shape[0]` is included to normalize the loss by the batch size, which is the number of samples being processed in a single forward pass of a model.

### Use case: But how does this pertain to predicting the next word?

Let’s say we have a phrase “I like to study at Starbucks.”

The expression P(Starbucks | I like to study at) is a conditional probability, where each term represents the probability of the corresponding word given the previous context. These probabilities can be estimated using a language model trained on a large corpus of text data. The cross-entropy loss can then be used to evaluate how well the language model is able to predict the actual next word given the previous context.

In one case, a Hidden Markov Model (HMM) can model the underlying state sequence, which determines the probabilities of the observed words. The cross-entropy loss is then used to evaluate how well the HMM is able to predict the actual next word given the current state and the previous words in the sequence.

In another case, we can use masked language modeling to learn the distribution directly from the data. The masked language model (MLM) loss is a variant of the cross-entropy loss that only considers the predictions for the masked words, ignoring the predictions for the non-masked words in the input sequence. The MLM loss can be computed using the negative log-likelihood of the correct target word given the predicted probability distribution over the vocabulary.

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