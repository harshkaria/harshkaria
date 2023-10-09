---
title: Why do LLMs play well with in context learning?
date: 2023-10-08
description: "Exploring findings related to what makes large language models effective to prompts"
draft: false
---

LLMs have become a common tool in various tasks such as generation and code completion. Models from OpenAI, Cohere, Anthropic, and others are widely used. In some cases, it makes sense to have your own model weights.

While fine-tuning has been a popular approach for transferring to new tasks (Devlin et al., 2019), it is impractical to fine-tune the entire model for simple tasks. In context learning (ICL) is a form of pruning the latent space of these models to focus on a specific use case.

## What is Prompt Engineering?
An example of a ICL based prompt for classifying NBA headlines is as follows:

- "LeBron James Leads Lakers to Victory with Triple-Double Performance" - **Positive**
- "Ja Morant ruled out for playoffs due to off-court misconduct" - **Negative**
- "Rookie Phenom Sets New Scoring Record in NBA Debut" - **Positive**

Chris Paul's Leadership and Playmaking Skills Propel Suns to Conference Finals **________**

## Why does ICL work?

Research has been conducted to correlate in context learning with term frequencies present in the training data (Razeghi et al., 2022) in a theoretical space. Practically, Min et al. (2022) identify four key concepts in ICL:

1. Input-label mapping
2. Distribution of inputs
3. Label Space
4. Format of the input

### Accurate Label Assignments Do Not Matter

Min et al. (2022) find that the accuracy or ground truth of the input-label mapping does not matter much. They performed experiments on LLMs of various sizes and training methods (decoder-only) in three different settings:

1. Zero-shot setting: An input X is fed in to determine class Y without any preceding examples. (A similar input can be the standalone Chris Paul headline above.)
2. Gold label setting: Prompts are fed in with the correct classes associated with them from a class set C (where class Y exists in C), like the three headlines above.
3. Random label setting: A class Y (which exists in class set C) is randomly sampled and assigned to the input, regardless of potential inaccuracy, and fed into the model.

The results show that having label mapping (settings 2 and 3) significantly contributes to correctly classifying a new input. However, there is only marginal improvement between settings 2 and 3, suggesting that the accuracy of the label itself does not matter. This contrasts with supervised fine-tuning, where backpropagation relies on the ground truth label.

Notably, performance improvements degrade with more than k = 8 examples with ICL, whereas in supervised learning, more data usually leads to better performance.

## Contributing Factors

### The inputs being in-distribution of the LLM training data

LLMs are trained on large corpora of data, so they tend to condition the latent space based on the provided input. For example, when Min et al. explored ICL based on data not present in the LLMs' training corpus, performance dropped significantly. This reinforces the importance of using in-distribution inputs in demonstrations to achieve performance gains.

### Label Space being Present

Min et al. trained both direct models and channel models. Direct models generate responses directly based on the input they receive, while channel models (also known as dialogue models or context models) consider the entire conversational history or context to generate responses (e.g., GPT-3.5 vs ChatGPT).

They found that removing the label space has a significant impact on the accuracy of direct models, which is expected, while channel models show little performance degradation without the label space.

### Formatting of the Prompt

In the example headlines provided, the format is {headline} - {sentiment}. Min et al. found that while the accuracy of the input-label mapping may not matter, the format is crucial. Removing this format significantly drops ICL accuracy.

### Meta Training with ICL Objective

Meta-training with an in-context learning objective involves exposing the model to a diverse range of language-related tasks, such as text classification, sentiment analysis, or named entity recognition. The model learns to extract generalizable patterns and knowledge from these tasks, enabling it to perform well on new, unseen tasks in language processing.

The in-context learning objective is incorporated into the training process by providing contextual cues or prompts that guide the model's learning. During training, the model is exposed to a sequence of tasks, each including a context and a target. The model is trained to predict the target given the context, with the objective of minimizing the prediction error. Through repeated training on such tasks, the model learns to extract relevant information from the context and make accurate predictions, which can then be transferred to new, similar tasks.

Min et al. found that LLMs trained specifically with this objective can be thought of as instruction-following LLMs, where the template matters more than the accuracy of the mappings. In this setting, either the label set or the input set can be fed independently, and accuracy can still be preserved. So, a random context with a correct label or a correct context with a random label will produce an accurate output as long as the template is preserved. This means that performing inference on models trained with this objective exploits the template/format rather than the context itself.

## Conclusion

ICL leverages the existing capabilities of LLMs without fundamentally changing the underlying model architecture. It allows the model to focus on the specific task at hand, enhancing performance. Moreover, it highlights the potential of utilizing the latent knowledge and patterns within large language models. The key factors in ICL are choosing the right input-label mapping, distribution of inputs, label space, and format of the input.

## References

Min, Sewon, et al. "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?." arXiv preprint arXiv:2202.12837 (2022).

Yasaman Razeghi, Robert L Logan IV, Matt Gardner, and Sameer Singh. "Impact of pretraining term frequencies on few-shot reasoning." arXiv preprint arXiv:2202.07206 (2022).