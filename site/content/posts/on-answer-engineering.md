---
title: On Prompt Answer Engineering
date: 2023-04-29
description: "Some key points around expirements performed with the motive of getting rid of manual annotation"
draft: false
---
**The following is an overview of this, which I wrote with my colleagues at the University of Southern California **

The prompt learning paradigm has become a prominent approach in natural language processing, exhibiting its potential in both discriminative and generative tasks. This technique involves pre-training language models on large amounts of raw text and then introducing a new prompting function that allows the model to adapt to new situations with little or no labeled data, using few-shot or zero-shot learning. To perform prompt-based tuning, the model incorporates the input text into a cloze question and maps output words to labels using a verbalizer. Verbalizers can be either manually designed or automatically constructed, but manual verbalizers require domain-specific prior knowledge and human effort, while finding suitable label words automatically remains challenging.

On the other hand, Large Language Models (LLMs) have demonstrated their in-context learning capability, which is effective in few-shot and zero-shot scenarios. In this paper, we propose a method that combines ProtoVerb, a verbalizer that learns prototype vectors as verbalizers through contrastive learning, with in-context learned verbalizers from LLMs. This approach aims to enhance the generalizability of the verbalizer and reduce human labor. We conduct experiments on various topic classification tasks, and the results show that our approach is comparable to or outperforms both the automatic ProtoVerb and the manual ProtoVerb variations.

## Why was this an important problem to look into?
Architectures as they pertain to the tasks of improved generalizations on unseen data, faster convergance (PeFT has helped with this) and owning the weights are still being explored at the time of writing this. 

In context learning, as aforementioned, also has demonstrated success in generalizing to downstream tasks

## Why does in-context learning work?



# What is a Prototypical Space?


# What is the InfoNCE Loss Function?



# Why do we need verbalizers?



# How can ICL be trusted to replace manual annotators?



## Cases where generic prompt had an issue



# What are some future work?


