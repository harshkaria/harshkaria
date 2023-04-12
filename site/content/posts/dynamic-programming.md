---
title: Dynamic Programming Learnings.
date: 2022-12-12
description: "Some key mental models for DP when it comes to writing algorithms"
---

# What is Dynamic Programming?
Dynamic programming is a way to save space! Imagine computing 1000 recursive calls for every operation, and then running into (1-p) of those recursive calls during another operation. Saving those (1-p) calls using memoization is a very helpful technique and caches these operations. 

## Case in Point: Fibonacci Sequence


