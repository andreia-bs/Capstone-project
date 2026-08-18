# Capstone-project
## Section 1: Project Overview
### BBO capstone project and its purpose?
This project tackles a Black-Box Optimisation (BBO) challenge across eight unknown functions. Each function accepts a vector of inputs and returns a scalar output, but its internal mechanics are completely hidden — there is no formula, no gradient, and no closed-form solution. The only way to learn about each function is to query it and observe the result.

The goal is to maximise each function's output by strategically selecting query points across three modules (12–14), with one new query per function per module.

### Why does this matter in real-world ML?
BBO is fundamental to modern machine learning. Every time a practitioner tunes hyperparameters they are solving a black-box optimisation problem. The function mapping hyperparameters to model performance is unknown, expensive to evaluate, and potentially non-convex with many local optima. The same challenge appears in drug discovery, materials science, robotics control, and financial portfolio optimisation. This project builds the core intuition and practical skills needed to navigate these settings systematically.

Career relevance
Developing rigorous BBO skills directly supports roles in ML engineering, data science, and research. The ability to optimise unknown systems efficiently — without wasting expensive evaluations — is a differentiating skill when deploying models in production, running A/B tests, or designing experiments under resource constraints. As a clinical research fellow, strengthening these skills will enable me to succeed when it comes to critical data analysis, exploration and designing research projects.

 

## Section 2: Inputs and Outputs
### Input format
Each query is submitted as a hyphen-separated string of values, where every value begins with 0 and is specified to six decimal places.

All input values are constrained to [0,1]. The eight functions span dimensionalities from 2D to 8D, with initial datasets ranging from 10 to 40 points.

### Expected output
Each query returns a single scalar value — the function's output at the queried point. Outputs vary significantly in scale and sign across functions. We may find some “output” clusters:
Tiny positive (near-zero) values: function 1
Moderate positive values: functions 2, 7, and 8
Negative values: functions 3, 4, and 6
Large positive values: function 5.
 

## Section 3: Challenge Objectives
### Goal: Maximise all eight functions
All eight tasks are framed as maximisation problems. Where the natural objective is minimisation (e.g. side effects in drug discovery, penalty scores in recipe optimisation), the output is negated so that higher always means better.

### Constraints
The main constraints associated with the BBO capstone project are related to:

Limited queries per functions: only one query per function per module;
Response delay: the results from a specific query are only returned up to one week after – one module later – without real-time feedback;
Unknown function structure: having no gradient nor formula available.
The combination of a tight query budget and unknown structure makes this a genuine sample-efficiency challenge - every query must be justified.

 

## Section 4: Technical Approach
### Evolution across Modules 12–14 – methods and heuristics used
Module 12 – Ranked interpolation: With no prior feedback, the strategy relied on ranking initial outputs, identifying the top-performing points, and computing weighted interpolations biased toward the best observed inputs. This is conceptually equivalent to a manual approximation of gradient ascent without a formal model.

Module 13 – Feedback correction: First query results revealed mixed outcomes – four functions improved, four regressed. Functions that improved (functions 1, 6-8) continued in the same direction; functions that regressed (functions 2, 3, and 5) were corrected by returning toward previous best points. This introduced function-specific recalibration based on observed outputs rather than assumed gradients.

Module 14 – Micro-exploitation: Persistent oscillation across multiple functions -  overshooting, over-correcting, overshooting again – prompted a shift to very small step sizes around confirmed best points. Step sizes are now heuristically halved at each correction, mimicking a decaying learning rate.

### Planned: Gaussian Process surrogate
The next evolution is fitting a Gaussian Process (GP) surrogate.

A kernel SVM (RBF kernel) will also be explored to classify input space into high/low performance regions, particularly for functions with complex, non-linear boundaries (functions 1, 4, and 8).

### Exploration–exploitation balance
The strategy has shifted progressively from 70/30 exploit/explore in Module 12 to 95/5 in Module 14. With only one query per function per module, the opportunity cost of exploration is extremely high. However, function 4's repeated near-identical outputs signal a potential local optimum – a future exploratory query in a new region is planned once the GP surrogate confirms the local landscape is flat.