---
layout: post
title: "Using Claude Code in Kaggle competition"
date:   2025-10-11 06:00:48 +1300
---

I used Claude Code to compete in a [Kaggle Competition](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) (From Jun 17, 2025 to Sep 16, 2025).

My setup was running python scripts on my local machine.
At the time, I found Claude Code produces relatively fewer errors creating and editing python scripts compared to python notebooks. It struggled to correctly edit the python content inside the ipynb file format, which led to more syntax errors and malformed code.

Using Opus plan mode, this means Opus 4.1 during planning, Sonnet for the actual coding to save Opus storage for the more complicated scenarios.

Here is what I found working well and not so well:

**Great at generating simple script**

When starting out a Kaggle competition, often you want to go through a basic exploratory data analysis or running a simple model as a baseline.

Claude Code is able to reduce a lot of this initial toil by generating these scripts or logic for you. For example:

Plotting the data

_prompt_
```
plot the target distribution in @data/train.csv, per target
```

_response_
```
● I'll help you plot the target distribution for each of the five targets in the training data. Let me start by exploring the data structure and then create visualizations for each target variable.

● Let me first read the train.csv file to understand its structure:

● Read(data/train.csv)
  ⎿  Read 10 lines (ctrl+o to expand)

● Now let me check how many rows are in the file and get a better understanding of the data:

● Bash(wc -l /home/wal8800/workspace/kaggle-notebook/data/train.csv)
  ⎿  7974 /home/wal8800/workspace/kaggle-notebook/data/train.csv
     /home/wal8800/.claude/shell-snapshots/snapshot-bash-1759630162041-t5c71e.sh: line 2858: alias: --: not found


<truncated>

╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Ready to code?                                                                                                                                                                                                                                                                         │
│                                                                                                                                                                                                                                                                                        │
│ Here is Claude's plan:                                                                                                                                                                                                                                                                 │
│ ╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮ │
│ │ Plan to Plot Target Distributions                                                                                                                                                                                                                                                  │ │
│ │                                                                                                                                                                                                                                                                                    │ │
│ │ I'll create a Python script to visualize the distribution of each target variable in the training data:                                                                                                                                                                            │ │
│ │                                                                                                                                                                                                                                                                                    │ │
│ │ 1. Create a visualization script (plot_target_distributions.py) that will:                                                                                                                                                                                                         │ │
│ │   - Load the train.csv file                                                                                                                                                                                                                                                        │ │
│ │   - Process the 5 target variables: Tg, FFV, Tc, Density, and Rg                                                                                                                                                                                                                   │ │
│ │   - Create distribution plots for each target using histograms and box plots                                                                                                                                                                                                       │ │
│ │   - Display summary statistics for each target                                                                                                                                                                                                                                     │ │
│ │   - Save the plots to a file                                                                                                                                                                                                                                                       │ │
│ │ 2. The script will generate:                                                                                                                                                                                                                                                       │ │
│ │   - A 2x3 subplot grid showing histograms for each target                                                                                                                                                                                                                          │ │
│ │   - A separate 1x5 subplot showing box plots for outlier detection                                                                                                                                                                                                                 │ │
│ │   - Summary statistics printed to console (count, mean, std, min/max, quartiles)                                                                                                                                                                                                   │ │
│ │   - Handle missing values appropriately                                                                                                                                                                                                                                            │ │
│ │ 3. Output:                                                                                                                                                                                                                                                                         │ │
│ │   - Save plots as target_distributions.png and target_boxplots.png                                                                                                                                                                                                                 │ │
│ │   - Display the plots if running interactively
```

**Great at reading unstructured text data, summarizing and answering question**

Researching on the state of the art approach often requires doing a web search and reading through papers.

Claude Code makes this research step a lot of easier since it can perform web search and read pdf file.

It does struggle to download the pdf directly from url link, so most of the time,  I had to download the file manually first.

Otherwise, once it reads the pdf, you can ask question about the content.

Using the [Polymer GNN paper](https://www.nature.com/articles/s41524-023-01034-3) as an example:

_prompt_

```
can you read polymerGNN.pdf, give me the high level summary first. I'm interested in the model architecture, how they combined the graph embedding with other descriptors.
```

_response_

```
High-Level Summary of PolymerGNN

PolymerGNN is a multitask machine learning architecture that uses graph neural networks (GNNs) to predict multiple polymer properties from monomer composition.
Model Architecture

The PolymerGNN architecture consists of three main components:
1. Molecular Embedding Block

    Uses Graph Neural Networks (GNNs) to encode each monomer as a molecular graph
    Employs a two-layer GNN architecture:
        First layer: Graph Attention Network (GAT)
        Second layer: GraphSAGE layer
    Separate GNN blocks for acids (Φa) and glycols (Φg)
    Uses Self-Attention Graph Pooling to produce graph-level embeddings
    Transforms molecular structures into fixed-size vector representations

[truncated for brevity]
```

Claude code is also great at diving into library codebase and understand how it works.
For example, we can clone the library repository then run /add-dir to include it as a working directory.

This allows Claude Code to have direct access to the actual implementation, builds up
the relevant context and provides the most accurate response.

_prompt_

```
how does the external descriptor if I use the rdkit normalised gets concatenated?
```

_response_

```
I'll search through the codebase to understand how external descriptors are concatenated when using RDKit normalized features.

Search(pattern: "external.*descriptor", path: "")

Search(pattern: "external", path: "")

Search(pattern: "rdkit.*normal", path: "")

Search(pattern: "concatenat", path: "")

Read(./chemprop/featurizers/molecule.py)

Search(pattern: "descriptor.*concatenat", path: "")

Read(./chemprop/data/datapoints.py:95,145)

Read(./chemprop/nn/message_passing/base.py:35,85)

[truncated for brevity]

Based on my examination of the codebase, when external descriptors are used with RDKit normalized features, they are concatenated in the following way:

For molecule-level external descriptors (V_d):

    External descriptors are concatenated to atom-level descriptors after message passing in chemprop/nn/message_passing/base.py:187

[truncated for brevity]
```

**Bad at more complex script or pipeline**

Claude Code lacks the awareness for some of the pitfalls that it can encounter. For example:

1. It tries to add try-except for error handling but this introduces more bugs. The script could swallow the error and continue with unexpected state.

2. Sometimes, the way it writes data preprocessing logic, it can introduce leakage from the training set into the test set. This means the changes it made appear to improve the performance (cross validation) but in reality, it was leaking the test set targets.

Claude Code seems more aware of common software development pitfalls than data science-specific ones. For example, if I asked it to create an entity that requires two database insert/update operations, it generally knows to wrap them in a database transaction to avoid inconsistent state. However, it lacks similar awareness for ML-specific issues like train-test leakage.

To improve, I tried adding instruction in CLAUDE.md but it didn't always listen to the instructions. What worked well was writing specific instructions in the prompt when doing specific task.

**Bad at analyzing relatively large CSV files**

Claude code is great at analyzing small amounts of data (less than 100 rows) that fit in a single read.

Once we are above 1000 rows, attempting to get it to perform zero shot analysis (identifying patterns directly from the raw data without intermediate processing), Claude code struggles.

It tries to create a python script to analyze but it generates a predetermined conclusion in the analyzed script. What this means, it writes a line `print("Recommends X Y Z")` within the script, before the script is executed and returns further summary/results.

To improve, I needed to prompt Claude code to take a multi step approach, generating summary and statistics then getting it to interpret the calculated values.

Also, making sure the context window in Claude Code is not near max or 80% (running /context to view the current context window usage), helps the right data/text to be at the forefront of the context window.

---

<br>

Overall, using Claude Code greatly reduces the toil when building models and experiments in a Kaggle competition. While there are areas that it struggles, it's often the more complex area where I need to spend more effort anyway to understand and fine-tune.

