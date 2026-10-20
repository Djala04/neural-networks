# CIFAR-10 Classification & Named Entity Recognition

The project has two parts: a custom CNN for CIFAR-10 image classification, and a comparison of BiLSTM and Transformer models for named entity recognition (NER) on WikiANN.

## Task 1: CIFAR-10 Image Classification

A CNN built from five **Intermediate Blocks** followed by an Output Block (channels 64 → 128 → 256 → 512). Each block runs three parallel convolution branches (1×1, 3×3, 5×5) and combines them with learned weights. The weights come from a global-average-pooled input passed through a fully connected layer.

- **Additions:** batch normalisation, GELU activation, MaxPool2d between stages, and dropout (p=0.2) in the Output Block
- **Training:** 200 epochs, batch size 256, SGD (momentum 0.9, weight decay 5e-4), cosine annealing from lr 0.05, label smoothing 0.1, gradient clipping at 1.0, CutMix and random erasing, seed 42
- **Result:** **93.3% test accuracy** (epoch 190)

## Task 2: Named Entity Recognition (WikiANN English)

Compares a **BiLSTM** and a **Transformer encoder** (both trained from scratch) on WikiANN English (PER, ORG, LOC; 20K train / 10K val / 10K test sentences). Results are averaged over 3 seeds (42, 123, 7) and evaluated with entity-level F1 (`seqeval`).

| Experiment | Question | Finding |
|------------|----------|---------|
| 1. Baseline | BiLSTM vs Transformer | BiLSTM wins: F1 **0.668** vs **0.407** |
| 2. Sentence length | Does the Transformer catch up on long inputs? | No, the BiLSTM leads in every length bin |
| 3. Positional encoding ablation | How much does PE matter? | Small drop (0.407 → 0.398) because sentences are short |
| 4. Dropout sweep | Best dropout rate (0.0 / 0.3 / 0.5)? | 0.5 was best for both models |

ORG was the hardest entity type for all models. Raising early-stopping patience from 5 to 15 improved Transformer F1 from 0.365 to 0.407.


## Authors

Marharyta Dzhafarova · Saif Ali Khan Mangan · Haadi Muhammad Iftikhar

## References

Key references include Vaswani et al. (2017), Yan et al. (2019), Yun et al. (2019, CutMix) and Srivastava et al. (2014). The full list is in the project report.
