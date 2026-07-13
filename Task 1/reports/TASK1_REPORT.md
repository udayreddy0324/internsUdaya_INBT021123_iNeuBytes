# Task 1 Report: Computer Vision Using CNN Models

**Course:** Artificial Intelligence (AIINB20726)  
**Intern:** _Add name_  
**Date:** _Add date_

## Abstract

This study establishes an AlexNet-inspired CIFAR-10 baseline and evaluates regularization, augmentation, optimization, and architectural changes under a fixed split, seed, and training budget. Replace this paragraph after training with the principal measured result and trade-off verdict.

## 1. Method

CIFAR-10 contains 60,000 32×32 RGB images across ten balanced classes. The official 10,000-image test set was retained; a seeded permutation split the original training data into 45,000 training and 5,000 validation samples. Pixels were normalized to [0,1]. Every controlled experiment used seed 20726, batch size 128, and 25 epochs.

## 2. Baseline

The baseline uses three convolutional blocks with increasing filters (64, 128, 256), ReLU activations, max pooling, global average pooling, and a dense classifier. It contains no augmentation, dropout, batch normalization, or L2 regularization.

Add the baseline architecture diagram/summary and its training/validation curves here.

## 3. Controlled Experiments

Insert `artifacts/master_experiment_table.csv` as the master table. For each group, state a hypothesis, compare only the relevant rows, and write a 1–2 sentence evidence-based conclusion.

### 3.1 Regularization

_Which single technique reduced the train–validation gap most? Did accuracy fall?_

### 3.2 Data Augmentation

_Compare light, moderate, and aggressive augmentation. Explain why large transformations can distort 32×32 images._

### 3.3 Optimization

_Identify the best optimizer/learning-rate combination using measured validation evidence._

### 3.4 Architecture

_Report accuracy gained, additional parameters, and additional training time._

## 4. Final Customized CNN

List only changes supported by Part B, naming the experiment and measured evidence that justified each change. Compare baseline and final learning curves.

## 5. Performance Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Parameters | Training time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| Final | TBD | TBD | TBD | TBD | TBD | TBD |

## 6. Confusion-Matrix Analysis

Insert baseline and final confusion matrices. Report the 2–3 largest bidirectional off-diagonal class-pair totals. Discuss whether the initial visual-similarity hypothesis was supported and quantify the improvement.

## 7. Accuracy–Cost Trade-off

Use percentage points for accuracy:

`gain_per_million_parameters = (final_accuracy - baseline_accuracy) × 100 / ((final_parameters - baseline_parameters) / 1,000,000)`

Also report accuracy gained per extra training minute. If the final model has fewer parameters or trains faster, describe it as a Pareto improvement rather than dividing by a negative cost. Finish with a clear verdict.

## 8. Conclusion

Summarize the baseline, strongest controlled result, final improvement, confused-pair change, and whether the extra computational cost was justified. Explicitly state whether the ≥70% baseline and ≥3 percentage-point improvement targets were met.

