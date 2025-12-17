# Deep-class-aware-ood  
A class-aware cross-validation framework for estimating out-of-distribution (OOD) uncertainty thresholds in open-set recognition.

## 🌱 Introduction
This project explores how to reliably set uncertainty thresholds for **open-set recognition (OSR)** when only in-distribution labels are available. We propose a simple but effective **cross-class validation** procedure that uses held-out classes as stand-ins for unseen categories, enabling robust OOD threshold estimation without external datasets.

The project was originally developed as the final project for the “Statistical Machine Learning” graduate course at Sharif University of Technology.

## 🎯 Problem Setup
Open-set recognition requires a model to:
1. **Classify** known classes.
2. **Detect** inputs belonging to unseen classes.

A major challenge is choosing an uncertainty threshold without access to real OOD samples. Our method addresses this by simulating unseen classes using the seen classes themselves.

## 🔍 Method: Cross-Class Validation
For each class in the dataset:

1. **Hold out one class** as a proxy OOD class.  
2. **Compute uncertainty scores** for:
   - In-distribution samples (remaining classes)
   - Proxy OOD samples (held-out class)
3. **Find the threshold** that maximizes the F1 score for ID vs OOD separation.
4. **Repeat** for all classes and average the thresholds.

This approach is model-agnostic and works consistently across uncertainty scoring methods.

## 📐 Uncertainty Estimation Methods
We implemented and evaluated several approaches:

- **Softmax entropy**
- **Energy-based score**
- **Distance to class centroids** in embedding space
- **Distribution-of-distances entropy** (our method)
- **Bayesian nonparametrics** (e.g., BGMM with Dirichlet priors)

## 🧠 Embeddings & SSL Extension
All experiments use **VGG-19** pretrained on ImageNet-1K as the main feature extractor.

We also explore **self-supervised embeddings (e.g., DINO)** for OOD detection, using angular distance and centroid-based scoring. DINO’s embedding space has recently been interpreted as a **von Mises–Fisher mixture**, making it naturally aligned with centroid-based OOD scoring.

Experiments include:
- CIFAR-10 → SVHN using DINO  
- Oxford-102 Flowers (main dataset)

## 🧪 Results
- The **cross-class validation** method consistently found thresholds close to the optimal manually-tuned OOD threshold, across all uncertainty scoring methods.
- The limiting factor was **Bayesian nonparametric clustering** using BGMM; Gaussian assumptions did not match the geometry of high-dimensional embeddings.
- SSL embeddings show strong promise, especially with centroid-based and vMF-inspired scoring.

## 🚀 Future Work
We plan to continue this project by exploring:
- **Dirichlet-process von Mises–Fisher mixture models**  
- **SSL embeddings (DINO, iBOT, MoCo-v3)** for robust OOD scoring  
- Application to larger fine-grained datasets (CUB-200, iNaturalist, Fungi-2018)

## 📦 Code
All code is implemented in **PyTorch**.  
See the `SSL+OOD` Jupyter notebook for the self-supervised experiments.

## 📄 License
MIT License.

