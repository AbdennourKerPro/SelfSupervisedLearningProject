# Self-Supervised Learning for Medical Image Analysis: Barlow Twins and SimCLR on MedMNIST

## Project Overview
This project explores **self-supervised learning (SSL)** techniques for medical image analysis, focusing on two state-of-the-art methods: **Barlow Twins** [1] and **SimCLR** [2]. The primary goal is to investigate whether these SSL approaches can learn useful image representations from medical datasets, specifically from **MedMNIST** [3], and to evaluate their effectiveness compared to models trained from scratch without pretraining.  

We implement both methods, apply them to multiple datasets from MedMNIST, and analyze their performance through **fine-tuning experiments** on classification tasks.  

#### Credits : Project done in colaboration with [Daniel Akbarinia](https://github.com/Daniel34990) and [Judith Amigo](https://github.com/crocojude).

---

## Motivation
Medical image datasets are often limited in size and expensive to annotate. Self-supervised methods offer a promising approach by leveraging large quantities of unlabeled data to learn transferable representations. Once pretrained, these models can be fine-tuned on smaller, labeled datasets, potentially improving performance and reducing the need for extensive annotations.

---

## Methodology
1. **Barlow Twins**  
   - A redundancy-reduction method that encourages embeddings of two augmented views of the same image to be highly correlated while minimizing redundancy between dimensions.  
   - Objective:  

     $$\mathcal{L} = \sum_i (1 - C_{ii})^2 + \lambda \sum_{i} \sum_{j \neq i} C_{ij}^2$$

     where $C$ is the cross-correlation matrix between embeddings.

2. **SimCLR**  
   - A contrastive learning method that maximizes agreement between augmented views of the same image via a contrastive loss.  
   - Relies on a large batch size and normalized temperature-scaled cross-entropy loss (NT-Xent).
    We define the cosine similarity between two embeddings $z_i$ and $z_j$ as:

$$
\text{sim}(z_i, z_j) = \frac{z_i \cdot z_j}{\|z_i\| \, \|z_j\|}
$$

The loss for a positive pair $(i,j)$ is given by:

$$ \ell_{i,j} = - \log \frac{\exp\left(\text{sim}(z_i, z_j)/\tau\right)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp\left(\text{sim}(z_i, z_k)/\tau\right)} $$

where:
- $\tau$ is the **temperature**,
- the numerator corresponds to the similarity between the two positive views,
- the denominator normalizes over all other negative pairs in the batch.

The final loss is averaged over all positive pairs:

$$ \mathcal{L} = \frac{1}{2N} \sum_{i=1}^{2N} \ell_{i,j(i)} $$

3. **Datasets: MedMNIST**  
   - A collection of lightweight benchmark datasets designed for biomedical image classification.  
   - Includes diverse modalities (X-ray, MRI, microscopy, histopathology, etc.) in a unified format.  
   - Examples: PathMNIST, ChestMNIST, DermaMNIST.  

4. **Evaluation**  
   - Pretrain using Barlow Twins and SimCLR on unlabeled MedMNIST data.  
   - Fine-tune the pretrained models on supervised classification tasks.  
   - Compare results with non-pretrained baselines (random initialization).  

---

## Expected Contributions
- Demonstrate whether Barlow Twins and SimCLR can improve downstream performance on medical image datasets.  
- Provide a systematic evaluation across multiple MedMNIST subsets.  
- Highlight the potential of SSL for reducing reliance on labeled medical data.  

---

## References
[1] Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). **Barlow Twins: Self-Supervised Learning via Redundancy Reduction.** *International Conference on Machine Learning (ICML).* [Paper](https://arxiv.org/abs/2103.03230)  

[2] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). **A Simple Framework for Contrastive Learning of Visual Representations (SimCLR).** *International Conference on Machine Learning (ICML).* [Paper](https://arxiv.org/abs/2002.05709)  

[3] Yang, J., Shi, R., Ni, B., Li, J., & Zhou, Z. (2021). **MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification.** *Scientific Data.* [MedMNIST Website](https://medmnist.com/)  

