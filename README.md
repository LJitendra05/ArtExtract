# 🎨 Art Extract: Deep Learning for Artistic Attribute Classification and Similarity Retrieval

## 📌 Project Overview

**Art Extract** is a deep learning-based framework developed for analyzing and retrieving meaningful visual information from artworks. The project is divided into two main tasks:

- **Task 1:** Classification of artistic attributes such as Style, Artist, and Genre using Convolutional-Recurrent Architectures  
- **Task 2:** Similarity Retrieval of paintings using deep visual feature embeddings  

The system leverages publicly available art datasets to extract stylistic and structural information from paintings.

---

# 🧩 Task 1: Convolutional-Recurrent Architecture for Artwork Classification

## 🎯 Objective

To build a deep learning model capable of classifying paintings based on:

- Artistic Style  
- Artist  
- Genre  
- Other visual attributes  

using a Convolutional-Recurrent Neural Network (CNN-RNN) architecture.

---

## 📂 Dataset Used

- **WikiArt Dataset (ArtGAN subset)**  
https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md  

Due to computational limitations, a subset of the WikiArt dataset was used instead of the full dataset.

---

## 🧠 Model Strategy

The classification model combines:

- **Convolutional Neural Network (CNN)**  
  - Extracts spatial features such as color distribution, texture and brushstroke patterns  

- **Recurrent Neural Network (RNN)**  
  - Captures high-level dependencies and structured visual relationships in extracted features  

This convolutional-recurrent architecture enables the model to learn both local and contextual visual patterns from artwork images.

---

## ⚙ Implementation

Task-1 was implemented using:

- **Google Colab (GPU Runtime)**

---

## 📊 Evaluation Metrics

Model performance was evaluated using:

- Precision  
- Recall  
- F1-score  
- Weighted Average Metrics  

These metrics were computed using a **Classification Report** to assess performance across stylistic classes.

---

## ⚠ Observations

The classification report indicates reduced performance for minority classes such as:

- Analytical Cubism  
- Pointillism  

This is primarily due to class imbalance within the selected subset of the WikiArt dataset.

The model tends to prioritize dominant styles such as:

- Impressionism  
- Realism  

resulting in undefined precision values for underrepresented classes. Weighted averages were therefore considered to account for dataset imbalance.

---

## 🚨 Outlier Detection

Outliers were identified as:

> Paintings whose predicted stylistic class significantly deviates from their assigned ground-truth label.

Such discrepancies may indicate visual overlap between artistic styles or potential dataset label noise.

---

# 🧩 Task 2: Painting Similarity Retrieval

## 🎯 Objective

To retrieve paintings with similar visual characteristics such as:

- Face orientation  
- Pose  
- Composition  
- Lighting  
- Scene structure  

using deep visual feature embeddings.

---

## 📂 Dataset Used

- **National Gallery of Art Open Dataset**  
https://github.com/NationalGalleryOfArt/opendata  

A subset of the NGA dataset was used for efficient feature extraction.

---

## ⚙ Implementation

Task-2 was implemented using:

- **VS Code (Local GPU Runtime – NVIDIA GTX 1050 Ti)**

---

## 🧠 Model Strategy

The similarity retrieval system follows a Content-Based Image Retrieval (CBIR) approach:

1. A pretrained **ResNet50 CNN** is used as a feature extractor  
2. Each painting is converted into a high-dimensional embedding vector  
3. Embeddings are normalized for cosine similarity comparison  
4. A **FAISS index** is used for efficient nearest-neighbour similarity search  

This embedding-based approach ensures visually similar paintings are mapped closer in the learned feature space.

---

## 📊 Evaluation Metrics

Since the NGA dataset subset does not contain labeled similarity annotations, evaluation was conducted using self-retrieval metrics, where the query image itself is treated as the ground-truth match.

The following metrics were used:

- Top-1 Accuracy  
- Recall@5  
- Mean Reciprocal Rank (MRR)  

---

## 📈 Results

The similarity retrieval model achieved:

- **Top-1 Accuracy = 1.0**  
- **Recall@5 = 1.0**  
- **MRR ≈ 0.999**

indicating that the learned deep feature embeddings effectively preserve visual similarity among paintings.

---

## 🖼 Sample Similarity Retrieval Result

Below is an example of Top-5 visually similar paintings retrieved for a given query image:

![Similarity Retrieval Example](results/sample_query.png)

---

# 🛠 Technologies Used

- Python  
- PyTorch  
- OpenCV  
- FAISS  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

# 📌 Note

Datasets used in this project are not included in the repository due to size constraints.Subset of that dataset is used.
Please refer to the dataset links provided above for downloading the required data.
