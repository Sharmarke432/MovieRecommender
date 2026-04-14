# 🍿 Content-Based Movie Recommender System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458.svg)

## 📌 Overview
An end-to-end Machine Learning web application that recommends movies based on user selection. Built using the **MovieLens dataset**, this system utilizes **Natural Language Processing (NLP)** techniques—specifically **TF-IDF Vectorization** and **Cosine Similarity**—to analyze movie genres and find the closest matching films. 

The application is deployed with an interactive **Streamlit** frontend, featuring auto-complete search functionality and hybrid sorting (similarity score + community ratings) to ensure high-quality recommendations.

**[Live App Link]** *(Link to your Streamlit Cloud or Hugging Face Space here)*

---

## 📸 Application Demo
*(Take a quick screen recording or screenshot of your app working and place the image/gif here. E.g., `![Demo GIF](demo.gif)`)*

---

## Technical Implementation

### 1. Data Preprocessing (`Pandas`)
* **Data Integration:** Merged MovieLens `movies.csv` and `ratings.csv` to combine text data with user rating behaviors.
* **Quality Control:** Filtered out obscure movies (fewer than 50 ratings) to prevent highly similar but poorly-reviewed movies from skewing recommendations.
* **Text Cleaning:** Normalized text data by replacing pipe delimiters (`|`) and applying lowercase transformations to genre strings.

### 2. Machine Learning Logic (`Scikit-Learn`)
* **TF-IDF Vectorization:** Converted the cleaned textual genre data into numerical vectors. Rare genres are given higher weight, penalizing generic terms.
* **Cosine Similarity:** Computed the angular distance between the vectorized input movie and the rest of the dataset. To optimize memory usage (preventing 16GB RAM overflow), similarity vectors are computed **on-the-fly** at runtime rather than storing an $N \times N$ matrix.

### 3. Deployment (`Streamlit` & `AWS S3` / `HuggingFace`)
* Built an interactive frontend using Streamlit.
* Implemented searchable dropdowns (`st.selectbox` with `index=None`) for a seamless User Experience.

---
