Go to the website: https://chili-disease-detection-system-4e7nhwgappvctdzaodnnbna.streamlit.app/


# Chili-disease-detection-system

This project aims to detect diseases in chili plant leaves using advanced Vision Transformer (ViT) models. It classifies images into three categories: **Aphids**, **MitesThrips**, and **Healthy**, and provides remedies for detected diseases. The solution is deployed as a simple and interactive **Streamlit web app**.

---

## 🧠 Models Used

The application supports multiple Vision Transformer architectures:
- `vit_base_patch16_224`
- `convit_base`

Users can choose the desired model at runtime via the Streamlit interface.

---

## 🔍 Features

- 🖼️ Upload chili leaf images for real-time disease prediction
- 🤖 Select between ViT-based models for inference
- 📊 Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, AUC-ROC, t-SNE
- 🧪 Visualizations of first-layer filters and activations
- 💡 Displays remedies for identified diseases (Aphids or MitesThrips)
- 📈 Cosine Annealing LR scheduling with early stopping
- 🧩 Modular, class-based training and evaluation framework

---

## 📂 Dataset

The dataset contains labeled images of chili leaves in three categories:
- `Aphids`
- `MitesThrips`
- `Healthy`

Preprocessing:
- Resized to 224x224
- Augmentations applied (for fine-tuning)

---

## 🛠️ Tech Stack

| Component          | Technology         |
|-------------------|--------------------|
| Programming       | Python             |
| Framework         | PyTorch, TIMM      |
| Visualization     | Matplotlib, Seaborn, scikit-learn |
| Web App           | Streamlit          |
| Deployment        | Local / Cloud      |
