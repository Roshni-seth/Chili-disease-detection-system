# Chili-disease-detection-system

This project presents a robust deep learning system to classify chili leaf images into Aphids, MitesThrips, and Healthy categories using state-of-the-art Vision Transformer (ViT) models. The app is built with PyTorch, TIMM, and deployed via Streamlit, offering real-time predictions and remedies.

Features
Classification into 3 classes: Aphids, MitesThrips, and Healthy

Multiple Vision Transformer architectures: vit_base_patch16_224, convit_base, deit_base, and others

Evaluation with:

Confusion Matrix

Accuracy, Precision, Recall, F1-score

AUC-ROC Curve

t-SNE visualization

First-layer filter & activation visualizations

Fine-tuning with:

Layer freezing and custom learning rates

CosineAnnealingLR learning rate scheduling

Early stopping

Image preprocessing and augmentation

Model selection interface + remedy suggestions in the Streamlit app

Tech Stack
Component	Technology Used
Language	Python
Deep Learning	PyTorch
Vision Models	TIMM (ViT, ConViT, DeiT, MobileViT, etc)
Visualization	Matplotlib, Seaborn, Sklearn
Web Interface	Streamlit
Dataset	Custom chili leaf dataset (3 classes)

Models Used
The following ViT-based architectures were trained and evaluated:

vit_base_patch16_224

vit_small_patch16_224

convit_base

convit_small

deit_base_patch16_224

mobilevitv2_100

efficientvit_b1

maxvit_small_tf_224

Each model was fine-tuned using:

Custom classifier heads

Mixed learning rates (e.g., 0.00001 for backbone, 0.001 for head)

Layer freezing and unfreezing strategies

Evaluation Metrics
Training/Validation loss and accuracy curves

Precision, Recall, F1-score

Confusion Matrix

AUC-ROC curves (for multi-class)

t-SNE plots to visualize feature space

Visualization of first-layer filters and activations

Model Training Strategy
Used feature extraction and fine-tuning strategies

CosineAnnealingLR scheduler with early stopping

Dropout and strong data augmentation to avoid overfitting

Ability to control:

Dropout rate

Activation function

Additional MLP layers

Normalization layers (BatchNorm/LayerNorm)

Streamlit App
The Streamlit web app allows:

Uploading a chili leaf image

Selecting ViT model for prediction

Viewing class prediction + disease remedy

Simple UI for real-world usage
