# RetinaGuard — Automated Diabetic Retinopathy Screening System

**A clinical AI application for early detection and severity staging of Diabetic Retinopathy from retinal fundus images using Transfer Learning and Explainable AI.**

---

## The Business Problem

Diabetic Retinopathy (DR) is a microvascular complication of diabetes mellitus and one of the leading causes of preventable blindness in working-age adults worldwide. The progression of DR is insidious — it remains clinically asymptomatic in its early and moderate stages, meaning patients experience no pain or noticeable vision loss until irreversible damage has occurred.

The global burden is severe, but the impact is disproportionately felt in developing healthcare ecosystems. India, home to over 77 million diagnosed diabetics — the second-largest diabetic population on Earth — faces a structural crisis: the ratio of certified ophthalmologists to rural patients is critically inadequate. The consequence is that a vast majority of high-risk individuals never receive a timely retinal examination. By the time a diagnosis is made, sight-threatening proliferative disease has often already developed.

The clinical and economic implications are significant. Blindness from DR leads to loss of productive employment, increased long-term dependency on healthcare infrastructure, and diminished quality of life. The World Health Organization estimates that 80 percent of DR-related vision loss is preventable through early intervention — specifically, through annual retinal screening and timely laser photocoagulation or anti-VEGF therapy when disease is detected at a reversible stage.

The barrier is not treatment — it is access to accurate, timely, and scalable screening.

**RetinaGuard was built to address this gap directly.** It provides a deployable, AI-powered clinical decision support tool that enables non-specialist healthcare workers — nurses, general practitioners, and medical officers in underserved settings — to perform meaningful, evidence-based retinal triage at the point of care, without requiring a specialist on-site.

---

## Project Overview

RetinaGuard is a full-stack medical AI application that accepts a retinal fundus photograph as input and returns a clinical severity classification across five standardized stages of Diabetic Retinopathy. In addition to the diagnosis, the system generates a visual explanation of the AI's decision using Gradient-weighted Class Activation Mapping (Grad-CAM), allowing clinicians to verify which regions of the retina informed the model's output.

The system also integrates patient clinical metadata — specifically, glycated hemoglobin (HbA1c) levels and diabetes duration — to generate a context-aware triage recommendation and a downloadable clinical report in PDF format.

The application is deployed as a cloud-accessible web portal, requiring no local installation on the end user's device.

---

## Core Technical Capabilities

- Five-class severity classification: No DR, Mild, Moderate, Severe, and Proliferative Diabetic Retinopathy
- Transfer Learning using a pre-trained EfficientNetB3 architecture fine-tuned on clinical retinal image data
- Grad-CAM visual explainability producing class-discriminative heatmaps overlaid on the source image
- Patient-contextualized triage logic combining model output with HbA1c and diabetes duration
- One-click downloadable PDF clinical report formatted for handoff to a specialist
- Cloud-deployed via Streamlit, accessible from any modern browser without software installation
- Docker-containerized for consistent execution across environments

---

## System Architecture

RetinaGuard follows a monolithic web application architecture. All inference, explainability computation, and report generation occur within a single Python process, eliminating the latency of inter-service communication and simplifying deployment.

The data flow through the system is as follows:

```
Retinal Fundus Image (JPEG / PNG)
           |
           v
  Image Preprocessing Layer
  - Resize to 224 x 224 pixels
  - Normalize pixel intensities to [0, 1]
           |
           v
  EfficientNetB3 Feature Extraction
  (12 million parameters, ImageNet pre-trained, DR fine-tuned)
           |
           v
  Custom Classification Head
  - GlobalAveragePooling2D
  - Dense(256, ReLU)
  - BatchNormalization
  - Dropout(0.5)
  - Dense(5, Softmax)
           |
     ------+------
     |            |
     v            v
Softmax        Grad-CAM
Prediction     Explainability
(5-class       (GradientTape
probability)    heatmap)
     |            |
     v            v
Triage Logic + Patient Metadata
  - HbA1c level
  - Years with diabetes
  - Patient age
           |
           v
  Clinical Dashboard + PDF Report
```

---

## Model Architecture and Training Strategy

### Base Model

The project uses **EfficientNetB3**, a convolutional neural network developed by Google Brain and introduced in the paper *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks* (Tan & Le, 2019). EfficientNet employs a compound scaling methodology that simultaneously scales network width, depth, and input resolution using a fixed set of scaling coefficients. This approach consistently outperforms manually scaled architectures of equivalent parameter count.

EfficientNetB3 was selected over alternatives such as VGG16, ResNet50, or InceptionV3 for the following reasons:

- It achieves higher accuracy at a lower parameter count, making inference faster and deployable on resource-constrained servers
- Its compound scaling is particularly beneficial for fine-grained medical image features, where resolution matters
- Pre-trained ImageNet weights provide a strong initialisation for the feature extraction layers, reducing the training data requirement substantially

### Training Strategy

Training proceeded in two distinct phases, both of which are industry-standard practice for Transfer Learning in medical imaging.

**Phase 1 — Feature Extraction (Frozen Base):**
The EfficientNetB3 base layers were frozen, preserving the ImageNet-derived weights. Only the custom classification head was trained. This prevents catastrophic forgetting of foundational visual features — edges, textures, and shapes — during the early epochs when gradients are large and unstable.

**Phase 2 — Fine-Tuning (Unfrozen Base):**
After the classification head converged, the base model was unfrozen and the entire network was re-trained at a significantly reduced learning rate (1e-4, one-tenth of the initial rate). This allows the model to subtly adjust its lower-level representations toward the specific visual patterns of diabetic retinal pathology — microaneurysms, hard exudates, neovascularisation, and vitreous haemorrhage — without overwriting the general feature hierarchy.

### Training Callbacks

Three callbacks were employed during training:

- **EarlyStopping** on validation loss with a patience of 10 epochs, restoring the best weights at termination
- **ReduceLROnPlateau** reducing the learning rate by a factor of 0.2 after 5 epochs of stagnant validation loss, with a floor of 1e-7
- **ModelCheckpoint** saving only the weights corresponding to the highest validation accuracy

### Evaluation Metrics

Given the class imbalance inherent in clinical retinal datasets (healthy retinas constitute the majority), raw accuracy alone is an insufficient measure. The model was evaluated against:

- **Accuracy** — overall proportion of correct classifications
- **Precision** — proportion of predicted positives that are true positives (minimising over-referral)
- **Recall / Sensitivity** — proportion of true disease cases correctly identified (the primary clinical metric, as false negatives carry the highest risk)
- **AUC (Area Under the ROC Curve)** — the standard metric in healthcare AI benchmarking, measuring discrimination ability across all classification thresholds

Reported baseline performance: 85% training accuracy, 82% validation accuracy.

---

## Explainability — Grad-CAM Implementation

A central design requirement of RetinaGuard was that the AI system must not function as an opaque black box in a clinical context. A clinician who cannot examine the evidence underlying a diagnostic decision cannot responsibly act on it. This is not merely a usability concern — it is an ethical and regulatory necessity for any AI system operating in medical triage.

Gradient-weighted Class Activation Mapping (Grad-CAM) was implemented to meet this requirement.

The Grad-CAM algorithm operates as follows:

1. The model is split into two sub-models: one that outputs the activations of the last convolutional layer, and one that takes those activations and produces the final classification.
2. A forward pass is run through both sub-models under `tf.GradientTape`, which records all operations.
3. The gradient of the top-predicted class score with respect to the last convolutional layer's output is computed by unrolling the tape.
4. These gradients are globally average-pooled to obtain a single importance weight per feature channel.
5. The feature maps are weighted by their corresponding importance scores and averaged across channels, producing a 2D heatmap.
6. The heatmap is upsampled to the original image resolution and visualised using the COLORMAP_JET colour scale, where red and yellow indicate regions of high computational importance.

The resulting overlay provides the clinician with a spatially localised explanation: the AI does not simply declare a severity stage, it highlights the specific lesions or vascular abnormalities that informed that decision.

---

## Directory Structure

```
Diabetic Retinopathy/
|
+-- app/
|   +-- frontend.py          # Main Streamlit application: UI, inference, report generation
|
+-- src/
|   +-- config.py            # Central configuration: hyperparameters, paths, class definitions
|   +-- model.py             # EfficientNetB3 architecture definition and compilation
|   +-- train.py             # Full training pipeline: DiabeticRetinopathyDetector class
|   +-- data_pipeline.py     # Dataset loading, augmentation, and preprocessing utilities
|   +-- explainability.py    # Grad-CAM heatmap generation and overlay logic
|
+-- models/
|   +-- dr_model_best.keras  # Saved model weights (pre-trained, production-ready)
|
+-- HF_deploy/
|   +-- Dockerfile           # Container definition for Hugging Face Spaces deployment
|
+-- sample_images/
|   +-- healthy.png          # Bundled reference image: healthy retina
|   +-- mild_dr.png          # Bundled reference image: early-stage DR
|
+-- requirements.txt         # Python dependency manifest
+-- .python-version          # Python version pin (3.11)
+-- README.md
```

---

## Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Core Language | Python 3.11 | Industry standard for machine learning and scientific computing |
| Deep Learning Framework | TensorFlow 2.x / Keras | Production-grade framework with strong support for SavedModel and quantisation |
| Base Architecture | EfficientNetB3 (ImageNet) | Best accuracy-to-parameter ratio in its class; supports compound scaling |
| Web Framework | Streamlit | Enables rapid development of data-rich interactive interfaces in pure Python |
| Image Processing | OpenCV (cv2) | Industry-standard library for real-time image manipulation and colour space conversion |
| Report Generation | FPDF | Lightweight, dependency-free PDF generation suitable for clinical document formatting |
| Numerical Computing | NumPy | Foundational array operations for tensor manipulation and heatmap construction |
| Model Evaluation | Scikit-learn | Standardised confusion matrix, classification report, and metric computation |
| Containerisation | Docker | Reproducible, isolated runtime for consistent cloud deployment |
| Cloud Hosting | Streamlit Cloud / Hugging Face Spaces | Purpose-built for machine learning applications; free tier suitable for portfolio deployment |
| Version Control | Git / GitHub | Standard source control and collaboration platform |

---

## The Five Clinical Severity Stages

The classification target follows the International Clinical Diabetic Retinopathy Disease Severity Scale:

**No DR (Healthy):** No visible vascular abnormalities. Annual screening recommended for all diabetic patients.

**Mild Non-Proliferative DR:** Presence of microaneurysms only — small, localised balloon-like outpouchings of retinal capillary walls. Clinically manageable with glycaemic optimisation and follow-up within 12 months.

**Moderate Non-Proliferative DR:** More than just microaneurysms, but less severe than severe NPDR. Includes dot and blot haemorrhages, hard exudates, and cotton-wool spots. Requires ophthalmologist review within 3 to 6 months.

**Severe Non-Proliferative DR:** Extensive intraretinal haemorrhages in all four quadrants, venous beading, and intraretinal microvascular abnormalities (IRMA). Greater than 15 percent risk of progression to proliferative disease within 12 months. Requires urgent specialist referral.

**Proliferative DR:** Neovascularisation — the growth of fragile, abnormal blood vessels on the retinal surface or optic disc. These vessels are prone to haemorrhage into the vitreous, leading to sudden and severe vision loss, tractional retinal detachment, and neovascular glaucoma. Constitutes a sight-threatening emergency requiring immediate intervention.

---

## Local Setup and Execution

### Prerequisites

- Python 3.11 or higher
- pip package manager
- The file `models/dr_model_best.keras` must be present in the repository. This file contains the pre-trained weights and is required for inference. It is tracked via Git LFS if not bundled directly.

### Installation

Clone the repository and install all required dependencies:

```bash
git clone https://github.com/adil-56/Clinical-Decision-Support-System-for-Diabetic-Retinopathy-using-Deep-Learning-.git
cd Clinical-Decision-Support-System-for-Diabetic-Retinopathy-using-Deep-Learning-
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app/frontend.py
```

The application will be accessible at `http://localhost:8501` in your browser.

### Running Model Training

To retrain the model from scratch, acquire the dataset from Kaggle (link below), place the extracted contents in the `data/raw/` directory, and execute:

```bash
python src/train.py
```

Training configuration parameters — image size, batch size, epoch count, and learning rate — are managed centrally in `src/config.py`.

---

## Cloud Deployment

### Streamlit Cloud

1. Push the repository to GitHub, ensuring `models/dr_model_best.keras` is committed.
2. Log in to [share.streamlit.io](https://share.streamlit.io) and connect the repository.
3. Set the main file path to `app/frontend.py`.
4. Set the Python version to `3.11` in the advanced settings.
5. Deploy. Streamlit Cloud will install dependencies and launch the application automatically.

### Hugging Face Spaces (Docker)

The `HF_deploy/Dockerfile` defines a containerised deployment configured for Hugging Face Spaces:

```bash
# The container exposes port 7860, which is mandatory for Hugging Face Spaces
# The startup command launches Streamlit on that port
```

To deploy, push the contents of `HF_deploy/` to a Hugging Face Space repository configured with the Docker SDK.

---

## Dataset

The model was trained and evaluated on publicly available clinical retinal imaging data sourced from Kaggle.

- **Primary Dataset:** [Diabetic Retinopathy 224x224 (2019 Data)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data) by Sovit Rath
- **Image Dimensions:** 224 x 224 pixels (pre-resized)
- **Classes:** 5 (No DR, Mild, Moderate, Severe, Proliferative)
- **Original Source:** APTOS 2019 Blindness Detection competition, Aravind Eye Hospital, Coimbatore, India

The dataset exhibits class imbalance reflective of clinical reality, with the No DR category comprising the majority of samples. This was addressed during training through stratified train/validation/test splitting.

---

## Performance Summary

| Metric | Value |
|---|---|
| Training Accuracy | 85% |
| Validation Accuracy | 82% |
| Primary Evaluation Metric | AUC (Area Under ROC Curve) |
| Optimiser | Adam |
| Initial Learning Rate | 1e-3 |
| Fine-Tuning Learning Rate | 1e-4 |
| Input Resolution | 224 x 224 x 3 |
| Number of Parameters | ~12 million (EfficientNetB3) |

---

## Important Limitations and Disclaimer

RetinaGuard is a research and portfolio demonstration project developed as part of an academic curriculum in Data Science and Machine Learning at Alliance University. It is not a certified, regulated, or validated medical device.

This application must not be used as a substitute for professional clinical judgment, ophthalmological examination, or any regulated diagnostic procedure. All diagnostic outputs produced by this system are advisory in nature and require validation by a qualified medical professional before any clinical decision is made.

The system has not undergone prospective clinical validation, regulatory review, or approval by any medical authority. It does not constitute a medical device as defined by applicable healthcare regulations in any jurisdiction.

---

## Future Development Roadmap

- Integration of additional foundation models — EfficientNetV2, Vision Transformer (ViT) — for performance benchmarking
- Prospective validation on a geographically diverse, multi-centre clinical dataset
- REST API layer to enable integration with Electronic Health Record systems and telemedicine platforms
- Quantisation and TFLite conversion for offline inference on edge devices and mobile applications
- Implementation of ensemble uncertainty estimation to flag low-confidence predictions for human review
- Support for optical coherence tomography (OCT) imaging as a secondary input modality

---

## Acknowledgements

- Dataset provided by the APTOS 2019 Blindness Detection Challenge, sourced from Aravind Eye Hospital, Coimbatore, India, and published via Kaggle
- EfficientNetB3 architecture by Mingxing Tan and Quoc V. Le, Google Brain, 2019
- Grad-CAM methodology by Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017
- TensorFlow and Keras documentation and open-source community
- Faculty guidance, Department of Data Science, Alliance University, Bengaluru

---

## Author

**Adil Khan**
MCA — Data Science Specialization, Alliance University, Bengaluru
GitHub: [github.com/adil-56](https://github.com/adil-56)
