# RetinaGuard — Clinical Decision Support System for Diabetic Retinopathy

**An AI-powered retinal screening application for early detection and severity staging of Diabetic Retinopathy using Transfer Learning and Explainable AI.**

**Live Application:** [diabetic-retinopathy-4556.streamlit.app](https://diabetic-retinopathy-4556.streamlit.app/)

---

## Table of Contents

1. [The Problem](#the-problem)
2. [The Solution](#the-solution)
3. [Why I Built This](#why-i-built-this)
4. [Clinical Scope and Real-World Impact](#clinical-scope-and-real-world-impact)
5. [How This Can Generate Value](#how-this-can-generate-value)
6. [Core Capabilities](#core-capabilities)
7. [System Architecture](#system-architecture)
8. [Model Architecture and Training Strategy](#model-architecture-and-training-strategy)
9. [Explainability — Grad-CAM](#explainability--grad-cam)
10. [The Five Clinical Severity Stages](#the-five-clinical-severity-stages)
11. [Technology Stack](#technology-stack)
12. [Project Structure](#project-structure)
13. [Local Setup](#local-setup)
14. [Deployment](#deployment)
15. [Dataset](#dataset)
16. [Performance Metrics](#performance-metrics)
17. [Limitations and Disclaimer](#limitations-and-disclaimer)
18. [Roadmap](#roadmap)
19. [Acknowledgements](#acknowledgements)
20. [Author](#author)

---

## The Problem

Diabetic Retinopathy (DR) is a microvascular complication of diabetes mellitus and one of the leading causes of preventable blindness in working-age adults globally. The progression of the disease is insidious — in its early and moderate stages, it produces no symptoms. Patients experience no pain, no noticeable vision deterioration, and no clinical warning signs until the damage has become irreversible.

The scale of the crisis is substantial. Over 463 million people worldwide are living with diabetes, and the International Diabetes Federation projects that number will rise to 700 million by 2045. India alone accounts for more than 77 million diagnosed diabetics — the second-largest diabetic population on earth — yet the national ratio of certified ophthalmologists to rural patients makes routine retinal screening structurally impossible for the majority of the at-risk population.

The barrier is not treatment. Effective treatments exist. Laser photocoagulation, anti-VEGF intravitreal injection, and vitreoretinal surgery can halt or reverse progression when intervention is applied at the right stage. The barrier is access to timely, accurate, and scalable screening.

By the time most patients in underserved settings receive a referral, proliferative disease has already developed — the stage at which abnormal blood vessels grow on the retinal surface, rupture into the vitreous, and cause sudden, irreversible vision loss. The World Health Organization estimates that 80 percent of DR-related blindness is preventable. The intervention required is not a surgical breakthrough. It is an earlier diagnosis.

The clinical challenge is compounded by the economics of skilled labour. A trained ophthalmologist can evaluate approximately 30 to 50 retinal images per day under manual grading conditions. At population scale, this creates a bottleneck that cannot be resolved by training more specialists on any realistic timeline. Automated screening, validated against expert-graded data and deployed at the point of care, is the only viable path to closing this gap.

---

## The Solution

RetinaGuard is a full-stack clinical decision support application that accepts a retinal fundus photograph as input and returns a five-class severity classification aligned with the International Clinical Diabetic Retinopathy Disease Severity Scale. In addition to the diagnostic output, the system generates a visual explanation using Gradient-weighted Class Activation Mapping (Grad-CAM), enabling the reviewing clinician to verify which anatomical regions of the retina drove the model's prediction.

The system integrates patient-level clinical metadata — glycated hemoglobin (HbA1c) and diabetes duration — to produce a contextualized triage recommendation and a downloadable clinical report in PDF format.

The application is deployed as a cloud-accessible web portal. No local installation is required. Any general practitioner, nurse, or medical officer with a smartphone or laptop and a retinal fundus camera can run a screening assessment at the point of care, in a primary health centre, or in a field camp, without requiring a specialist on-site.

### The Core Workflow

```
Clinician uploads retinal fundus image
           |
           v
Image preprocessing: resize to 224x224, normalize pixel values
           |
           v
EfficientNetB3 inference: generates 5-class softmax probability distribution
           |
           v
Grad-CAM: generates class-discriminative heatmap overlay
           |
           v
Triage logic: combines model output with HbA1c and diabetes duration
           |
           v
Clinical dashboard: displays diagnosis, confidence, heatmap, and recommendation
           |
           v
PDF report generated and made available for download
```

---

## Why I Built This

The choice of this project was driven by a specific line of reasoning, not by trend-following.

Most machine learning projects in academic portfolios select safe, well-documented problems — house price prediction, sentiment analysis, customer churn — where the dataset is clean, the metric is obvious, and the real-world consequences of a wrong prediction are negligible. I wanted to build something where the stakes of the problem were real, even if the application itself was a demonstration.

Diabetic Retinopathy met that standard. The data is publicly available from a peer-reviewed Kaggle competition sourced from an actual hospital. The classification target is standardized by the international medical community. The deployment context — primary care workers in settings with limited specialist access — is one where a tool like this would have genuine use if taken to production and clinically validated.

My secondary motivation was to understand what it actually means to build an AI system responsibly for a high-stakes domain. The standard machine learning curriculum teaches you to optimize accuracy. But a screening tool for a blinding disease requires a different orientation. False negatives — predicting "No DR" when early disease is present — carry a completely different clinical consequence than false positives. The model must be evaluated against recall and AUC, not just aggregate accuracy. Explainability is not a feature add-on; it is a prerequisite for any clinician to reasonably act on the output.

This project forced me to think about machine learning not as a prediction problem, but as a system design problem with a human decision-maker at the end of the pipeline.

My approach was to build the full stack: data pipeline, model training with two-phase transfer learning, Grad-CAM explainability, patient-level triage logic, report generation, UI, and cloud deployment. The goal was to understand every component of a clinical AI system, not just the model training loop.

---

## Clinical Scope and Real-World Impact

### Who Uses This

The intended user is not a data scientist or an engineer. The intended user is a general practitioner, a primary health centre nurse, or a community health worker in a district where the nearest ophthalmology department is hours away.

The system is designed to support triage decisions, not to replace specialist diagnosis. Its output answers a specific question: does this patient's retina show signs that warrant urgent referral to an ophthalmologist, or can this patient be safely scheduled for a routine follow-up?

### The Population-Scale Argument

If a single model can grade one retinal image in under three seconds, and if that model is accessible via a web browser, the same screening workflow that currently requires a specialist visit can be performed at a primary health centre during a routine diabetes check-up. At scale across a district health system, this could mean thousands of additional patients screened per month without adding a single ophthalmologist to the system.

The clinical value is front-loaded at the moderate-to-severe transition. Identifying moderate NPDR (Diabetic Retinopathy) before it crosses into the severe category — where the risk of progression to proliferative disease within 12 months exceeds 15 percent — is the intervention that preserves vision.

### What This Project Demonstrates

Even as a portfolio demonstration, this project establishes a functional prototype of the following system components that would need to exist in a production clinical AI deployment:

- Automated image preprocessing at a clinical quality standard
- A fine-tuned deep learning classifier with documented performance metrics
- Explainability output suitable for clinical review
- Metadata-contextualized triage logic
- Structured reporting for handoff to a specialist

---

## How This Can Generate Value

### Healthcare SaaS

A validated version of this system can be deployed as a Software as a Service product for telemedicine companies, hospital chains, and national health programmes. A per-screening fee model is standard for diagnostic AI in radiology and pathology. At scale, even a marginal per-image fee generates significant recurring revenue when multiplied across district-level health systems.

### Government and NGO Contracts

National diabetic retinopathy screening programmes exist in the United Kingdom (NHS DR Screening Programme), parts of Australia, and several Indian state health departments. These programmes operate at scale and have documented needs for automated grading tools. A validated AI grading system that meets regulatory standards can compete directly for these contracts.

### Integration with Electronic Health Records

The system's REST API layer (included in the deployment architecture) enables integration with existing electronic health record platforms. Hospitals and clinic networks pay for pre-integrated modules. A grading API that plugs into an EHR system adds direct clinical workflow value and commands a different pricing category than a standalone application.

### Medical Device Regulatory Pathway

In the United States, the FDA has already cleared AI-based diabetic retinopathy screening tools under the De Novo pathway — most notably IDx-DR, which received clearance in 2018. The European MDR and India's CDSCO have analogous pathways. A system that passes prospective clinical validation and achieves the required sensitivity and specificity thresholds is eligible to pursue regulatory clearance, which converts it from a software tool into a certified medical device. Certified medical devices command a fundamentally different commercial value than software products.

---

## Core Capabilities

| Capability | Description |
|---|---|
| Five-class severity classification | No DR, Mild NPDR, Moderate NPDR, Severe NPDR, Proliferative DR |
| Transfer Learning inference | EfficientNetB3 fine-tuned on clinical retinal image data (APTOS 2019) |
| Grad-CAM visual explainability | Class-discriminative heatmap overlaid on source image |
| Patient-contextualized triage logic | Output adjusted by HbA1c level and years with diabetes |
| PDF clinical report | Downloadable, formatted report for specialist handoff |
| Cloud-deployed web portal | No local installation required; browser-accessible |
| Docker containerization | Reproducible execution across environments |

---

## System Architecture

RetinaGuard follows a monolithic web application architecture. All inference, explainability computation, and report generation execute within a single Python process. This design eliminates inter-service latency and simplifies deployment on resource-constrained infrastructure — a deliberate choice for a tool intended for deployment in settings where DevOps capacity is minimal.

```
Retinal Fundus Image (JPEG / PNG)
           |
           v
  Image Preprocessing Layer
  - Resize to 224 x 224 pixels
  - Normalize pixel intensities to [0.0, 1.0]
  - CLAHE contrast enhancement (optional)
           |
           v
  EfficientNetB3 Feature Extraction
  (12 million parameters, ImageNet pre-trained weights, DR fine-tuned)
           |
           v
  Custom Classification Head
  - GlobalAveragePooling2D
  - Dense(256, ReLU activation)
  - BatchNormalization
  - Dropout(0.5)
  - Dense(5, Softmax activation)
           |
      -----+-----
      |          |
      v          v
Softmax        Grad-CAM
Prediction     Explainability
(5-class       (GradientTape
probability)   heatmap)
      |          |
      v          v
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

### Base Architecture

The project uses EfficientNetB3, a convolutional neural network developed by Google Brain and introduced in *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks* (Tan and Le, 2019). EfficientNet employs compound scaling, simultaneously optimizing network width, depth, and input resolution using a fixed set of scaling coefficients derived through neural architecture search. This approach consistently achieves higher accuracy at lower parameter count compared to manually scaled networks of equivalent size.

EfficientNetB3 was selected over alternatives including VGG16, ResNet50, and InceptionV3 on the following grounds:

- It achieves state-of-the-art accuracy with fewer parameters, reducing inference latency on CPU-class cloud servers
- Compound scaling is particularly effective for fine-grained medical imaging tasks where spatial resolution carries diagnostic information
- Pre-trained ImageNet weights provide a strong initialization for the feature extraction layers, substantially reducing the volume of labelled clinical data required for effective fine-tuning

### Two-Phase Training Protocol

Training followed the industry-standard two-phase transfer learning protocol for medical imaging.

**Phase 1 — Feature Extraction with Frozen Base**

The EfficientNetB3 base was frozen, preserving ImageNet-derived weights. Only the custom classification head was trained. This prevents catastrophic forgetting of foundational visual representations — edges, textures, shapes — during early epochs when gradient magnitudes are large and destabilizing.

**Phase 2 — Global Fine-Tuning**

After the classification head converged, the base model was unfrozen and the entire network was retrained at a reduced learning rate of 1e-4. This allows the lower-level convolutional representations to adapt toward diabetic retinal pathology — microaneurysms, hard exudates, neovascularization, and vitreous hemorrhage — without erasing the general feature hierarchy established in Phase 1.

### Training Callbacks

Three callbacks were employed during training:

| Callback | Configuration | Purpose |
|---|---|---|
| EarlyStopping | patience=10, restore_best_weights=True, monitor=val_loss | Prevents overfitting; restores optimal checkpoint |
| ReduceLROnPlateau | factor=0.2, patience=5, min_lr=1e-7, monitor=val_loss | Recovers from learning plateaus without manual tuning |
| ModelCheckpoint | monitor=val_accuracy, save_best_only=True | Persists only the weights achieving peak validation accuracy |

### Evaluation Methodology

Given the class imbalance inherent in clinical retinal datasets — where healthy retinas constitute the majority of samples — aggregate accuracy is an insufficient evaluation criterion. The model was evaluated against the following metrics:

- **Accuracy:** Overall proportion of correct classifications
- **Precision:** Proportion of predicted disease cases that are true disease cases; minimizes unnecessary referrals
- **Recall / Sensitivity:** Proportion of true disease cases correctly identified; the primary clinical metric, because missed diagnoses carry the highest patient risk
- **AUC (Area Under the ROC Curve):** Standard benchmark in healthcare AI; measures discrimination ability across all classification thresholds independent of class prevalence

---

## Explainability — Grad-CAM

A non-negotiable design requirement for any clinical AI system is interpretability. A clinician who cannot examine the evidence underlying a diagnostic recommendation cannot responsibly act on it. This is not merely a usability consideration — it is an ethical requirement and a regulatory prerequisite for AI operating in medical triage contexts.

Gradient-weighted Class Activation Mapping (Grad-CAM) was implemented to meet this requirement. The algorithm, introduced by Selvaraju et al. in *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (ICCV 2017), produces a class-discriminative spatial heatmap that highlights which regions of the input image most strongly influenced the model's classification.

### Implementation

The Grad-CAM computation proceeds as follows:

1. The trained model is partitioned into two sub-models: a feature extractor terminating at the final convolutional layer, and a classifier that takes those activations to the final softmax output.
2. A forward pass is executed through both sub-models under `tf.GradientTape`, which records all tensor operations for automatic differentiation.
3. The gradient of the predicted class score with respect to the output of the final convolutional layer is computed by unrolling the recorded tape.
4. These gradients are globally average-pooled across spatial dimensions, producing a scalar importance weight for each feature channel.
5. The feature maps are weighted by their corresponding importance coefficients and summed across channels, producing a 2D activation map.
6. The activation map is passed through a ReLU function (retaining only positive influences on the predicted class), upsampled to the original image resolution via bilinear interpolation, and visualized using the COLORMAP_JET scale — where red and yellow regions represent high importance and blue regions represent low importance.

The resulting overlay gives the reviewing clinician a spatially localized explanation. The model does not simply assert a severity stage — it shows which specific regions of the retinal image informed that decision, whether microaneurysms in the macular region, hard exudates in the arcade vessels, or neovascularization at the optic disc.

---

## The Five Clinical Severity Stages

Classification follows the International Clinical Diabetic Retinopathy Disease Severity Scale.

**No DR — Healthy Retina**
No vascular abnormalities are visible. Annual fundoscopic screening is recommended for all patients with diagnosed diabetes mellitus.

**Mild Non-Proliferative Diabetic Retinopathy**
Microaneurysms only — localized balloon-like outpouchings of retinal capillary walls at sites of structural weakness. Clinically manageable with optimized glycaemic control and follow-up within 12 months.

**Moderate Non-Proliferative Diabetic Retinopathy**
More extensive vascular abnormalities than microaneurysms alone, but not meeting the criteria for severe NPDR. Includes dot and blot hemorrhages, hard exudates representing lipid deposition from leaky vessels, and cotton-wool spots indicating focal ischemia. Requires ophthalmologist review within 3 to 6 months.

**Severe Non-Proliferative Diabetic Retinopathy**
Extensive intraretinal hemorrhages in all four retinal quadrants, venous beading, and intraretinal microvascular abnormalities (IRMA). Carries greater than 15 percent risk of progression to proliferative disease within 12 months without treatment. Requires urgent specialist referral.

**Proliferative Diabetic Retinopathy**
Neovascularization — the pathological growth of fragile, abnormal blood vessels on the retinal surface or optic disc in response to retinal ischemia. These vessels are structurally compromised and prone to rupture into the vitreous cavity, producing sudden and severe vision loss. Advanced complications include tractional retinal detachment and neovascular glaucoma. Constitutes a sight-threatening emergency requiring immediate vitreoretinal intervention.

---

## Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Core Language | Python 3.11 | Industry standard for machine learning and scientific computing |
| Deep Learning Framework | TensorFlow 2.x / Keras | Production-grade; strong support for SavedModel, quantization, and GradientTape |
| Base Architecture | EfficientNetB3 (ImageNet weights) | Best accuracy-to-parameter ratio in its class; compound scaling effective for medical imaging |
| Web Framework | Streamlit | Enables interactive, data-rich interfaces in pure Python without frontend engineering overhead |
| Image Processing | OpenCV (cv2) | Industry-standard library for image manipulation, color space conversion, and CLAHE enhancement |
| Report Generation | FPDF | Lightweight, dependency-free PDF generation suitable for clinical document formatting |
| Numerical Computing | NumPy | Foundational array operations for tensor manipulation, heatmap construction, and normalization |
| Model Evaluation | Scikit-learn | Standardized classification report, confusion matrix, and stratified split utilities |
| Containerization | Docker | Reproducible, isolated runtime environment for consistent cloud deployment |
| Cloud Hosting | Streamlit Cloud / Hugging Face Spaces | Purpose-built platforms for machine learning applications with free-tier deployment |
| Version Control | Git / GitHub | Standard source control; Git LFS for large model weight files |

### Why TensorFlow over PyTorch

Both frameworks are viable for this problem class. TensorFlow was selected for three specific reasons: the Keras `SavedModel` format provides a stable serialization target across framework versions, `tf.GradientTape` exposes a clean imperative interface for the Grad-CAM computation, and TensorFlow Lite conversion — required for the mobile edge deployment path on the roadmap — is natively supported without third-party tooling.

---

## Project Structure

```
Clinical-Decision-Support-System-for-Diabetic-Retinopathy/
|
+-- app/
|   +-- frontend.py          # Streamlit application: UI, inference pipeline, report generation
|
+-- src/
|   +-- config.py            # Central configuration: hyperparameters, paths, class label definitions
|   +-- model.py             # EfficientNetB3 architecture definition and compilation
|   +-- train.py             # Full training pipeline: DiabeticRetinopathyDetector class
|   +-- data_pipeline.py     # Dataset loading, augmentation, CLAHE preprocessing utilities
|   +-- explainability.py    # Grad-CAM heatmap generation and overlay composition
|   +-- __init__.py
|
+-- models/
|   +-- dr_model_best.keras  # Saved model weights (pre-trained, production-ready)
|   +-- READ.md              # Model usage notes and weight provenance
|
+-- HF_deploy/               # Hugging Face Spaces deployment variant
|   +-- Dockerfile           # Container definition for port 7860
|   +-- app/                 # Application code for HF environment
|   +-- src/                 # Source modules mirrored for HF deployment
|
+-- sample_images/
|   +-- healthy.png          # Reference image: healthy retina (No DR)
|   +-- mild_dr.png          # Reference image: early-stage DR
|
+-- requirements.txt         # Python dependency manifest
+-- .python-version          # Python version pin: 3.11
+-- README.md
```

---

## Local Setup

### Prerequisites

- Python 3.11 or higher
- pip package manager
- The pre-trained model file `models/dr_model_best.keras` must be present. This file is tracked via Git LFS if not bundled directly in the repository clone.

### Installation

```bash
git clone https://github.com/adil-56/Clinical-Decision-Support-System-for-Diabetic-Retinopathy-using-Deep-Learning-.git
cd Clinical-Decision-Support-System-for-Diabetic-Retinopathy-using-Deep-Learning-
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app/frontend.py
```

The application will be available at `http://localhost:8501`.

### Running Model Training

To retrain the model from scratch, acquire the APTOS 2019 dataset from Kaggle (linked in the Dataset section), extract the contents to `data/raw/`, and execute:

```bash
python src/train.py
```

All training configuration parameters — image dimensions, batch size, epoch count, learning rates, and class definitions — are managed centrally in `src/config.py`.

---

## Deployment

### Live Application

The application is publicly accessible at:

**[https://diabetic-retinopathy-4556.streamlit.app/](https://diabetic-retinopathy-4556.streamlit.app/)**

No installation is required. Open the URL in any modern browser, upload a retinal fundus image or select a bundled sample, enter patient metadata, and run the analysis.

### Streamlit Cloud

1. Push the repository to GitHub, ensuring `models/dr_model_best.keras` is committed (via Git LFS if necessary).
2. Log in to [share.streamlit.io](https://share.streamlit.io) and connect the repository.
3. Set the main file path to `app/frontend.py`.
4. Set the Python version to `3.11` in the advanced deployment settings.
5. Deploy. Streamlit Cloud will install all dependencies from `requirements.txt` and launch the application automatically.

### Hugging Face Spaces (Docker)

The `HF_deploy/Dockerfile` defines a containerized deployment configured specifically for Hugging Face Spaces. The container exposes port 7860, which is mandatory for the Hugging Face runtime. The startup command launches Streamlit bound to that port.

To deploy, push the contents of `HF_deploy/` to a Hugging Face Space repository configured with the Docker SDK.

---

## Dataset

The model was trained and validated on a publicly available clinical retinal imaging dataset.

| Attribute | Value |
|---|---|
| Primary Dataset | Diabetic Retinopathy 224x224 (2019 Data), Sovit Rath on Kaggle |
| Dataset Link | [kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data) |
| Original Source | APTOS 2019 Blindness Detection Competition, Aravind Eye Hospital, Coimbatore, India |
| Image Dimensions | 224 x 224 pixels (pre-resized) |
| Classes | 5: No DR, Mild, Moderate, Severe, Proliferative |
| Class Imbalance | Present; No DR constitutes the largest class |
| Imbalance Handling | Stratified train / validation / test splitting |

The class distribution in the dataset reflects the actual clinical prevalence distribution — healthy retinas are more common than advanced disease in a general screening population. This is intentional and clinically appropriate. Artificially balancing the dataset through oversampling would distort the model's calibration and make its confidence scores unreliable in real-world deployment.

---

## Performance Metrics

| Metric | Value |
|---|---|
| Training Accuracy | 85% |
| Validation Accuracy | 82% |
| Primary Evaluation Metric | AUC (Area Under the ROC Curve) |
| Optimizer | Adam |
| Initial Learning Rate (Phase 1) | 1e-3 |
| Fine-Tuning Learning Rate (Phase 2) | 1e-4 |
| Input Resolution | 224 x 224 x 3 |
| Total Parameters | approximately 12 million (EfficientNetB3) |
| Loss Function | Categorical Cross-Entropy |

---

## Limitations and Disclaimer

RetinaGuard is a research and portfolio demonstration project developed as part of an academic curriculum in Data Science and Machine Learning. It is not a certified, regulated, or clinically validated medical device.

This application must not be used as a substitute for professional clinical judgment, ophthalmological examination, or any regulated diagnostic procedure. All diagnostic outputs are advisory in nature and require validation by a qualified medical professional before any clinical decision is made.

The system has not undergone prospective clinical validation, regulatory review, or approval by any medical authority in any jurisdiction. It does not constitute a medical device as defined by the FDA, CE MDR, CDSCO, or equivalent regulatory body.

Performance metrics reported here were obtained on a held-out test split of the APTOS 2019 dataset. Real-world performance on a prospective, geographically diverse clinical population may differ materially.

---

## Roadmap

**Short Term**
- Benchmarking against EfficientNetV2 and Vision Transformer (ViT) architectures on the same dataset
- Per-class performance breakdown with precision-recall curves published in the repository
- REST API endpoint layer for integration with third-party EHR and telemedicine systems

**Medium Term**
- Prospective validation study on a geographically diverse, multi-centre dataset
- TensorFlow Lite conversion for offline inference on edge devices and mobile applications
- Ensemble uncertainty estimation to flag low-confidence predictions for mandatory human review

**Long Term**
- Optical coherence tomography (OCT) imaging as a secondary input modality
- Multi-label classification for concurrent detection of additional retinal pathologies — diabetic macular edema, age-related macular degeneration
- Regulatory pre-submission consultation for De Novo or 510(k) pathway in the United States; Class IIa MDR pathway in the European Union

---

## Acknowledgements

- Dataset: APTOS 2019 Blindness Detection Challenge, Aravind Eye Hospital, Coimbatore, India; published via Kaggle by Sovit Rath
- Architecture: EfficientNetB3 — Mingxing Tan and Quoc V. Le, Google Brain, 2019
- Explainability: Grad-CAM — Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017
- TensorFlow and Keras open-source community
- Faculty guidance, Department of Data Science, Alliance University, Bengaluru

---

## Author

**Adil Khan**
MCA — Data Science Specialization, Alliance University, Bengaluru
GitHub: [github.com/adil-56](https://github.com/adil-56)
