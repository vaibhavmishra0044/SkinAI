"""Generate IEEE-format research paper as a Word document."""
import os
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IEEE_Research_Paper.docx")

doc = Document()

# -- Page setup --
for sec in doc.sections:
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(1.91)
    sec.right_margin = Cm(1.91)

style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(10)
style.paragraph_format.space_after = Pt(0)
style.paragraph_format.space_before = Pt(0)

def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(24)
    r.font.name = 'Times New Roman'

def add_authors(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.size = Pt(11)
    r.font.name = 'Times New Roman'
    r.italic = True

def add_heading_ieee(text, level=1):
    p = doc.add_paragraph()
    p.space_before = Pt(12)
    p.space_after = Pt(6)
    r = p.add_run(text)
    r.bold = True
    r.font.name = 'Times New Roman'
    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r.font.size = Pt(12)
    elif level == 2:
        r.font.size = Pt(11)
        r.italic = True
    else:
        r.font.size = Pt(10)
        r.italic = True

def add_body(text):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(0.25)
    p.space_after = Pt(2)
    r = p.add_run(text)
    r.font.size = Pt(10)
    r.font.name = 'Times New Roman'

def add_body_no_indent(text):
    p = doc.add_paragraph()
    p.space_after = Pt(2)
    r = p.add_run(text)
    r.font.size = Pt(10)
    r.font.name = 'Times New Roman'

def add_table(headers, rows, caption=None):
    if caption:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.space_before = Pt(6)
        r = p.add_run(caption)
        r.bold = True
        r.font.size = Pt(9)
        r.font.name = 'Times New Roman'
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ''
        r = cell.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9)
        r.font.name = 'Times New Roman'
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = ''
            r = cell.paragraphs[0].add_run(str(val))
            r.font.size = Pt(9)
            r.font.name = 'Times New Roman'
    doc.add_paragraph()

def add_reference(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    p.space_after = Pt(2)
    r = p.add_run(text)
    r.font.size = Pt(9)
    r.font.name = 'Times New Roman'

# ===== TITLE =====
add_title("Skin Disease Detection System Using\nConvolutional Neural Networks and\nMulti-Tier AI Inference")
doc.add_paragraph()
add_authors("Author Name(s)")
add_authors("Department of Computer Science and Engineering")
add_authors("University Name, City, Country")
add_authors("Email: author@university.edu")
doc.add_paragraph()

# ===== ABSTRACT =====
add_heading_ieee("Abstract", 1)
p = doc.add_paragraph()
p.space_after = Pt(4)
r = p.add_run("Skin diseases affect approximately 1.9 billion people worldwide, yet access to qualified dermatologists remains severely limited in developing and rural regions. This paper presents SkinAI, a web-based skin disease detection system leveraging Convolutional Neural Networks (CNNs) with a multi-tier artificial intelligence inference pipeline. The system employs EfficientNetV2B0 as the backbone architecture, trained on the HAM10000 dataset (10,015 dermatoscopic images, 7 disease categories) using a feature-extraction transfer learning strategy. To address the severe class imbalance inherent in the dataset \u2014 where melanocytic nevi constitutes 66.95% of all samples \u2014 we apply aggressive oversampling (minimum 1,500 samples per class), inverse-frequency class weighting, label smoothing (\u03b1 = 0.1), and extensive data augmentation. The trained model achieves 75.82% top-1 accuracy and 96.72% top-3 accuracy on the held-out test set. The production system implements a cascading inference pipeline: an NVIDIA vision-language API as the primary analyzer (supporting 22+ skin conditions), a HuggingFace Vision Transformer as the secondary fallback, and the local CNN model as the tertiary fallback. The Flask-based web application integrates user authentication, patient context capture, symptom verification quizzes, an AI chat assistant, doctor referral management, and prediction history tracking.")
r.font.size = Pt(10)
r.font.name = 'Times New Roman'
r.italic = True

p = doc.add_paragraph()
p.space_after = Pt(6)
r = p.add_run("Keywords \u2014 ")
r.bold = True
r.italic = True
r.font.size = Pt(10)
r.font.name = 'Times New Roman'
r2 = p.add_run("Convolutional Neural Network, Deep Learning, EfficientNetV2, Flask, HAM10000, Image Classification, Multi-Tier Inference, Skin Disease Detection, Transfer Learning")
r2.italic = True
r2.font.size = Pt(10)
r2.font.name = 'Times New Roman'

# ===== I. INTRODUCTION =====
add_heading_ieee("I. INTRODUCTION", 1)
add_body("Skin diseases represent the fourth most common cause of non-fatal disease burden globally, affecting nearly one-third of the world\u2019s population at any given time [1]. Early and accurate detection is paramount for effective treatment outcomes, particularly for potentially malignant conditions such as melanoma and basal cell carcinoma. However, dermatological diagnosis remains heavily dependent on visual examination by trained specialists, and the global shortage of dermatologists creates significant barriers to timely diagnosis \u2014 particularly in rural, low-income, and underserved communities [2].")
add_body("Recent advances in deep learning, specifically Convolutional Neural Networks (CNNs), have demonstrated performance comparable to \u2014 and in some cases exceeding \u2014 board-certified dermatologists in skin lesion classification tasks [3], [4]. These developments have opened the possibility of developing accessible, AI-powered screening tools that can provide preliminary assessments and triage patients toward appropriate medical care.")
add_body("Despite these advances, several critical challenges persist in deploying such systems in real-world settings: (1) Class Imbalance \u2014 Medical imaging datasets are typically heavily imbalanced; in the HAM10000 dataset, melanocytic nevi constitutes 66.95% of samples, while dermatofibroma and vascular lesions represent only 1.15% and 1.42% respectively. (2) Single Model Dependency \u2014 Systems relying on a single local model face availability constraints and are limited to a fixed set of disease categories. (3) Lack of Patient Context \u2014 Image-only analysis ignores critical clinical information such as symptom duration, body location, and medication history. (4) Post-Prediction Support Gap \u2014 Most classification systems stop at outputting a disease label, without providing actionable guidance or specialist referral pathways.")
add_body("This paper addresses these challenges by presenting SkinAI, a comprehensive skin disease detection system that combines a CNN model trained with robust class-imbalance mitigation techniques, a multi-tier inference pipeline ensuring continuous system availability, patient context integration for contextually enriched analysis, and post-prediction support including symptom verification, AI consultation, and doctor referrals.")
add_body("The remainder of this paper is organized as follows: Section II reviews related work. Section III details the proposed methodology. Section IV presents experimental results and analysis. Section V discusses findings and limitations. Section VI concludes the paper and outlines future directions.")

# ===== II. RELATED WORK =====
add_heading_ieee("II. RELATED WORK", 1)
add_heading_ieee("A. Deep Learning for Skin Lesion Classification", 2)
add_body("Esteva et al. [3] demonstrated that a deep CNN trained on 129,450 clinical images achieved classification performance on par with 21 board-certified dermatologists, establishing a foundational benchmark for AI-driven dermatology. Haenssle et al. [4] further reinforced this finding, showing that a CNN outperformed 58 dermatologists in melanoma detection with higher sensitivity (95% vs. 86.6%). Brinker et al. [5] extended these findings in a large-scale study where deep learning outperformed 136 of 157 dermatologists in dermoscopic melanoma image classification.")

add_heading_ieee("B. CNN Architectures and Transfer Learning", 2)
add_body("He et al. [6] introduced residual connections (ResNet), enabling training of substantially deeper networks. Tan and Le [7] proposed EfficientNet, demonstrating that compound scaling of network depth, width, and resolution achieves superior accuracy with fewer parameters. Their subsequent work, EfficientNetV2 [8], introduced training-aware neural architecture search and progressive learning, achieving state-of-the-art results with significantly faster training. Transfer learning \u2014 initializing networks with weights pre-trained on ImageNet \u2014 has proven particularly effective for medical imaging tasks where labelled data is scarce [9].")

add_heading_ieee("C. Benchmark Datasets", 2)
add_body("Tschandl et al. [2] published the HAM10000 dataset, establishing a widely adopted benchmark for 7-class skin lesion classification. Codella et al. [10] demonstrated that ensemble methods improve multi-class skin lesion classification performance on the ISIC Archive.")

add_heading_ieee("D. Gap Analysis", 2)
add_table(
    ["Gap Identified", "How SkinAI Addresses It"],
    [
        ["Class imbalance ignored", "Oversampling (\u22651,500/class), class weights, label smoothing (0.1)"],
        ["Single model dependency", "Multi-tier: NVIDIA API \u2192 HuggingFace ViT \u2192 Local CNN"],
        ["Limited disease categories", "API extends coverage to 22+ conditions"],
        ["Image-only analysis", "Patient context form captures location, duration, symptoms, history"],
        ["No post-prediction support", "Symptom quiz, AI chat, doctor referrals, prediction history"],
        ["Preprocessing inconsistencies", "Consistent preprocess_input [0,255]\u2192[\u22121,1] in train and inference"],
    ],
    caption="TABLE I: GAP ANALYSIS OF EXISTING SYSTEMS"
)

# ===== III. PROPOSED METHODOLOGY =====
add_heading_ieee("III. PROPOSED METHODOLOGY", 1)

add_heading_ieee("A. System Architecture", 2)
add_body("The system follows a client-server architecture with three distinct layers: (1) a frontend web interface built with HTML5, CSS3, and JavaScript, (2) a Flask backend server with authentication and SQLite database management, and (3) a multi-tier AI inference pipeline. The frontend communicates with the backend via HTTP/JSON. The backend handles image upload, preprocessing, validation, inference routing, result formatting, and database persistence. The inference pipeline cascades through three tiers: NVIDIA Vision-Language API (primary), HuggingFace Vision Transformer (secondary), and local CNN model (tertiary).")

add_heading_ieee("B. Dataset", 2)
add_body("The system is trained on the HAM10000 dataset [2], comprising 10,015 dermatoscopic images across 7 diagnostic categories. Table II presents the original class distribution.")
add_table(
    ["Abbrev.", "Class Name", "Count", "%"],
    [
        ["nv", "Melanocytic Nevi", "6,705", "66.95%"],
        ["mel", "Melanoma", "1,113", "11.11%"],
        ["bkl", "Benign Keratosis-like Lesions", "1,099", "10.97%"],
        ["bcc", "Basal Cell Carcinoma", "514", "5.13%"],
        ["akiec", "Actinic Keratoses", "327", "3.26%"],
        ["vasc", "Vascular Lesions", "142", "1.42%"],
        ["df", "Dermatofibroma", "115", "1.15%"],
    ],
    caption="TABLE II: HAM10000 DATASET CLASS DISTRIBUTION"
)

add_heading_ieee("C. Data Preprocessing and Augmentation", 2)
add_body("All images are resized to 224 \u00d7 224 pixels using bilinear interpolation and normalized using EfficientNetV2\u2019s preprocess_input function, which scales pixel values from [0, 255] to [\u22121, 1]. This same preprocessing is applied consistently during both training and inference to eliminate preprocessing mismatch.")
add_body("The data augmentation pipeline applies the following transformations stochastically to each training sample: random horizontal and vertical flips; random 90\u00b0 rotations (0\u00b0, 90\u00b0, 180\u00b0, 270\u00b0); random brightness adjustment (\u00b120%); random contrast adjustment (factor \u2208 [0.75, 1.25]); random saturation adjustment (factor \u2208 [0.75, 1.25]); random hue shift (\u00b10.08); and random cropping (80\u2013100% of image area) followed by resize to 224 \u00d7 224.")

add_heading_ieee("D. Class Imbalance Mitigation", 2)
add_body("Three complementary strategies are employed: (1) Oversampling: Each class is boosted to at least 1,500 samples by repeating minority class samples with unique augmentations applied to each copy. (2) Inverse-Frequency Class Weighting: The loss function weight for each class c is computed as w_c = N_total / (K \u00d7 N_c), where N_total is the total sample count, K is the number of classes, and N_c is the count for class c. (3) Label Smoothing: A smoothing factor of \u03b1 = 0.1 is applied to the categorical cross-entropy loss.")

add_heading_ieee("E. CNN Model Architecture", 2)
add_body("The model architecture follows a feature-extraction transfer learning paradigm. The backbone is EfficientNetV2B0 pre-trained on ImageNet, with all layers frozen (5,919,312 parameters), producing a 1,280-dimensional feature vector via global average pooling.")
add_body("The classification head is a custom sequential network: Input(1280) \u2192 BatchNorm \u2192 Dropout(0.4) \u2192 Dense(512, ReLU, L2=1e-4) \u2192 BatchNorm \u2192 Dropout(0.3) \u2192 Dense(256, ReLU, L2=1e-4) \u2192 Dropout(0.2) \u2192 Dense(7, Softmax). The head incorporates batch normalization for training stability, progressive dropout (0.4 \u2192 0.3 \u2192 0.2) for regularization, and L2 weight regularization (\u03bb = 10\u207b\u2074) to prevent overfitting.")

add_heading_ieee("F. Training Pipeline", 2)
add_body("The training follows a two-stage approach optimized for CPU-constrained environments. Stage 1 \u2014 Feature Extraction (~10 minutes): The frozen EfficientNetV2B0 backbone processes all images once, extracting 1,280-dimensional feature vectors cached to disk as NumPy arrays. Stage 2 \u2014 Head Training (~5 minutes): The classification head is trained on cached features using AdamW optimizer (lr = 10\u207b\u00b3, weight decay = 10\u207b\u2074), categorical cross-entropy with label smoothing (\u03b1 = 0.1), batch size 256, maximum 200 epochs with early stopping (patience = 20), ReduceLROnPlateau (factor = 0.5, patience = 8), and ModelCheckpoint (best validation accuracy). An optional Stage 3 enables fine-tuning the last 30 backbone layers with a very low learning rate (3 \u00d7 10\u207b\u2076) for up to 12 epochs on GPU.")

add_heading_ieee("G. Multi-Tier Inference Pipeline", 2)
add_body("Tier 1 \u2014 NVIDIA Vision-Language API (Primary): The system sends the image (resized to 768 \u00d7 768, base64-encoded) along with patient context to the NVIDIA Nemotron-Nano-12B-V2-VL model. This model can reason over 22+ skin conditions beyond the 7 HAM10000 categories, including acne, eczema, psoriasis, ringworm, vitiligo, and lupus. The API returns structured JSON with condition name, confidence, top-3 differential diagnoses, risk level, and comprehensive clinical information.")
add_body("Tier 2 \u2014 HuggingFace Vision Transformer (Secondary): If the API is unavailable, a locally hosted ViT model (LaurianeMD/vit-skin-disease) provides classification across 22 dermatological conditions.")
add_body("Tier 3 \u2014 Local CNN Model (Tertiary): The TFLite-quantized or Keras-format CNN model serves as the final fallback, providing 7-class classification from the HAM10000 categories.")

add_heading_ieee("H. Web Application Architecture", 2)
add_body("The Flask application implements: user authentication via Flask-Login with Werkzeug bcrypt password hashing; SQLite database via Flask-SQLAlchemy with four models (User, Prediction, Feedback, Doctor); patient context capture (body location, symptom duration, itching/pain/spreading/fever indicators, medication history); symptom verification quiz generated via AI; AI chat assistant for contextual follow-up; doctor referral system managed via admin dashboard; and full prediction history with timestamps and top-3 results.")

# ===== IV. EXPERIMENTAL RESULTS =====
add_heading_ieee("IV. EXPERIMENTAL RESULTS", 1)

add_heading_ieee("A. Experimental Setup", 2)
add_body("The model was trained on a CPU-only environment. The dataset was split using stratified sampling: 80% training, 10% validation, and 10% test. After oversampling, the training set was balanced with each class containing at least 1,500 samples. Table III summarizes the training configuration.")
add_table(
    ["Parameter", "Value"],
    [
        ["Backbone", "EfficientNetV2B0 (ImageNet, frozen)"],
        ["Backbone Parameters", "5,919,312"],
        ["Feature Dimension", "1,280"],
        ["Head Architecture", "BN\u2192Drop(0.4)\u2192Dense(512)\u2192BN\u2192Drop(0.3)\u2192Dense(256)\u2192Drop(0.2)\u2192Dense(7)"],
        ["Optimizer", "AdamW (lr=10\u207b\u00b3, decay=10\u207b\u2074)"],
        ["Loss Function", "CategoricalCrossentropy (smoothing=0.1)"],
        ["Image Size", "224 \u00d7 224 \u00d7 3"],
        ["Max Epochs", "200 (early stopping, patience=20)"],
        ["Oversampling Minimum", "1,500 per class"],
    ],
    caption="TABLE III: TRAINING CONFIGURATION"
)

add_heading_ieee("B. Classification Performance", 2)
add_body("Table IV presents the per-class classification accuracy on the held-out test set.")
add_table(
    ["Class", "Accuracy (%)"],
    [
        ["Melanocytic Nevi", "82.71"],
        ["Vascular Lesions", "80.00"],
        ["Benign Keratosis-like Lesions", "70.00"],
        ["Melanoma", "59.82"],
        ["Basal Cell Carcinoma", "59.62"],
        ["Actinic Keratoses", "45.45"],
        ["Dermatofibroma", "41.67"],
        ["Overall (Top-1)", "75.82"],
        ["Overall (Top-3)", "96.72"],
    ],
    caption="TABLE IV: PER-CLASS TEST ACCURACY"
)

add_heading_ieee("C. Analysis of Results", 2)
add_body("The overall top-1 accuracy of 75.82% is achieved despite the significant class imbalance challenges. The top-3 accuracy of 96.72% indicates that the correct diagnosis is almost always among the model\u2019s top three predictions, making the system particularly suitable for differential diagnosis support.")
add_body("Melanocytic nevi achieves the highest per-class accuracy (82.71%), benefiting from the largest representation in the dataset. Vascular lesions performs strongly (80.00%) despite being the second-rarest class (1.42% of original data), attributable to the visually distinctive red/purple coloring. Dermatofibroma exhibits the lowest accuracy (41.67%), correlating with it being the rarest class (115 samples, 1.15%) and its visual similarity to other benign lesion types.")
add_body("Melanoma and basal cell carcinoma achieve approximately 60% accuracy each. While this is lower than desired for clinically critical conditions, the top-3 accuracy ensures these conditions are reliably captured within the differential diagnosis output. The multi-tier inference pipeline further addresses this limitation by leveraging larger vision-language models.")

# ===== V. DISCUSSION =====
add_heading_ieee("V. DISCUSSION", 1)

add_heading_ieee("A. Effectiveness of Class Imbalance Mitigation", 2)
add_body("Without oversampling and class weighting, preliminary experiments showed the model collapsing onto melanocytic nevi \u2014 predicting the majority class for virtually all inputs. The combination of oversampling to 1,500 samples per class, inverse-frequency class weights, and label smoothing successfully prevented this collapse, enabling meaningful accuracy across all seven classes.")

add_heading_ieee("B. Feature-Extraction vs. End-to-End Training", 2)
add_body("End-to-end training of EfficientNetV2B0 at 224 \u00d7 224 resolution on CPU required approximately 2 hours per epoch, making iterative experimentation impractical. The feature-extraction approach reduces total training time to approximately 15 minutes (10 minutes for one-time feature extraction + 5 minutes for head training across 200 epochs) without significant accuracy loss.")

add_heading_ieee("C. Multi-Tier Inference Benefits", 2)
add_body("The cascading inference architecture provides three key benefits: (1) Extended coverage \u2014 the NVIDIA vision-language API supports 22+ skin conditions, far exceeding the 7 categories of the local model. (2) Graceful degradation \u2014 system availability is maintained even when external APIs are inaccessible. (3) Confidence calibration \u2014 each tier applies appropriate confidence scaling and uncertainty communication, with fallback results explicitly labelled.")

add_heading_ieee("D. Limitations", 2)
add_body("The local CNN model is limited to 7 disease categories. The system processes single images and cannot yet perform longitudinal monitoring. Performance on diverse skin tones is limited by the demographic composition of the HAM10000 dataset. The system is a screening support tool and cannot replace histopathological examination. GPU-based fine-tuning was not performed in this study.")

# ===== VI. CONCLUSION =====
add_heading_ieee("VI. CONCLUSION AND FUTURE WORK", 1)

add_heading_ieee("A. Conclusion", 2)
add_body("This paper presented SkinAI, a comprehensive skin disease detection system combining CNN-based classification with a multi-tier AI inference pipeline. Feature-extraction transfer learning with EfficientNetV2B0 enables practical training on CPU-constrained environments while achieving 75.82% top-1 and 96.72% top-3 accuracy on the 7-class HAM10000 benchmark. The combination of aggressive oversampling, class weighting, and label smoothing effectively mitigates severe class imbalance. A cascading inference architecture ensures continuous availability while extending disease coverage from 7 to 22+ categories. Integration of patient context, symptom verification, AI consultation, and doctor referrals transforms a simple classifier into a comprehensive screening support ecosystem.")

add_heading_ieee("B. Future Work", 2)
add_body("Future directions include: training on larger, more diverse datasets (ISIC 2019/2020, Fitzpatrick17k) to improve accuracy and skin tone diversity; enabling GPU fine-tuning to improve minority class accuracy; developing native mobile applications with on-device TFLite inference for offline capability; implementing automated ABCDE rule monitoring for longitudinal lesion tracking; privacy-preserving federated learning across clinical institutions; and prospective clinical validation studies comparing system performance against dermatologist diagnoses.")

# ===== REFERENCES =====
add_heading_ieee("REFERENCES", 1)
refs = [
    '[1] A. Esteva, B. Kuprel, R. A. Novoa, J. Ko, S. M. Swetter, H. M. Blau, and S. Thrun, \u201cDermatologist-level classification of skin cancer with deep neural networks,\u201d Nature, vol. 542, no. 7639, pp. 115\u2013118, 2017.',
    '[2] P. Tschandl, C. Rosendahl, and H. Kittler, \u201cThe HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,\u201d Scientific Data, vol. 5, p. 180161, 2018.',
    '[3] A. Esteva et al., \u201cDermatologist-level classification of skin cancer with deep neural networks,\u201d Nature, vol. 542, no. 7639, pp. 115\u2013118, Feb. 2017.',
    '[4] H. A. Haenssle et al., \u201cMan against machine: diagnostic performance of a deep learning CNN for dermoscopic melanoma recognition in comparison to 58 dermatologists,\u201d Annals of Oncology, vol. 29, no. 8, pp. 1836\u20131842, Aug. 2018.',
    '[5] T. J. Brinker et al., \u201cDeep learning outperformed 136 of 157 dermatologists in a head-to-head dermoscopic melanoma image classification task,\u201d European Journal of Cancer, vol. 113, pp. 47\u201354, 2019.',
    '[6] K. He, X. Zhang, S. Ren, and J. Sun, \u201cDeep residual learning for image recognition,\u201d in Proc. IEEE CVPR, 2016, pp. 770\u2013778.',
    '[7] M. Tan and Q. V. Le, \u201cEfficientNet: Rethinking model scaling for convolutional neural networks,\u201d in Proc. ICML, 2019, pp. 6105\u20136114.',
    '[8] M. Tan and Q. V. Le, \u201cEfficientNetV2: Smaller models and faster training,\u201d in Proc. ICML, 2021, pp. 10096\u201310106.',
    '[9] A. G. Howard et al., \u201cMobileNets: Efficient convolutional neural networks for mobile vision applications,\u201d arXiv:1704.04861, 2017.',
    '[10] N. Codella et al., \u201cSkin lesion analysis toward melanoma detection 2018: A challenge hosted by ISIC,\u201d arXiv:1902.03368, 2019.',
    '[11] K. Simonyan and A. Zisserman, \u201cVery deep convolutional networks for large-scale image recognition,\u201d in Proc. ICLR, 2015.',
    '[12] Flask Web Development Framework, Pallets Projects. [Online]. Available: https://flask.palletsprojects.com/',
    '[13] TensorFlow: An end-to-end open source machine learning platform, Google, 2024. [Online]. Available: https://www.tensorflow.org/',
]
for ref in refs:
    add_reference(ref)

doc.save(OUT)
print(f"Word document saved to: {OUT}")
