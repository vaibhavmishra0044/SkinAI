# Skin Disease Detection System

## Project Overview
This project utilizes Convolutional Neural Networks (CNN) to develop an automated skin disease detection system. By leveraging image-based techniques, the system aims to improve accuracy and efficiency in diagnosing various skin diseases.

---

## Problem Statement
Traditional methods for diagnosing skin diseases are prone to inaccuracies and delays. This project addresses these challenges by creating an automated solution capable of early and precise detection.

---

## Objectives
- Develop a CNN-based system for skin disease classification.
- Enhance the accuracy of early diagnosis.
- Automate and streamline the detection process for faster results.

---

## State of the Art
- Current approaches involve manual examination, which is subjective and time-consuming.
- Existing solutions often fail to handle diverse datasets effectively.
- Limited adoption of advanced machine learning techniques impacts the precision of diagnosis.

---

## Methodology
### Data Flow/Architecture
1. **Dataset Collection:** Gather diverse skin image datasets.
2. **Preprocessing:** Normalize and augment data for robustness.
3. **Model Training:** Train a CNN model tailored for classification.
4. **Evaluation:** Use metrics like accuracy, precision, and recall to assess performance.

---

## Results
- Achieved significant improvement in classification accuracy.
- Visualized performance through graphs and confusion matrices.
- Demonstrated robustness across various skin conditions.

---

## Potential Findings
- 15% improvement in diagnostic accuracy compared to baseline methods.
- Scalability and reliability for integration into real-world applications.

---

## Applications
- Diagnosis support in dermatology clinics.
- Integration with telemedicine for remote consultations.
- Reduction in hospital readmission rates by 20% through early diagnosis.

---

## How to Run
1. Clone the repository.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the preprocessing script to prepare the dataset.
4. Execute the training script to train the CNN model.
5. Use the prediction script to classify new images.

---

## Training (Higher Accuracy)
This repo includes a stronger training pipeline using EfficientNetV2B0 on the HAM10000 dataset.

### Dataset Setup (HAM10000)
1. Download the HAM10000 dataset and metadata CSV.
2. Create this folder structure:

```
data/
	ham10000/
		HAM10000_metadata.csv
		images/                 (or)
		HAM10000_images_part_1/
		HAM10000_images_part_2/
```

The training script will scan those folders and match images with metadata.

If your dataset is already extracted under the repo's `archive/` folder, the training script will auto-detect it. You can also point to a custom location by setting `HAM10000_DIR`.

### Train
```
python train.py
```

Outputs:
- `CNN1_trained_model.keras` (best model)
- `class_names.json` (label order used at inference time)

After training, run the app as usual:
```
python app.py
```

---

## Prerequisites
- Python 3.8 or higher
- TensorFlow 2.x
- NumPy
- Matplotlib

---

## Contributors
- # Skin Disease Detection System

## Project Overview
This project utilizes Convolutional Neural Networks (CNN) to develop an automated skin disease detection system. By leveraging image-based techniques, the system aims to improve accuracy and efficiency in diagnosing various skin diseases.

---

## Problem Statement
Traditional methods for diagnosing skin diseases are prone to inaccuracies and delays. This project addresses these challenges by creating an automated solution capable of early and precise detection.

---

## Objectives
- Develop a CNN-based system for skin disease classification.
- Enhance the accuracy of early diagnosis.
- Automate and streamline the detection process for faster results.

---

## State of the Art
- Current approaches involve manual examination, which is subjective and time-consuming.
- Existing solutions often fail to handle diverse datasets effectively.
- Limited adoption of advanced machine learning techniques impacts the precision of diagnosis.

---

## Methodology
### Data Flow/Architecture
1. **Dataset Collection:** Gather diverse skin image datasets.
2. **Preprocessing:** Normalize and augment data for robustness.
3. **Model Training:** Train a CNN model tailored for classification.
4. **Evaluation:** Use metrics like accuracy, precision, and recall to assess performance.

---

## Results
- Achieved significant improvement in classification accuracy.
- Visualized performance through graphs and confusion matrices.
- Demonstrated robustness across various skin conditions.

---

## Potential Findings
- 15% improvement in diagnostic accuracy compared to baseline methods.
- Scalability and reliability for integration into real-world applications.

---

## Applications
- Diagnosis support in dermatology clinics.
- Integration with telemedicine for remote consultations.
- Reduction in hospital readmission rates by 20% through early diagnosis.

---

## How to Run
1. Clone the repository.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the preprocessing script to prepare the dataset.
4. Execute the training script to train the CNN model.
5. Use the prediction script to classify new images.

---

## Prerequisites
- Python 3.8 or higher
- TensorFlow 2.x
- NumPy
- Matplotlib

---

## Contributors
- [Tanush Balaraj Alva (4MT22AI402)]
- [Dhanush M Shetty (4MT21AI018)]
- [Ismail Sinan (4MT21AI023)]
- [Prajwal J Rai (4MT21AI037)]

---

## Acknowledgments
This project was guided by [Dr. Yogeesha C B] at the Department of Artificial Intelligence and Machine Learning, Mangalore Institute of Technology & Engineering (MITE).
