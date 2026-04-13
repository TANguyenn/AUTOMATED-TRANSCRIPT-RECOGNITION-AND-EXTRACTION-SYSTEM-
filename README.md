# AUTOMATED-TRANSCRIPT-RECOGNITION-AND-EXTRACTION-SYSTEM
# 🎓 Automatic Transcript Extraction System using Deep Learning

## Introduction
This project develops an automated system for recognizing and extracting information from student transcripts using image inputs. The system consists of three main components:

- A Transformer-based model (DETR) for **table detection**  
- A Transformer-based model (DETR) for **table structure recognition**  
- An OCR model (VietOCR) for **character recognition**  

---

## Processing Pipeline
Input Image 
![0_0](https://github.com/user-attachments/assets/763ef70b-3dc9-4cc3-8ae8-0616d053ee41)

→ Table Detection (DETR)
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/4d886b8e-0465-462b-a79d-0cbbce33be04" />

→ Table Cropping 

→ Table Structure Recognition (Rows & Columns) 
<img width="413" height="506" alt="image" src="https://github.com/user-attachments/assets/616f4a7f-3254-4603-bc89-66960946a732" />

→ Cell Extraction (Row ∩ Column) 

→ Character Recognition (VietOCR) for each cell → Post-processing → Output Results

---

## Dataset
- Approximately **500 real transcript images**  
- Annotated using **Datatorch**  
- Format: **COCO (JSON)**  

### Preprocessing
- Convert images to grayscale  
- Image alignment  
- Noise removal  
- Size normalization  

---

## Model Evaluation

### Metrics used:
- **mAP (Mean Average Precision)**  
Table detection:
<img width="940" height="340" alt="image" src="https://github.com/user-attachments/assets/2e91a482-1607-4c30-9c8e-7a7f773f449e" />
- **AR (Average Recall)**  
Table structure recognition:
<img width="940" height="490" alt="image" src="https://github.com/user-attachments/assets/7fdb06fe-b8ef-4380-a8b7-2d8d584458da" />
- **OCR Accuracy**  
<img width="489" height="160" alt="image" src="https://github.com/user-attachments/assets/26df34fc-fe00-40cb-ad7d-2c7a7611a3fb" />

---

## Project Structure
project/
│
├── data/ # Sample data
├── models/ # Trained models
├── src/ # Processing pipeline
├── notebooks/ # Experiments
├── app/ # Streamlit interface
├── outputs/ # Results
└── README.md
