# Handwritten Digits Classifier using CNN

This project demonstrates a **Convolutional Neural Network (CNN)** built and deployed using **FastAPI** to classify handwritten digits from the **MNIST dataset**.  
It showcases both **AI model development** and **API deployment**, allowing users to upload digit images for real-time classification.

---

## Project Overview

- **Dataset:** MNIST (70,000 grayscale images of handwritten digits from 0–9)  
- **Frameworks:** TensorFlow/Keras for model training, FastAPI for deployment  
- **Objective:** Build and serve a deep learning model capable of accurately classifying handwritten digits.  
- **Business Goal:**  
  - Automate digit recognition (useful in postal systems, bank check processing, form reading, etc.)  
  - Demonstrate a deployable AI service using FastAPI and TensorFlow  
  - Enable real-world integration of AI models through REST APIs  

---

## Features

- CNN-based digit classification
- REST API built with FastAPI
- Image upload and prediction endpoint
- Model trained and saved as `.keras` file
- Easily deployable using Uvicorn

---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "Handwritten Digits Classifier using CNN"
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI App
```bash
uvicorn main:app --reload
```
Then open the following URL in your browser:
```
http://127.0.0.1:8000/docs
```
You can use the Swagger UI to upload a handwritten digit image and get a prediction.

---

## Model Architecture

- Input Layer: 28x28 grayscale image
- Conv2D + MaxPooling layers for feature extraction
- Dense layers for classification
- Output layer: 10 neurons (digits 0–9)

---

## Test Image

!(test.png)


## Performance

- Accuracy: ~99% on MNIST test set  
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Epochs: 5–10 (adjustable)

---

## Directory Structure

```
Handwritten Digits Classifier using CNN/
│
├── main.py                  # FastAPI server file
├── mnist_cnn/               # Saved model file upon running
├── test.png                 # Test image for upload
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── venv/                    # Virtual environment (excluded from Git)
```

---

## FastAPI Endpoints

| Endpoint | Method | Description |
|-----------|--------|-------------|
| `/` | GET | Welcome route |
| `/predict` | POST | Upload an image for digit prediction |

---

## Business Goals Achieved

This project demonstrates how **machine learning models can be integrated into production environments** through APIs.  
It serves as a foundational template for automating recognition systems in:
- Banking (check number recognition)  
- Postal and logistics digit scanning  
- Handwritten form analysis  
- OCR-based document processing  

---

## Requirements

Dependencies are listed in `requirements.txt`. You can download them using:
```bash
pip install -r requirements.txt
```

---

## Future Improvements

- Add model retraining via API  
- Include frontend dashboard for visual predictions  
- Support multi-digit image recognition  

---

## Author

**Muzammil Sohail**  
AI Engineer | Machine Learning & Deep Learning Enthusiast  
📧 [muzammilsohail1718@gmail.com]

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
