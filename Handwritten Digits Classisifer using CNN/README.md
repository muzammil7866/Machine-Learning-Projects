# 🧠 MNIST Digit Classification using CNN  

## 📌 Project Overview  
This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) from the **MNIST dataset**. The model achieves high accuracy by leveraging deep learning techniques.  

The project also includes an easy-to-run pipeline so anyone can train, evaluate, and test the model with minimal setup.  

---

## 🎯 Business Goals Achieved  
- ✅ Automated recognition of handwritten digits, which can be extended to **OCR (Optical Character Recognition)** systems.  
- ✅ Demonstrates the use of **deep learning** in solving **real-world image classification problems**.  
- ✅ Provides a baseline for integrating **AI-based digit recognition** into applications such as:  
  - Bank cheque processing  
  - Postal code reading  
  - Digital form automation  

---

## ⚙️ Tech Stack  
- **Python 3.11.5**  
- **TensorFlow / Keras** – Model building & training  
- **NumPy & Pandas** – Data processing  
- **Matplotlib** – Visualizations  
- **PIL (Pillow)** – Image handling  

---

## 🚀 How to Run the Project  

### 1. Clone the Repository  
```bash
git clone <your-repo-link>
cd mnist-cnn
```

### 2. Create Virtual Environment  
```bash
python -m venv venv
```

### 3. Activate Virtual Environment  
- **Windows (CMD):**
```bash
venv\Scripts\activate
```
- **Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```
- **Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 5. Run the Project  
```bash
python main.py
```

---

## 📊 Results  
- Achieved high test accuracy on MNIST dataset.  
- Demonstrated robust CNN performance on digit classification tasks.  

---

## 📂 Project Structure  
```
mnist-cnn/
│── main.py              # Main training & evaluation script
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
│── venv/                # Virtual environment (not for deployment)
```
