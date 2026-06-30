# AgriCrop-Geospatial-Plant-Disease-Soil-Moisture-Intelligence-Network
AgriCrop is an AI-powered smart agriculture platform that detects plant diseases from leaf images, predicts soil moisture levels, and visualizes crop health trends using geospatial mapping.

# 🌱 AgriCrop: Geospatial Plant Disease & Soil Moisture Intelligence Network

AgriCrop is an AI-powered smart agriculture platform that helps farmers detect plant diseases early, predict soil moisture levels, and monitor crop health geographically through an interactive dashboard.

## 🚀 Features

- 🌿 Plant Disease Detection using MobileNetV2 and PlantVillage Dataset
- 💧 Soil Moisture Prediction using Machine Learning Regression
- 🗺️ Geospatial Crop Disease Dashboard using Leaflet.js
- 📤 Leaf Image Upload and Disease Prediction
- 🗄️ MongoDB Integration for Prediction Storage
- 📊 Interactive Disease Monitoring Dashboard
- 📈 Real-time Prediction Results and History

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Leaflet.js

### Backend
- Python
- Flask

### Machine Learning
- TensorFlow
- Keras
- MobileNetV2
- Scikit-learn

### Database
- MongoDB

---

## 📂 Project Structure


AgriCrop/

│
├── app.py

├── train_disease.py

├── predict.py

├── train_soil.py

├── database.py

├── requirements.txt

├── classes.json

│

├── models/

│   ├── disease_model.h5

│   └── soil_model.pkl

│

├── dataset/

│   ├── PlantVillage Dataset/

│   └── Plant_Parameters.csv

│

├── static/

│   ├── uploads/

│   ├── css/

│   └── js/

│

└── templates/

    ├── index.html
    
    ├── result.html
    
    └── dashboard.html
```

---

## 📊 Dataset

### Plant Disease Dataset
- PlantVillage Dataset
- 54,000+ images
- 38 disease classes

### Soil Dataset
- Plant_Parameters.csv
- Used for soil moisture prediction

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/AgriCrop.git
cd AgriCrop
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Train Disease Model

```bash
python train_disease.py
```

### Train Soil Model

```bash
python train_soil.py
```

### Start Flask Application

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 🎯 Future Enhancements

- Real-time weather API integration
- Multi-language farmer support
- Mobile application development
- Disease severity estimation
- Automated treatment recommendations
- Satellite-based crop monitoring

---

## 👩‍💻 Author

**Himabindu Karri**
B.Tech Computer Science Engineering
---

## ⭐ Project Objective

To empower farmers with AI-driven tools for early disease detection, intelligent soil monitoring, and geospatial crop health visualization, enabling better agricultural decision-making and improved crop productivity.
