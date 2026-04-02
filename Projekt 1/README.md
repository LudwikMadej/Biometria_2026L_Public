# 🖼️ Image Processing App

An interactive web application for image processing, built in Python with Streamlit. Supports loading images and applying a wide range of operations - from basic pixel transformations, through filters and edge detection, to morphological operations and brightness projections.



## 👥 Authors

- Maciej Andrzejewski
- Ludwik Madej



## ✨ Features

### Pixel Operations

- Grayscale conversion, inversion, binarization
- Brightness and contrast adjustment
- Histogram equalization, brightness stretching
- Logarithmic and power (gamma) transformation

### Filters

- Averaging and Gaussian blur
- Sharpening, Laplacian operator
- Custom user-defined filters

### Edge Detection

- Sobel, Scharr, Prewitt operators
- Roberts Cross

### Morphological Operations

- Erosion, dilation, opening, closing
- Morphological gradient, top-hat, black-hat
- Skeletonization, morphological reconstruction, hit-or-miss

### Projections

- Brightness projections along X and Y axes

### Image Statistics

- RGB channel histogram viewer
- Metadata table: dimensions, contrast, min/max/mean/median brightness



## 🚀 Getting Started

### Requirements

- Python 3.10

### Steps

1. **Clone the repository and navigate to the project directory:**
   
   ```bash
   git clone <repository-url>
   cd Projekt1
   ```

2. **Create and activate a virtual environment:**
   
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install dependencies:**
   
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   
   ```bash
   streamlit run frontend/main.py
   ```

5. On first launch, Streamlit will ask for an email address - skip it by pressing **Enter**.

6. The app will open automatically in your default browser.





## 🛠️ Tech Stack

| Library          | Purpose                                                   |
| ---------------- | --------------------------------------------------------- |
| **NumPy**        | Vectorized matrix operations — the core of all algorithms |
| **Pillow (PIL)** | Image loading and saving                                  |
| **Streamlit**    | Interactive web interface                                 |

All image processing algorithms were implemented from scratch using NumPy arrays - without relying on dedicated libraries such as OpenCV.
