# Clothing-Segmentation-Recommendation

## Project Overview

This project focuses on clothing segmentation and recommendation using deep learning techniques. The primary goal is to segment clothing items from images and recommend similar items based on the segmented regions. The project involves working with image data, preprocessing, model training, and generating recommendations based on the learned features.

### Key Features
- **Clothing Segmentation:** Using deep learning models to accurately segment clothing items from images.
- **Feature Extraction:** Extracting features from the segmented images to understand and classify different clothing types.
- **Recommendation System:** Recommending similar clothing items based on the extracted features.

## Project Structure

```
├── Resources
│   ├── images
│   ├── masks
├── notebooks
│   ├── clothing_segmentation.ipynb
│   ├── clothing_recommendation.ipynb
├── models
│   ├── model.h5
├── README.md
├── requirements.txt
```

- **Resources:** Contains the dataset of images and masks used for training and validation.
- **notebooks:** Jupyter notebooks with code for segmentation and recommendation tasks.
- **models:** Directory where trained models are saved.
- **README.md:** This file, providing an overview of the project.
- **requirements.txt:** Lists all Python packages required to run the project.

## Setup Instructions

### Prerequisites

To run this project, you need the following installed on your system:
- Python 3.7 or higher
- pip (Python package manager)
- Virtualenv (recommended)

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/clothing-segmentation-recommendation.git
   cd clothing-segmentation-recommendation
   ```

2. **Create a virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Download or prepare the dataset:**
   - Ensure that your dataset is placed in the `Resources/images` and `Resources/masks` directories.

## Running the Project

### 1. Training the Segmentation Model

- Open the `clothing_segmentation.ipynb` notebook in Jupyter.
- Follow the instructions in the notebook to preprocess the data and train the segmentation model.
- The trained model will be saved in the `models` directory.

### 2. Generating Clothing Recommendations

- Open the `clothing_recommendation.ipynb` notebook.
- Load the pre-trained model and run the cells to generate clothing recommendations based on segmented images.

### 3. Visualization and Analysis

- Use the provided notebooks to visualize feature maps, Grad-CAM, and other aspects of the model to understand its behavior better.
- Analyze the recommendations to evaluate the model's performance.

## Visualizations

### Feature Maps
Feature maps from different convolutional layers help in understanding what the model learns at each stage.

### Grad-CAM
Grad-CAM visualizations show which parts of the image contribute most to the model's predictions.

### PCA
Principal Component Analysis is used to reduce the dimensionality of image data and visualize patterns in a 2D space.

## Results

The project successfully segments clothing items from images and provides relevant recommendations based on the extracted features. The visualizations and analysis show that the model captures important features of clothing items, leading to accurate and contextually relevant recommendations.

## Acknowledgments

- The dataset used in this project was obtained from [kaggle].
- This project was developed as part of the AI Bootcamp at Columbia Engineering.

#Team
Jaidev Kler, Christine Chung, Alan Khalili, Emmanuel Charles, Enock Mudzamiri, Grigoriy Isayev, Daniyar Temirkhanov
 
