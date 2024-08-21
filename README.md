## CODE & COUTURE

**UNet Segmentation and Recommendation Pipeline - Project 3 group #7:**
Jaidev Kler, Christine Chung, Alan Khalili, Emmanuel Charles, Enock Mudzamiri, Grigoriy Isayev, Daniyar Temirkhanov


## Project Overview

This project presents a robust image segmentation pipeline utilizing a UNet model with a MobileNetV2 backbone. The solution is designed for high-performance image segmentation tasks, enabling detailed analysis and processing of visual data for applications in fashion retail and e-commerce. The pipeline integrates advanced machine learning techniques with cutting-edge APIs, providing a complete solution from model training to actionable insights and product recommendations.

### Key Features
- **High-Performance Segmentation**: Utilizes UNet architecture with MobileNetV2 for efficient and accurate segmentation suitable for real-time applications.
- **Data Augmentation**: Comprehensive data augmentation techniques are employed to enhance model robustness and generalization.
- **API Integration**: Seamless integration with OpenAI and SerpApi for enhanced data analysis and product recommendations.
- **Scalable and Modular**: Designed to be scalable and easily customizable for various industrial applications.

## Use Cases
- **Fashion Industry**: Automatic segmentation of clothing items from images for inventory management, product categorization, and virtual try-ons.
- **E-Commerce**: Enhance product search and recommendation engines by accurately identifying and categorizing products in images.
- **Retail Analytics**: Leverage segmentation for in-store analytics such as shopper behavior analysis and shelf monitoring.

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

- **Resources**: Contains the dataset of images and masks used for training and validation.
- **notebooks**: Jupyter notebooks with code for segmentation and recommendation tasks.
- **models**: Directory where trained models are saved.
- **README.md**: This file, providing an overview of the project.
- **requirements.txt**: Lists all Python packages required to run the project.

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- OpenAI API Key
- SerpApi Key

### Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/unet-segmentation-pipeline.git
   cd unet-segmentation-pipeline
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

4. **Set up environment variables for API keys:**
   ```sh
   export OPENAI_API_KEY="your-openai-api-key"
   export SERPAPI_KEY="your-serpapi-key"
   ```

5. **Download or prepare the dataset:**
   - Ensure that your dataset is placed in the `Resources/images` and `Resources/masks` directories.

## Pipeline Components

### 1. Dataset Preparation

**DatasetLoader**: Handles the end-to-end process of loading, preprocessing, and splitting the dataset into training and validation sets.

- **Input**: Image and mask datasets.
- **Output**: TensorFlow datasets ready for training.

### 2. Model Architecture

**ModelBuilder**: Constructs the UNet model with a MobileNetV2 backbone optimized for image segmentation tasks.

- **UNet Model**: Combines the strengths of the UNet architecture with the MobileNetV2 feature extractor for robust segmentation performance.

### 3. Training and Fine-Tuning

The model is trained using a combination of standard and fine-tuning phases. Early stopping and checkpointing are implemented to ensure the best model is saved.

- **Initial Training**: Trains the model with a frozen backbone.
- **Fine-Tuning**: Unfreezes the backbone and fine-tunes the model for additional epochs to improve performance.

### 4. Image Segmentation and Prediction

The trained model is used to segment new images, identifying key regions (e.g., clothing items). The segmentation masks are processed to extract bounding boxes for each class.

- **Bounding Box Extraction**: Extracts bounding boxes from the predicted segmentation masks to isolate individual objects or regions.

### 5. Integration with External APIs

**OpenAI API**: Generates descriptive information about segmented regions (e.g., clothing color, type, and length).

**SerpApi Integration**: Retrieves product recommendations based on the segmented clothing items, enabling seamless e-commerce integration.

- **Input**: Segmented images and descriptive metadata.
- **Output**: Product recommendations and detailed analysis.

### 6. Performance Monitoring

**PerformanceReport**: Provides tools for evaluating the model's performance, including loss and accuracy plots and confusion matrix visualizations.

- **Metrics**: Tracks key performance indicators such as accuracy, precision, recall, and F1-score.

## Running the Project

### Training the Segmentation Model

- Open the `clothing_segmentation.ipynb` notebook in Jupyter.
- Follow the instructions in the notebook to preprocess the data and train the segmentation model.
- The trained model will be saved in the `models` directory.

### Generating Clothing Recommendations

- Open the `clothing_recommendation.ipynb` notebook.
- Load the pre-trained model and run the cells to generate clothing recommendations based on segmented images.

### Visualization and Analysis

- Use the provided notebooks to visualize feature maps, Grad-CAM, and other aspects of the model to understand its behavior better.
- Analyze the recommendations to evaluate the model's performance.

## Deployment

This pipeline can be deployed in various environments, including cloud-based platforms and on-premise servers. It is designed to handle large-scale datasets and can be integrated into existing retail and e-commerce systems.

### Suggested Deployment Steps

- **Containerization**: Use Docker to containerize the application for consistent deployment across different environments.
- **Cloud Deployment**: Deploy the containerized application on cloud platforms like AWS, GCP, or Azure for scalability and availability.
- **API Integration**: Expose the segmentation and recommendation functionalities via RESTful APIs for easy integration with other services.

## Scalability and Customization

The pipeline is built to be highly scalable and customizable:

- **Scalability**: Capable of handling large datasets and can be parallelized across multiple GPUs.
- **Customization**: Modular design allows easy integration of alternative models, datasets, and APIs based on specific industry requirements.

## Acknowledgments

- The dataset used in this project was obtained from [[kaggle](https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation?select=labels.csv)].
- This project was developed as part of the AI Bootcamp at Columbia Engineering.


 
