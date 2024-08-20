from apply_unet_model import apply_unet_model
from get_bounding_images import get_bounding_images
from image_to_text import image_to_text
from search_recommendations import google_search
# Remove later
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

def main():
    # Model path where the model is saved
    model_path = '../Models/segmentation_model.h5'
    # Image path where test image is saved
    img_path = 'Resources/test_Images/test_image_001.png'
    # Load image
    image = load_img(img_path, target_size=(384,384))  # Replace height and width with your model's input size
    # Call the apply_unet_model function
    predicted_mask = apply_unet_model(model_path, image)
    # Print the class objects
    print(f'\nThe predicted mask created had the following class objects:')
    print(*np.unique(predicted_mask).tolist(),sep=', ')
    print()
    # Get bounding images
    categories = get_bounding_images(predicted_mask, image)
    # List of texts
    texts = []
    for category in categories:
      # Create image path
      image_path = f"Output/images/{category}.png"
      # Rund the iamge to text function with OpenAI
      texts.append(image_to_text(image_path, category))
    # List of recommendations
    recommedations = []
    # Loop through the texts and run a seach 
    for text in texts:
       print(text)
       recommedations.append(google_search(text))


    for recommendation in recommedations:
       print(recommendation)


if __name__ == "__main__":
  main()
