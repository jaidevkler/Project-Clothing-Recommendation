import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def apply_unet_model(path,image):
    # Load the model
    model = load_model(path)
    # Convert the image to an array
    img_array_org = img_to_array(image)
    # Add batch dimension
    img_array = np.expand_dims(img_array_org, axis=0)
    # Normalize the image
    img_array = img_array / 255.0
    # Predict image mask
    predicted_mask = model.predict(img_array)
    # Change dimension of the predicted mask   
    predicted_mask = np.squeeze(predicted_mask, axis=0)
    predicted_mask = np.argmax(predicted_mask, axis=-1)
    # Return predicted mask
    return predicted_mask

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

if __name__ == "__main__":
  main()