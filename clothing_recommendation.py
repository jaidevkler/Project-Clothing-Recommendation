import streamlit as st
import requests
from PIL import Image
import io
import numpy as np

from apply_unet_model import apply_unet_model
from get_bounding_images import get_bounding_images
from image_to_text import image_to_text
from search_recommendations import google_search

def get_recommendations(image):
    image = image.resize((384, 384))
    #image = tf.image.resize(image, (384, 384))  # Ensure the image is resized to 384x384 if necessary
    #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Model path where the model is saved
    model_path = '../Models/segmentation_model.h5'
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
    recommendations = []
    # Loop through the texts and run a seach 
    for text in texts:
       recommendations.append(google_search(text))

    print(recommendations)
    # Return the list of recommendations
    return recommendations, categories

def main():
    # Display header of streamlit app
    st.write("# Code & Couture")
    st.write("#")
    st.write("### Upload an image that inspires you! ")
    uploaded_file = st.file_uploader(label='', type=['png', 'jpg'])

    if uploaded_file is not None:
      # Get the image
      image = Image.open(uploaded_file)
      # Display the image
      st.image(image)
      # Get recommendations
      recommendations, categories = get_recommendations(image)
      # Loop through recommendations
      for recommendation in recommendations:
        # List of pictures to be displayed
        pictures=[]
        # Loop through the recommendations
        for i in range(len(recommendation)):
          # url of the picture
          pic_url = recommendation['thumbnail'][i]
          # Save picture to list
          pictures.append(Image.open(requests.get(pic_url, stream=True).raw))

        # Create three columns
        col1, col2, col3 = st.columns(3)
        # Get the length of the recommendation
        if len(recommendation) > 3:
           length = 3
        else:
            length = len(recommendation)
        # Loop through the search results
        for i in range(length):
            # Get the image
            image = pictures[i]
            # Column 1
            if i%3 == 0:
                col1.image(image)
                col1.write(f'[{recommendation['title'][i]}]({recommendation['product_link'][i]})')
                col1.write(recommendation['price'][i])
            # Column 2
            elif i%3 == 1:
                col2.image(image)
                col2.write(f'[{recommendation['title'][i]}]({recommendation['product_link'][i]})')
                col2.write(recommendation['price'][i])
            # Column 3
            else:
                col3.image(image)
                col3.write(f'[{recommendation['title'][i]}]({recommendation['product_link'][i]})')
                col3.write(recommendation['price'][i])


if __name__ == "__main__":
  main()
