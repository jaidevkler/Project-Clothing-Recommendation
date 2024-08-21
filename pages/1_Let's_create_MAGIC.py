import streamlit as st
import requests
from PIL import Image
import io
import numpy as np

from apply_unet_model import apply_unet_model
from get_bounding_images import get_bounding_images
from image_to_text import image_to_text
from search_recommendations import google_search

st.set_page_config(page_title="Let's Create Magic!", page_icon="✨")

def get_recommendations(image, budget):
    image = image.resize((384, 384))
    # Model path where the model is saved
    model_path = '../Models/segmentation_model.h5'
    # Call the apply_unet_model function
    print('\nApplying the UNet model to the image...')
    predicted_mask = apply_unet_model(model_path, image)
    # Get bounding images
    print('\nGetting bounding images...')
    categories = get_bounding_images(predicted_mask, image)
    # List of texts
    texts = []
    print('\nGetting text from the image...')
    for category in categories:
      # Create image path
      image_path = f"Output/images/{category}.png"
      # Rund the iamge to text function with OpenAI
      text = image_to_text(image_path, category)
      print(f'{category.capitalize()}: {text}')
      texts.append(text)
    # List of recommendations
    recommendations = []
    # Loop through the texts and run a seach 
    print('\nGetting search results...')
    count=0
    for text in texts:
       recommendations.append(google_search(text,budget[count]))
       count+=1
    # Return the list of recommendations
    return recommendations, categories

def main():
    # Display header of streamlit app
    st.write("# Code & Couture")
    # Image uploader
    uploaded_file = st.file_uploader(label='Upload an image that inspires you!', type=['png', 'jpg'])

    if uploaded_file is not None:
      # Get the image
      image = Image.open(uploaded_file)
      # Display the image
      col1, col2, col3 = st.columns([2,4,2])
      
      with col2:
        st.image(image)
      with col3:
         st.write("#### ")
         st.write("#### ")
         st.write("#### ")
         st.write("#### That's a really cool look!")
         st.write("#### Let's see what we can find for you...")
      
      col1, col2, col3 = st.columns(3)
      
      with col1:
        budget_upper = st.text_input(
        "Upper Clothing budget: ",
        "100",
        key="placeholder")
      with col2:
        budget_lower = st.text_input(
        "Lower Clothing budget: ",
        "100")
      with col3:
        budget_shoes = st.text_input(
        "Shoes budget: ",
        "100")
      if st.button("Let the magic begin", type="primary"):
        budget = [float(budget_upper), float(budget_lower), float(budget_shoes)]
        print(budget)
        with st.spinner('Wait for it...'):
          # Get recommendations
          recommendations, categories = get_recommendations(image, budget)
          # Displaying results
          print('\nDisplaying results...')
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
            st.write(f"#### {categories.pop(0).capitalize()}")
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
                    col1.image(image.resize((384, 384)))
                    col1.write(f'[{recommendation['title'][i]}]({recommendation['product_link'][i]})')
                    col1.write(recommendation['price'][i])
                # Column 2
                elif i%3 == 1:
                    col2.image(image.resize((384, 384)))
                    col2.write(f'[{recommendation['title'][i]}]({recommendation['product_link'][i]})')
                    col2.write(recommendation['price'][i])
                # Column 3
                else:
                    col3.image(image.resize((384, 384)))
                    col3.write(f'[{recommendation['title'][i]}]({recommendation['product_link'][i]})')
                    col3.write(recommendation['price'][i])
          st.success('Done!')
          st.balloons()       
          print('\nDone!')


if __name__ == "__main__":
  main()
