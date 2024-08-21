import google.generativeai as genai 
import PIL.Image 
import os  
import base64
from dotenv import load_dotenv

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

def gemini_image_to_text(image_path, category):
    if category == 'upper':
        text = f"Identify the following in the image: {category.capitalize()} clothing gender, color, type, length, fit, style, details, accents, neckline, sleeve, pattern. Message should only include these details. Don't inlude none categories."
    elif category == 'lower':
       text = f"Identify the following in the image:{category.capitalize()} clothing gender, color, type, length, fit, style, material/fabric, pattern, accents. Message should only include these details. Don't inlude none categories."
    else:
       text = f"Identify the following in the image:{category.capitalize()} gender, color, brand, collection, type, style, material, toe shape, heel type, sole type, fastening, pattern, accents, purpose. Message should only include these details"
    # Load OpenAI api key
    load_dotenv() 
    api_key = os.getenv("GEMINI_API_KEY")
    # Getting the base64 string
    base64_image = encode_image(image_path)
    # Configuring the API
    genai.configure(api_key=api_key) 
    # Creating the model
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    # Generating the content
    response = model.generate_content([text, base64_image]) 
    print(response.text)
    return response.text