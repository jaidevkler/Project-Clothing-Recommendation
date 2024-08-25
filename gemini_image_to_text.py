import google.generativeai as genai
from dotenv import load_dotenv
import os

def gemini_image_to_text(image, category):
    if category == 'upper':
        text = f"Describe {category.capitalize()} clothing, main color, type, material, length, fit, style, details, accents, neckline, sleeve, pattern. Message should only include these details. Don't inlude none categories. Start message with the category"
    
    elif category == 'lower':
       text = f"Describe {category.capitalize()} clothing, color, type, material, length, fit, style, pattern, accents. Message should only include these details. Don't inlude none categories. Start message with the category"
    else:
       text = f"Describe {category.capitalize()} main color, brand, collection, type, style, material, toe shape, heel type, sole type, fastening, pattern, accents, purpose. Message should only include these details. Start message with the category"
    # Load OpenAI api key
    load_dotenv() 
    api_key=os.getenv("GEMINI_API_KEY")
    # Configure the model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    # Get a response
    response = model.generate_content([image,text])
    # Format and return the response
    return response.text\
            .replace('*', '')\
            .replace('- ', '')\
            .replace('.', '')\
            .strip()\
            .replace('\n',',')
# Main function for testing
def main():
    image_path = "Output/images/shoes.png"
    category = "shoes"
    print(gemini_image_to_text(image_path, category))

if __name__ == "__main__":
    main()
