import os
import base64
import requests
from dotenv import load_dotenv

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')


def image_to_text(image_path, category):
    if category == 'upper':
        text = f"{category.capitalize()} clothing gender, color, type, length, fit, style, details, accents, neckline, sleeve, pattern. Message should only include these details. Don't inlude none categories."
    elif category == 'lower':
       text = f"{category.capitalize()} clothing gender, color, type, length, fit, style, material/fabric, pattern, accents. Message should only include these details. Don't inlude none categories."
    else:
       text = f"{category.capitalize()} gender, color, brand, collection, type, style, material, toe shape, heel type, sole type, fastening, pattern, accents, purpose. Message should only include these details"
    # Load OpenAI api key
    load_dotenv() 
    api_key = os.getenv("OPENAI_API_KEY")
    # Getting the base64 string
    base64_image = encode_image(image_path)
    # Header
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }
    # Payload
    payload = {
      "model": "gpt-4o",
      #"temperature": 0.0,
      "messages": [
        { 
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": text
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
    }
    # Response
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    # Return the text
    return response.json()['choices'][0]['message']['content']\
                        .replace('*', '')\
                        .replace('- ', ', ')\
                        .strip()\
                        .replace('\n',',')
