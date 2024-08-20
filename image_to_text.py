import os
import base64
import requests
from dotenv import load_dotenv

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')


def image_to_text(image_path, category):
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
      "model": "gpt-4o-mini",
      "messages": [
        { 
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"{category} clothing color, type, length, fit, style. Message should only include these two details"
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

    return response.json()['choices'][0]['message']['content'].replace('\n',', ')
    
