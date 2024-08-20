import pandas as pd
import os

from dotenv import load_dotenv
from serpapi import GoogleSearch

def get_recommendations(item):
    # Load environment variables and API key
    load_dotenv()   
    # Create query for search
    query = f"Shop for Women's {item}"
    # Search google shopping
    search = GoogleSearch({
        "q": query, 
        "api_key": os.getenv("GOOGLE_SEARCH_API_KEY"),
        "tbm": "shop",  # shop
        #"tbs": "p_ord:rv",
        "num": 100
        })
    # Convert results into a dataframe
    recommendations = pd.DataFrame(search.get_dict()['shopping_results'])
    # Return recommendation
    return recommendations

def google_search(text):
    # Get recommendations
    recommendation = get_recommendations(text)
    # Filter recommendations
    recommendation = recommendation[(recommendation['rating'] > 4) & (recommendation['reviews'] > 10)].reset_index(drop=True)
    # Return recommendation dataframe
    return recommendation