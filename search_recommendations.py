import pandas as pd
import os

from dotenv import load_dotenv
from serpapi import GoogleSearch

def get_recommendations(item):
    # Load environment variables and API key
    load_dotenv()   
    # Create query for search
    query = f"Shop for {item}"#, Brand: Everlane"
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

def google_search(text,budget,additional_info, brand):
    # Add additional information to the text
    if brand is not '':
        text = f"Brand: {brand} {text}, Additonal information: {additional_info}"
        print(text)
    else:
        text = f"{text}, Additonal information: {additional_info}"
    # Get recommendations
    recommendation = get_recommendations(text)
    # Add float price column
    recommendation['float_price'] = recommendation['price'].apply(lambda x: float(x.replace('$','').replace(',','')))
    recommendation = recommendation[(recommendation['float_price'] < float(budget)*1.25)]\
                        .sort_values(by='float_price', ascending=False)\
                        .reset_index(drop=True)
    #print(recommendation)
    # Filter recommendations
    #recommendation = recommendation[(recommendation['rating'] > 4) & (recommendation['reviews'] > 10)].reset_index(drop=True)
    # Return recommendation dataframe
    return recommendation