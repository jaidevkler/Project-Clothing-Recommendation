import pandas as pd
import os

from dotenv import load_dotenv
from serpapi import GoogleSearch

# Get recommendation from google search
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

# Google search function
def google_search(text,budget,additional_info, brand):
    # Add additional information to the text
    if brand != '':
        text = f"Women's Brand: {brand} {text}, Additonal information: {additional_info}"
        print(text)
    else:
        text = f"Women's {text}, Additonal information: {additional_info}"
    # Get recommendations
    print(text)
    recommendation = get_recommendations(text)
    # Add float price column
    recommendation['float_price'] = recommendation['price'].apply(lambda x: float(x.replace('$','').replace(',','')))
    recommendation = recommendation[(recommendation['float_price'] < float(budget)*1.25)]\
                        .sort_values(by='float_price', ascending=False)\
                        .reset_index(drop=True)
    
    # Return recommendation dataframe
    return recommendation

# Main function for testin
def main():
    print(google_search('White adidas shoes - Stan Smith',100,'',''))

if __name__ == '__main__':
    main()