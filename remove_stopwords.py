import nltk
# Import stopwords from the nltk corpus 
from nltk.corpus import stopwords
# Import tokenizers and pandas
from nltk.tokenize import word_tokenize

def remove_stopwords(text):
    # Download the stopwords from the nltk package
    stop_words = set(stopwords.words('english'))
    # Tokenize the text into words
    word_tokens = word_tokenize(text)
    # Filter the text by removing the stopwords
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    # Return the filtered text
    return ' '.join(filtered_text)

def main():
    # Tokenize the document into sentences
    sentence = 'I am looking for something casual to wear for a wedding. I am looking for a dress that is comfortable and not too formal'
    # Remove stopwords from the document
    filtered_text = remove_stopwords(sentence)
    # Print the filtered text
    print(filtered_text)

if __name__ == '__main__':
    main()
    