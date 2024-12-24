from transformers import pipeline
from huggingface_hub import login
import re

# Authenticate using Hugging Face API key
login(token='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')  # Use your actual API key

# Define keywords associated with each area
area_keywords = {
    'dining': ['dining', 'food', 'meal', 'restaurant', 'cold', 'terrible'],
    'reception': ['reception', 'check-in', 'staff', 'service', 'friendly']
}

# Sentiment analysis model setup
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_feedback(feedback):
    # Split feedback into sentences
    sentences = re.split(r'(?<=[.!?])\s+', feedback)
    
    # Dictionary to store sentiment for each area
    area_sentiments = {area: 'POSITIVE' for area in area_keywords}

    # Analyze each sentence for sentiment
    for sentence in sentences:
        for area, keywords in area_keywords.items():
            if any(keyword in sentence.lower() for keyword in keywords):
                prompt = f"Please analyze the following sentence for sentiment: '{sentence}' for {area} experience"
                result = classifier(prompt)
                sentiment = result[0]['label']

                # Update sentiment if it's negative for that specific area
                if sentiment == 'NEGATIVE':
                    area_sentiments[area] = sentiment

    # Determine which area has the most negative sentiment
    negative_areas = [area for area, sentiment in area_sentiments.items() if sentiment == 'NEGATIVE']
    
    if negative_areas:
        return f"The area(s) responsible for the negative review are: {', '.join(negative_areas)}"
    else:
        return "The feedback is positive for both dining and reception."

def main():
    # Accepting feedback input from the user
    feedback = input("Enter feedback: ")

    # Analyze feedback
    result = analyze_feedback(feedback)

    # Printing the result
    print(result)

if __name__ == "__main__":
    main()

