import serpapi
import os 
from dotenv import load_dotenv
import re
from transformers import pipeline
load_dotenv()

api_key = os.getenv("SERPAPI")
client = serpapi.Client(api_key=api_key)
ner = pipeline("ner", "dslim/bert-large-NER", tokenizer="dslim/bert-large-NER", aggregation_strategy="simple")

# function which performs search at ticketmaster website
def look_for_artists(artist_name): 
    result = client.search(
        q=f"site:ticketmaster.com {artist_name}",
        engine="google",
        language="en"
    )
    return result

# function for extraction specific information from the look_for_artist function
def extract_tour_data(results):
    extracted_events  = []
    raw_events  = results.get("organic_results", []) # extracts list of events

    for event in raw_events :
        title = event.get("title") # receive title
        link = event.get("link") # receive link to the event
        
        # this block of code looks if date exists in returned information. If not, it tries to get the date from the link if it is there, otherwise date set to Not Available
        if event.get("date"):
            date = event.get("date") 
        else:
            date_search = r"\d{2}-\d{2}-\d{4}"
            match = re.search(date_search, link)
            if match:
                date = match.group()
            else:
                date = "The date is not available yet"
         
        # this block of code tries to get locations from the link by NER model. For better recognition by model performance the split and word capitalization was used.     
        joined_link = " ".join([word.capitalize() for word in link.split("-")])
        location_entities = [ent["word"] for ent in ner(joined_link) if "LOC" in ent["entity_group"]]
        if location_entities:
            location = " ".join(location_entities)
        else:
            location = "Location is not stated"

        # appending all information received
        if title and link:
            extracted_events.append({
                "title": title,
                "link": link,
                "date": date,
                "location": location
            })
            
    return extracted_events




    