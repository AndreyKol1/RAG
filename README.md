# ProvectusInternship project AndriiKoliush

# Approach and design choices

I decided to start with UI at first, because it much easier to see what you do and how everything working and connected. The UI contain two main fields: question filed and document uploading field. I decided to change a little bit execution of first task by avoiding querying "upload documents to database" by simply adding **Process** button, which does it. The process of checking document relevance is divided into two stages: NER and keyword matching. Large Bert gives most of the times meaningful information. Of course, for other topic it will also return information about ["PER", "LOC", "ORG", "MISC"], but I decided to add it as additional check, because seemed interesting to me to use it in that way. Main check is a keyword matching, which simply looks for keywords, which are related to the concert topic. After processing the document the summarization is given to the user, all documents are saved in ChromaDB, and the LLM, which will answer user questing, is instantiated. For summarization my choice was bart-large because it gave the most meaningful summary from light-weight models. ChromaDB was chosen because of it flexibility and easy to use. For LLM flan-t5-base was picked because it fits the same needs as summarization model. For bonus task, SerpAPi was chosen because of it user-friendly and simple documentation and free tier for 100 requests per month, which allows to test it properly. 


# Project set-up

First of all, you need to create virtual environment and install all dependencies from requirements file by running command **pip install -r requirements.txt**. Secondly, you need to have two APIs: Hugging Face and SerpAPI. The link for Hugging Face: https://huggingface.co/settings/tokens. Create a new token with a Read mode. The link for SerpAPI: https://serpapi.com/manage-api-key. Copy your private token. Then, you need to create .env file and assign tokens to them. For Hugging Face name should be HUGGINGFACEHUB_API_TOKEN and for SerpAPI SERPAPI. 

# Running a project

In command line with a directory of your project run command **streamlit run app.py**. The project should start and you will be able to test it.