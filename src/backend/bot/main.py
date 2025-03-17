from dotenv import load_dotenv
import os

# Load variables from the .env file
load_dotenv()

# Retrieve the API key from the environment
API_KEY = os.getenv("POLYGON_API_KEY")
