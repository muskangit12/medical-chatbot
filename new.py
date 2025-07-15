import os
from huggingface_hub import whoami

# Load your .env file if you haven't already
from dotenv import load_dotenv
load_dotenv()

# Get and verify token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    print("❌ Error: No token found in .env file")
else:
    try:
        user_info = whoami(token=hf_token)
        print(f"✅ Token is valid! Logged in as: {user_info['name']}")
        print(f"Token permissions: {user_info['auth']['accessToken']['role']}")
    except Exception as e:
        print(f"❌ Token verification failed: {str(e)}")