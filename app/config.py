import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
AWS_PROFILE = os.getenv("AWS_PROFILE")

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")