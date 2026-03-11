import json
import boto3
from typing import List
from app.models import Document
from app.config import AWS_REGION, AWS_PROFILE, EMBEDDING_MODEL_ID

class BedrockEmbedder:

    def __init__(self):
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        self.client = session.client("bedrock-runtime")

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        embeddings = []

        for doc in documents:
            payload = {"inputText": doc.content}

            response = self.client.invoke_model(
                modelId=EMBEDDING_MODEL_ID,
                body=json.dumps(payload).encode("utf-8"),
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response["body"].read())
            embeddings.append(response_body["embedding"])

        return embeddings
