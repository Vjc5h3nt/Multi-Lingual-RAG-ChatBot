import json
import boto3
from app.config import AWS_REGION, AWS_PROFILE, BEDROCK_MODEL_ID
from app.pricing import calculate_cost, get_model_pricing
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

class ClaudeClient:
    def __init__(self):
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        self.client = session.client("bedrock-runtime")
        self.model_id = BEDROCK_MODEL_ID
        self.pricing = get_model_pricing(self.model_id)

    @traceable(run_type="llm", name="AWS Bedrock Claude")
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(payload).encode("utf-8"),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response["body"].read())
        
        # Extract token usage from Bedrock response
        usage = response_body.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        # Calculate cost using centralized pricing
        costs = calculate_cost(self.model_id, input_tokens, output_tokens)
        
        # Update the current run with token usage and costs
        try:
            run = get_current_run_tree()
            if run:
                # Update run with token counts and costs
                # LangSmith will aggregate these for monitoring
                run.outputs = {
                    "output": response_body["content"][0]["text"],
                    "usage_metadata": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    }
                }
                
                # Add costs to metadata
                run.extra = {
                    "metadata": {
                        "model_name": self.model_id,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "input_cost_usd": costs["input_cost"],
                        "output_cost_usd": costs["output_cost"],
                        "total_cost_usd": costs["total_cost"],
                        "pricing_input_per_1k": self.pricing["input"],
                        "pricing_output_per_1k": self.pricing["output"],
                    }
                }
                
        except Exception as e:
            print(f"Warning: Could not update run metadata: {e}")
        
        # Return text and usage metadata
        return response_body["content"][0]["text"], {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
