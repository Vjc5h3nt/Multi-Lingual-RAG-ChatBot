"""
Model pricing configuration for AWS Bedrock models.
Update this file when switching models or when AWS updates pricing.
"""

# AWS Bedrock Model Pricing (per 1,000 tokens)
# Source: https://aws.amazon.com/bedrock/pricing/
# Last updated: 2024

MODEL_PRICING = {
    # Claude 3 Models
    "anthropic.claude-3-sonnet-20240229-v1:0": {
        "input": 0.003,   # $0.003 per 1K input tokens
        "output": 0.015,  # $0.015 per 1K output tokens
    },
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "input": 0.00025,  # $0.00025 per 1K input tokens
        "output": 0.00125, # $0.00125 per 1K output tokens
    },
    "anthropic.claude-3-opus-20240229-v1:0": {
        "input": 0.015,   # $0.015 per 1K input tokens
        "output": 0.075,  # $0.075 per 1K output tokens
    },
    
    # Claude 3.5 Models
    "anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "input": 0.003,   # $0.003 per 1K input tokens
        "output": 0.015,  # $0.015 per 1K output tokens
    },
    
    # Add more models as needed
}


def get_model_pricing(model_id: str) -> dict:
    """
    Get pricing for a specific model.
    
    Args:
        model_id: The AWS Bedrock model ID
        
    Returns:
        dict with 'input' and 'output' pricing per 1K tokens
        
    Raises:
        ValueError: If model pricing is not configured
    """
    if model_id not in MODEL_PRICING:
        raise ValueError(
            f"Pricing not configured for model: {model_id}. "
            f"Please add pricing to app/pricing.py. "
            f"Available models: {list(MODEL_PRICING.keys())}"
        )
    return MODEL_PRICING[model_id]


def calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> dict:
    """
    Calculate cost for a model invocation.
    
    Args:
        model_id: The AWS Bedrock model ID
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        dict with input_cost, output_cost, and total_cost
    """
    pricing = get_model_pricing(model_id)
    
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }
