"""
bedrockclient.py

Purpose: Transform structured JSON forecasting facts into concise, actionable B2B briefs for fashion and beauty enterprises
"""
import json
import os
from typing import Any, Dict, Optional

import boto3

# System Prompt: Enforcing strict grounding and executive tone
SYSTEM_PROMPT = (
  "You are a professional trend analyst writing for beauty and fashion businesses. "
  "Use ONLY the forecast facts provided in the prompt payload. "
  "Never change verdicts or invent numerical statistics. "
  "If an item is marked with low confidence, explicitly state that it carries uncertainty. "
  "Be concise, objective, and commercially actionable."
)

"""
BedrockWriter: Client wrapper for Amazon Bedrock runtime generation 
"""
class BedrockWriter: 
  # init function: to initialise the Bedrock runtime client
  # arguments: 
    # model_id: specific Amazon Bedrock model ID
    # region : AWS region name
  def __init__(self, model_id: Optional[str] = None, region: Optional[str] = None):
    self.model_id = model_id or os.environ.get("BEDROCK_MODEL_ID")
    if not self.model_id:
      raise ValueError("A model ID must be provided or set in the BEDROCK_MODEL_ID environment variable")

    self.client = boto3.client(
      "bedrock-runtime",
      region_name = region or os.getenv("AWS_REGION", "us-east-1")
    )
  
  # write_brief function : generates an executive B2B trend brief. One with structured forecast facts.
  # arguments: 
    # forecast: Dictionary that has decision metrics, confidence scores, nd trend indicators.
    # audience: Target stakeholder group for tone tuning (default: merchandising team)
  def write_brief(
    self,
    forecast: Dict[str, Any],
    audience: str = "merchandising team"
  ) -> str:
  # Constructing a structured prompt feeding JSON facts to the model
    prompt = (
      f"Target Audience: {audience}\n\n"
      f"Forecast Facts & Metrics (JSON):\n"
      f"{json.dumps(forecast, indent=2)}\n\n"
      "Write a concise B2B brief containing: "
      "1. Headline, 2. Market Outlook, 3. Three recommended actions, 4. Key risks"
    )

  # Invoking Bedrock using the unified converse api
    response = self.client.converse(
      modelId = self.model_id,
      system = [{"text": SYSTEM_PROMPT}],
      messages = [
        {
          "role": "user",
          "content": [{"text": prompt}]
        }
      ],
      inferenceConfig = {
        "maxTokens": 800,
        "temperature": 0.3 # low temp to ensure adherence to provided data facts
      },
    )

    # Extracting and returning the generated text
    return response["output"]["message"]["content"][0]["text"]
  