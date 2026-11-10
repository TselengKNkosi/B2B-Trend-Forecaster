"""
forecast.py: 

  FILE FOCUS: Holds orchestration pipeline connecting signal extraction, the Strands Decider, confidence gating, + Amazon Bedrock breif generation
"""

import json
import os
from typing import Any, Dict, Optional

# Imports from other .py files
from decider import Decider, HeuristicDecider, QuestionKind, Question
from signals import TrendSignal, build_state

# Setting minimun calibrated confidence threshold (to pass automated gating)
MIN_CONFIDENCE = 0.65

# Set of standardised questions evaluated for every trend 
QUESTIONS = [
  Question(
    id = "mainstream_90d",
    kind = QuestionKind.YES_NO,
    text = "Will this trend go mainstream within 90 days?"
  ),
  Question(
    id = "momentum",
    kind = QuestionKind.SCORE,
    text = "Rate current momentum from 0 to 1."
  ),
  Question(
    id = "stage",
    kind = QuestionKind.CHOICE,
    text = "Which lifecycle stage is this trend in?",
    options = ("emerging", "growing", "peak", "declining")
  ),
]

# forcast function:
# Executes the full forecasting pipeline for a given trend signal
# arguments:
  # sig: populated TrendSignal instance with time series metrics
  # decider: an instance implementing the Decider protocol
  # writer: Optional BedrockWriter instance for generating text B2B briefs
def forecast(
  sig: TrendSignal,
  decider: Decider,
  writer: Optional[Any] = None
) -> Dict[str, Any]:
  # Serialising trend signal into numeric state paylload
  state_payload = json.dumps(build_state(sig))
  decisions_results: Dict[str, Dict[str, Any]] = {}

  # Evaluating all questions through the decider
  for q in QUESTIONS:
    decision = decider.decide(state_payload, q)
    is_low_confidence = decision.confidence < MIN_CONFIDENCE

    decisions_results[q.id] = {
      "value": decision.value,
      "confidence": decision.confidence,
      "low_confidence": is_low_confidence, # to trigger confidence gate flag
  }

  # Compiling core forecast structure and checking for review requirements
  needs_review = any(d["low_confidence"] for d in decisions_results.values())

  forecast_output = {
    "trend": sig.name,
    "category": sig.category,
    "decisions": decisions_results,
    "needs_human_review": needs_review,
  }

  # Continally syntesizing narrative brief via Amazon Bedrock (if writer is available)
  if writer is not None:
    try:
      forecast_output["brief"] = writer.write_brief(forecast_output)
    except Exception:
      forecast_output["brief"] = None
      forecast_output["brief_error"] = "Brief generation unavailable"

  return forecast_output

if __name__ == "__main__":
  # defining a demo trend signal (end-to-end pipeline verification)
  from dotenv import load_dotenv
  load_dotenv()
  
  demo_signal = TrendSignal(
    name = "glass skin",
    category = "beauty",
    search_index = [20, 22, 21, 25, 30, 38, 47, 60],
    social_mentions = [100, 110, 120, 150, 190, 260, 340, 450],
  )

  # initialising Bedrock writer conditionally
  bedrock_writer = None
  if os.getenv("BEDROCK_MODEL_ID"):
    from bedrockclient import BedrockWriter
    bedrock_writer = BedrockWriter()

  # running execution using the HeuristicDecider stand-in + printing JSON result
  result = forecast(
    demo_signal, 
    HeuristicDecider(),
    writer = bedrock_writer
    )
  print(json.dumps(result, indent = 2))