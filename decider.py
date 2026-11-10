"""
decider.py

This file defines core decision types, the Decider protocol, and implementation adapters for both the real Strands Decider 2B model and the Heuristic stand-in
"""
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol, Tuple

class QuestionKind(str, Enum):
  # supported question types for Strands Decider evals
  YES_NO = "yes_no" # evaluates to boolean
  SCORE = "score" # evaluates to value between 0.0 and 1.0
  CHOICE = "choice" # slects an option from predefined options tuple

@dataclass(frozen = True)
class Question:
  # Represents a structured evaluation question
  id: str
  kind: QuestionKind
  text: str
  options: Optional[Tuple[str, ...]] = None
 
@dataclass(frozen = True)
class Decision: 
  # Immutable return object containing a Decider's decision
  value: Any
  confidence: float

class Decider(Protocol):
  """
  decide functon: 

  Evaluates a numeric state against a typed question.
  """
  def decide(self, state: str, question: Question) -> Decision:
    ...

DEFAULT_CHECKPOINT = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
SCORE_LEVELS = ["very low", "low", "moderate", "high", "very high"]

class StrandsDecider:
  # Adapter implementation for AWS STrands Decider 2B decision model
  # Evaluates numeric state JSON against typed questionsusing local pointer heads

  def __init__(self, checkpoint: Optional[str] = None, device: Optional[str] = None): # initialises and loads the Strands Decider model checkpoint
    self.checkpoint = checkpoint or DEFAULT_CHECKPOINT
    self.engine = self._load_engine()

  def _load_engine(self) -> Any:
    # loads underlying Strands Decider model instance
    try:
      from strands_decider.infer import load_engine
    except ImportError as exc:
      raise ImportError("strands-decider is not installed: pip install strands-decider") from exc
    return load_engine(self.checkpoint, device = self.device)

  @staticmethod
  def _to_strands(q: Question) -> Any:
    # Mapping question kinds onto the package's 3 primitives
    from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

    if q.kind is QuestionKind.YES_NO:
      return NoulQuestion(instructions = q.text)
    if q.kind is QuestionKind.SCORE:
      return ScoreQuestion(instructions = q.text, criteria = SCORE_LEVELS)
    return ChoiceQuestion(instructions = q.text, criteria = {o: "" for o in q.options or ()})

  def decide(self, state: str, question: Question) -> Decision:
    answer = self.engine.ask(state, {question.id: self._to_strands(question)}).answers[question.id]

    if answer.type == "noul":
      return Decision(answer.noul >= 0.5, round(abs(2 * answer.noul -1), 4))

    if answer.type == "score":
      value = max(0.0, min(1.0, answer.score / (len(SCORE_LEVELS) -1)))
      return Decision(round(value, 2), answer.confidence)

    return Decision(answer.choice, answer.confidence)

class HeuristicDecider:
  # Rule-based stand-in so the pipeline runs without the model
  def decide(self, state: str, q: Question) -> Decision:
    data = json.loads(state)

    #calculate composite velocity growth metric
    search_growth = data.get("search_growth", 0.0)
    social_growth = data.get("social_growth", 0.0)
    search_level = data.get("search_level", 0.0)

    avg_growth = (search_growth + social_growth) / 2.0
    confidence = round(min(0.95, 0.5 + abs(avg_growth)), 2)

    # q.id: Yes/No Mainstream Trajectory
    if q.id == "mainstream_90d":
      is_mainstream = avg_growth > 0.25
      return Decision(value = is_mainstream, confidence = confidence)

    # q.id: Score 0.0 - 1.0 Momentum Score
    if q.id == "momentum":
      momentum_score = round(max(0.0, min(1.0, 0.5 + (avg_growth / 2.0))), 2)
      return Decision(value = momentum_score, confidence = 0.65)

    # q.id: Choice-based lifecycle stage evaluation
    if q.id == "stage":
      if avg_growth > 0.4 and search_level < 0.5:
        stage = "emerging"
      elif avg_growth > 0.1:
        stage = "growing"
      elif search_level > 0.8:
        stage = "peak"
      else:
        stage = "declining"

      return Decision(value = stage, confidence = confidence)

    raise ValueError(f"Unknown question ID evaluated: {q.id}")

def get_decider() -> Decider: 
  decider_mode = os.getenv("DECIDER", "heuristic").lower() 

  if decider_mode == "strands":
    return StrandsDecider(os.getenv("DECIDER_MODEL_PATH") or None)

  return HeuristicDecider()
    