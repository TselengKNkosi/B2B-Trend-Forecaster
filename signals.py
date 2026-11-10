"""
signals.py

Transforms raw time-series metrics from fashion and beauty trends into a normalized numerical state representation using decision models like Strands Decider 2B
"""

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional

@dataclass
class TrendSignal:
  # Encapsulates raw time-series data captured for a single trend
  name: str
  category: str # vertical classification ("beauty" or "fashion")
  search_index: List[float] # weekly search interest data
  social_mentions: List[float]
  sell_through: Optional[List[float]] = None

"""
calculate_growth function

function calculates non-linear growth by comparing the average value of the most recent window period against the average of the preceding window period 
  """
def calculate_growth(series: List[float], window: int = 4) -> float:
  required_length = 2 * window
  if not series or len(series) < required_length:
    return 0.0

  prior_period = series[-required_length: -window]
  recent_period = series[-window:]

  prior_mean = mean(prior_period)
  recent_mean = mean(recent_period)

  if prior_mean == 0.0:
    return 0.0

  return (recent_mean/prior_mean) - 1.0

"""
build_state function: 

constructs a numeric dictionary schema representing the trend's state. -> provides a deterministic input payload for Strands Decider 2B.
"""
def build_state(signal: TrendSignal) -> Dict[str, Any]:
  # Preventing empty search series'
  if not signal.search_index:
    peak_ratio = 0.0
  else:
    max_search = max(signal.search_index)
    peak_ratio = (signal.search_index[-1] / max_search) if max_search > 0 else 0.0

  # Calculating growth rates across metrics
  search_growth = round(calculate_growth(signal.search_index), 3)
  social_growth = round(calculate_growth(signal.social_mentions), 3)
  sales_growth = (
    round(calculate_growth(signal.sell_through), 3)
    if signal.sell_through is not None
    else None
  )

  return {
    "trend": signal.name,
    "category": signal.category,
    "search_growth": search_growth,
    "social_growth": social_growth,
    "sale_growth": sales_growth,
    "search_level": round(peak_ratio, 3), # scaled based on proximity to historic peak
    "weeks_observed": len(signal.search_index),
  }