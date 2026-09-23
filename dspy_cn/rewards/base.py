"""
Author: Preethi Gajawada
Description: Base dataclass definitions used across reward modules.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardInput:
    hate_speech: str
    counter_narrative: str
    ground_truth: Optional[str] = None
    knowledge: Optional[str] = None
