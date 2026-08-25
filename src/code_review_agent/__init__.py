"""
Code Review Agent — AI-powered code review using LiteLLM plus:
- ml_smells_detector: ML framework anti-pattern detection
- python_smells_detector: General Python code quality analysis
- text_classification (tdsuite): Technical debt classification
"""

from code_review_agent.agent import CodeReviewAgent, LiteLLMAgent

__all__ = ["CodeReviewAgent", "LiteLLMAgent"]
__version__ = "0.2.0"
