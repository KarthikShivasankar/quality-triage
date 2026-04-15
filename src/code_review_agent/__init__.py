"""
Code Review Agent — AI-powered code review using:
- ml_smells_detector: ML framework anti-pattern detection
- python_smells_detector: General Python code quality analysis
- text_classification (tdsuite): Technical debt classification
"""

from code_review_agent.agent import CodeReviewAgent

__all__ = ["CodeReviewAgent"]
__version__ = "0.1.0"
