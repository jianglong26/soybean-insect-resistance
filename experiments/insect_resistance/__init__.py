"""
Insect Resistance Analysis Module
抗虫性分析模块

This module provides comprehensive tools for analyzing insect resistance 
in soybean breeding experiments using multi-modal features (DINOv3 + Vegetation Indices).
"""

from .core.analyzer import MultiModalInsectResistanceAnalyzer

__version__ = "1.0.0"
__all__ = [
    'MultiModalInsectResistanceAnalyzer',
]
