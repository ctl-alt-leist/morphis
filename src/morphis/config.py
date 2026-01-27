"""
Morphis Configuration

Global settings for the morphis geometric algebra library.
"""

# Numerical tolerance for comparisons and safe division
# Used throughout the library for:
# - Zero detection in normalization
# - Floating point comparisons
# - Safe division to avoid divide-by-zero
TOLERANCE: float = 1e-12
