"""Evaluation package initialization."""
from .metrics import (
    calculate_mse,
    calculate_mae,
    calculate_psnr,
    calculate_ssim,
    evaluate_image_pair,
    compute_difference_map
)
from .comparison import create_comparison_figure
from .evaluator import ModelEvaluator
