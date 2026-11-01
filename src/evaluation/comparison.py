"""
Visual comparison generators for optical, reference SAR, synthetic SAR, and residual difference maps.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Tuple
from .metrics import compute_difference_map

def create_comparison_figure(
    optical_rgb: np.ndarray,
    reference_sar: np.ndarray,
    synthetic_sar: np.ndarray,
    metrics: Dict[str, float]
) -> go.Figure:
    """
    Build an interactive 4-panel Plotly figure displaying:
    1. Optical Input
    2. Actual SAR Reference
    3. Model Synthetic SAR
    4. Absolute Residual Error Map
    """
    diff_map = compute_difference_map(reference_sar, synthetic_sar)
    
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=(
            "1. Optical Input (S2 RGB)",
            "2. Actual SAR Reference (S1)",
            f"3. Synthetic SAR (PSNR: {metrics.get('psnr', 0):.2f} dB)",
            f"4. Error Map (SSIM: {metrics.get('ssim', 0):.3f})"
        ),
        horizontal_spacing=0.03
    )
    
    # Panel 1: Optical RGB
    fig.add_trace(go.Image(z=optical_rgb), row=1, col=1)
    
    # Panel 2: Actual SAR
    ref_2d = reference_sar.squeeze() if reference_sar.ndim > 2 else reference_sar
    fig.add_trace(go.Heatmap(z=ref_2d, colorscale="Greys_r", showscale=False), row=1, col=2)
    
    # Panel 3: Synthetic SAR
    syn_2d = synthetic_sar.squeeze() if synthetic_sar.ndim > 2 else synthetic_sar
    fig.add_trace(go.Heatmap(z=syn_2d, colorscale="Greys_r", showscale=False), row=1, col=3)
    
    # Panel 4: Difference Map
    fig.add_trace(go.Heatmap(z=diff_map, colorscale="Hot", showscale=True, colorbar=dict(title="Error", thickness=10)), row=1, col=4)
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0")
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig
