"""
3D Visualization Components for TruthLens
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Professional 3D visualizations using Plotly with animated cameras and effects.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.animation_config import (
    get_adjusted_duration,
    get_camera_frames,
    get_stagger_sequence,
    ANIMATION_CONFIG,
    ANIMATION_COLORS,
)


def create_3d_truth_score_sphere(truth_score: float, verdict: str, speed_multiplier: float = 1.0) -> go.Figure:
    """
    Create a rotating 3D sphere representing the TruthScore with animated camera.

    Parameters
    ----------
    truth_score : float (0-100)
    verdict : str
    speed_multiplier : float
        Animation speed multiplier (1.0 = normal)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Determine color based on score
    if truth_score <= 30:
        colors = ["#FF4D6D", "#FF738D"]  # Red gradient
        primary_color = "#FF4D6D"
    elif truth_score <= 55:
        colors = ["#FFB800", "#FFD700"]  # Orange/Yellow gradient
        primary_color = "#FFB800"
    else:
        colors = ["#00FF99", "#56FFAE"]  # Green gradient
        primary_color = "#00FF99"

    # Create sphere surface
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta_mesh, phi_mesh = np.meshgrid(theta, phi)

    x = np.sin(phi_mesh) * np.cos(theta_mesh)
    y = np.sin(phi_mesh) * np.sin(theta_mesh)
    z = np.cos(phi_mesh)

    # Score-based intensity map
    intensity = np.ones_like(x) * (truth_score / 100)

    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=z,
        surfacecolor=intensity,
        colorscale=[[0, colors[0]], [1, colors[1]]],
        showscale=False,
        hovertemplate="<b>TruthScore:</b> %{surfacecolor:.0%}<extra></extra>",
        name="",
        lighting=dict(
            ambient=0.6,
            diffuse=0.8,
            roughness=0.3,
            specular=0.8
        ),
        lightposition=dict(x=100, y=100, z=100),
    )])

    # Add central value indicator
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='text',
        text=[f'<b>{truth_score:.0f}</b>'],
        textfont=dict(size=20, color=primary_color),
        hovertemplate=f"<b>TruthScore™:</b> {truth_score:.1f}/100<extra></extra>",
        showlegend=False,
        name=""
    ))

    # Create orbital camera animation frames
    num_frames = get_camera_frames(ANIMATION_CONFIG["CAMERA_ROTATION_DURATION"], speed_multiplier)
    frames = []
    for i in range(num_frames):
        angle = (2 * np.pi * i) / num_frames
        camera_x = 1.2 * np.cos(angle)
        camera_y = 1.2 * np.sin(angle)
        camera_z = 1.0
        frames.append(go.Frame(
            layout=go.Layout(
                scene=dict(
                    camera=dict(
                        eye=dict(x=camera_x, y=camera_y, z=camera_z),
                        center=dict(x=0, y=0, z=0)
                    )
                )
            ),
            name=str(i)
        ))

    duration_ms = int(get_adjusted_duration(ANIMATION_CONFIG["CAMERA_ROTATION_DURATION"], speed_multiplier) * 1000)

    fig.frames = frames
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-1.2, 1.2]),
            yaxis=dict(visible=False, range=[-1.2, 1.2]),
            zaxis=dict(visible=False, range=[-1.2, 1.2]),
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.0),
                center=dict(x=0, y=0, z=0)
            ),
            bgcolor="rgba(0,0,0,0)",
            aspectmode='cube'
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                label="▶",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=int(duration_ms / num_frames), redraw=False),
                    fromcurrent=True,
                    mode='immediate',
                    transition=dict(duration=0)
                )]
            )],
            x=0.0,
            y=0.0,
            visible=False
        )],
        sliders=[dict(
            active=0,
            visible=False,
            steps=[dict(args=[[str(i)], dict()], label=str(i), method='animate') for i in range(num_frames)]
        )]
    )

    return fig


def create_3d_algorithm_comparison(model_predictions: dict, speed_multiplier: float = 1.0) -> go.Figure:
    """
    Create 3D scatter plot simulating algorithm comparison bars with staggered entry.

    Parameters
    ----------
    model_predictions : dict
    speed_multiplier : float
        Animation speed multiplier

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not model_predictions:
        return go.Figure()

    models = []
    confidences = []
    verdicts = []

    for key, pred in model_predictions.items():
        model_name = key.replace("_", " ").title()
        models.append(model_name)
        conf = pred.get("confidence", 0.5) * 100
        confidences.append(conf)
        verdicts.append(pred.get("verdict", "UNCERTAIN"))

    # Create 3D scatter points (simulating bars)
    colors = []
    for verdict in verdicts:
        if verdict == "FAKE":
            colors.append(ANIMATION_COLORS["fake"])
        elif verdict == "REAL":
            colors.append(ANIMATION_COLORS["real"])
        else:
            colors.append(ANIMATION_COLORS["uncertain"])

    # Create staggered entry frames
    n = len(models)
    stagger_delays = get_stagger_sequence(n, speed_multiplier)
    num_frames = int(get_adjusted_duration(ANIMATION_CONFIG["CAMERA_ROTATION_DURATION"], speed_multiplier) * 30)
    frames = []

    # Create frames with staggered entry animation
    for frame_idx in range(num_frames):
        frame_data = []
        for i, (model, conf, color) in enumerate(zip(models, confidences, colors)):
            # Calculate entry progress for each bar
            entry_time = stagger_delays[i]
            entry_frame = int(entry_time * 30)

            if frame_idx >= entry_frame:
                # Animate Z position from 0 to confidence value
                progress = min((frame_idx - entry_frame) / 20, 1.0)
                z_val = conf * progress
            else:
                z_val = 0

            frame_data.append(go.Scatter3d(
                x=[i], y=[conf], z=[z_val],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=color,
                    opacity=0.8,
                    line=dict(color="rgba(255,255,255,0.2)", width=1)
                ),
                text=[f"{conf:.0f}%<br>{verdicts[i]}"],
                textposition="top center",
                textfont=dict(color="#F0F4FF", size=9),
                name=model,
                hovertemplate=f"<b>{model}</b><br>Confidence: {conf:.0f}%<extra></extra>",
                showlegend=(frame_idx == 0)
            ))

        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))

    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Model", tickfont=dict(color="#A0A5B5"), backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Confidence (%)", tickfont=dict(color="#A0A5B5"), backgroundcolor="rgba(0,0,0,0)", range=[0, 110]),
            zaxis=dict(title="Animation", tickfont=dict(color="#A0A5B5"), backgroundcolor="rgba(0,0,0,0)", range=[0, 110]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
            ),
            bgcolor="rgba(12,12,14,0.6)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        hovermode='closest'
    )

    return fig


def create_3d_credibility_breakdown(breakdown: dict, speed_multiplier: float = 1.0) -> go.Figure:
    """
    Create a 3D scatter plot with animated signal points in 3D space.

    Parameters
    ----------
    breakdown : dict mapping signal_name → score (0-100)
    speed_multiplier : float
        Animation speed multiplier

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not breakdown:
        return go.Figure()

    # Map signals to 3D coordinates
    signals = list(breakdown.keys())
    scores = list(breakdown.values())

    # Invert clickbait probability for better visualization
    for i, (sig, score) in enumerate(breakdown.items()):
        if "Clickbait" in sig:
            scores[i] = 100 - score

    # Create pseudo-3D representation
    n = len(signals)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    x = np.cos(angles) * (scores[0] / 100)
    y = np.sin(angles) * (scores[1] / 100)
    z = np.ones(n) * (np.mean(scores) / 100)

    # Color based on score
    colors_map = []
    for score in scores:
        if score < 35:
            colors_map.append(ANIMATION_COLORS["fake"])
        elif score < 60:
            colors_map.append(ANIMATION_COLORS["uncertain"])
        else:
            colors_map.append(ANIMATION_COLORS["real"])

    # Create animation frames for points scaling in from center
    stagger_delays = get_stagger_sequence(n, speed_multiplier)
    num_frames = int(get_adjusted_duration(ANIMATION_CONFIG["CREDIBILITY_SIGNAL_ENTRY"], speed_multiplier) * 30)
    frames = []

    for frame_idx in range(num_frames):
        frame_data = []

        # Add animated points
        for i in range(n):
            entry_frame = int(stagger_delays[i] * 30)

            if frame_idx >= entry_frame:
                # Animate scale from 0 to 1 and opacity
                progress = min((frame_idx - entry_frame) / 15, 1.0)
                point_x = x[i] * progress
                point_y = y[i] * progress
                point_z = z[i] * progress
                point_size = 12 * progress
                point_opacity = 0.3 + (0.6 * progress)
            else:
                point_x, point_y, point_z, point_size, point_opacity = 0, 0, 0, 2, 0

            frame_data.append(go.Scatter3d(
                x=[point_x], y=[point_y], z=[point_z],
                mode='markers+text',
                marker=dict(
                    size=point_size,
                    color=colors_map[i],
                    opacity=point_opacity,
                    line=dict(color="rgba(255,255,255,0.3)", width=2),
                    symbol='diamond'
                ),
                text=[f"{scores[i]:.0f}"] if progress > 0.5 else [""],
                textposition="top center",
                textfont=dict(color="#F0F4FF", size=11),
                name=signals[i],
                hovertemplate=f"<b>{signals[i]}</b>: {scores[i]:.0f}<extra></extra>",
                showlegend=(frame_idx == num_frames - 1)
            ))

        # Add connections (always present)
        for i in range(n):
            next_i = (i + 1) % n
            frame_data.append(go.Scatter3d(
                x=[x[i], x[next_i]],
                y=[y[i], y[next_i]],
                z=[z[i], z[next_i]],
                mode='lines',
                line=dict(color="rgba(0,212,255,0.2)", width=2),
                showlegend=False,
                hoverinfo='skip',
                name=""
            ))

        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))

    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=1.3, y=1.3, z=1.2),
                center=dict(x=0, y=0, z=0)
            ),
            bgcolor="rgba(0,0,0,0)",
            aspectmode='cube'
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )

    return fig


def create_3d_cluster_explorer(cluster_data: dict, new_point: dict, speed_multiplier: float = 1.0) -> go.Figure:
    """
    Create enhanced 3D scatter plot of article clusters with animated camera pan.

    Parameters
    ----------
    cluster_data : dict
    new_point : dict
    speed_multiplier : float
        Animation speed multiplier

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not cluster_data or not cluster_data.get("x"):
        return go.Figure()

    x = cluster_data["x"]
    y = cluster_data["y"]
    labels = ["Fake" if l == 0 else "Real" for l in cluster_data["true_label"]]

    # Add pseudo-z dimension based on cluster
    z = np.array([cluster_data["cluster"][i] * 10 for i in range(len(x))])

    color_map = {"Fake": ANIMATION_COLORS["fake"], "Real": ANIMATION_COLORS["real"]}
    colors = [color_map.get(label, ANIMATION_COLORS["uncertain"]) for label in labels]

    # Create animation frames with camera pan/zoom
    cluster_pan_frames = int(get_adjusted_duration(ANIMATION_CONFIG["CLUSTER_PAN_DURATION"], speed_multiplier) * 30)
    frames = []

    for frame_idx in range(cluster_pan_frames):
        progress = frame_idx / max(cluster_pan_frames - 1, 1)

        # Animate camera from wide zoom to closer view
        eye_z = 1.0 + (1.5 * (1 - progress))
        eye_x = 1.2 + (0.3 * (1 - progress))
        eye_y = 1.2 + (0.3 * (1 - progress))

        frame_data = [
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=4,
                    color=colors,
                    opacity=0.5,
                    line=dict(color="rgba(255,255,255,0.1)", width=0.5)
                ),
                text=labels,
                name="Historical Articles",
                hovertemplate="<b>%{text}</b><extra></extra>"
            ),
            go.Scatter3d(
                x=[new_point.get("x", 0)],
                y=[new_point.get("y", 0)],
                z=[50],
                mode='markers+text',
                marker=dict(
                    size=16,
                    color=ANIMATION_COLORS["uncertain"],
                    symbol="diamond",
                    line=dict(color="white", width=2),
                    opacity=0.95
                ),
                text=["Your Article"],
                textposition="top center",
                name="Your Article",
                hovertemplate="<b>Your Article</b><extra></extra>"
            )
        ]

        frames.append(go.Frame(
            data=frame_data,
            layout=go.Layout(
                scene=dict(
                    camera=dict(
                        eye=dict(x=eye_x, y=eye_y, z=eye_z),
                    )
                )
            ),
            name=str(frame_idx)
        ))

    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="PC1", gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#A0A5B5")),
            yaxis=dict(title="PC2", gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#A0A5B5")),
            zaxis=dict(title="Cluster", gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#A0A5B5")),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=2.5),
            ),
            bgcolor="rgba(12,12,14,0.4)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='closest',
        showlegend=True,
        legend=dict(font=dict(color="#A0A5B5"), bgcolor="rgba(0,0,0,0.3)")
    )

    return fig
