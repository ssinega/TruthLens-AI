"""
Animation Configuration Module for TruthLens 3D Effects
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Central configuration for all animation timings, speeds, and effects.
This allows animations to be easily adjusted globally or per-user preference.
"""

# Global animation durations (in seconds)
ANIMATION_CONFIG = {
    # Core 3D chart animations
    "CAMERA_ROTATION_DURATION": 6.0,        # Sphere orbit rotation time
    "STAGGER_DELAY": 0.1,                   # Delay between staggered elements
    "CREDIBILITY_SIGNAL_ENTRY": 0.6,        # Time for each signal to appear
    "CLUSTER_PAN_DURATION": 3.0,            # Camera pan on cluster load

    # CSS background animations
    "PARTICLE_FLOAT_DURATION": 15.0,        # Floating particles cycle
    "GRID_FLOW_DURATION": 8.0,              # Grid lines flowing effect
    "SHIMMER_DURATION": 3.0,                # Shimmer effect on badges

    # Default multipliers
    "DEFAULT_SPEED_MULTIPLIER": 1.0,        # 1x = normal speed
    "REDUCED_MOTION_DURATION": 0.1,         # For accessibility (instant)

    # Easing and style
    "EASING_FUNCTION": "cubic-bezier(0.4, 0, 0.2, 1)",  # Smooth easing
    "ANIMATION_SMOOTHING": True,            # Enable smooth transitions
}


def get_adjusted_duration(base_duration: float, speed_multiplier: float = 1.0, reduce_motion: bool = False) -> float:
    """
    Calculate adjusted animation duration based on user preferences.

    Parameters
    ----------
    base_duration : float
        Base animation duration in seconds
    speed_multiplier : float
        User speed preference (0.5 = half speed, 2.0 = double speed)
    reduce_motion : bool
        If True, return minimal duration for accessibility

    Returns
    -------
    float
        Adjusted duration in seconds
    """
    if reduce_motion:
        return ANIMATION_CONFIG["REDUCED_MOTION_DURATION"]
    return max(0.1, base_duration * speed_multiplier)


def get_stagger_sequence(count: int, speed_multiplier: float = 1.0, reduce_motion: bool = False) -> list:
    """
    Generate staggered animation delays for sequential element entry.

    Parameters
    ----------
    count : int
        Number of elements to stagger
    speed_multiplier : float
        User speed preference
    reduce_motion : bool
        If True, all items appear at once

    Returns
    -------
    list
        List of delay values in seconds for each element
    """
    if reduce_motion:
        return [0.0] * count

    base_delay = ANIMATION_CONFIG["STAGGER_DELAY"] / speed_multiplier
    return [base_delay * i for i in range(count)]


def get_camera_frames(base_duration: float, speed_multiplier: float = 1.0) -> int:
    """
    Calculate number of animation frames for camera rotation.

    Parameters
    ----------
    base_duration : float
        Base duration in seconds
    speed_multiplier : float
        User speed preference

    Returns
    -------
    int
        Number of frames (approx 30fps)
    """
    adjusted_duration = get_adjusted_duration(base_duration, speed_multiplier)
    return max(20, int(adjusted_duration * 30))  # ~30fps


# Animation presets for different use cases
ANIMATION_PRESETS = {
    "slow": {"multiplier": 0.5, "label": "🐢 Slow (0.5x)"},
    "normal": {"multiplier": 1.0, "label": "⚡ Normal (1x)"},
    "fast": {"multiplier": 1.5, "label": "🚀 Fast (1.5x)"},
    "very_fast": {"multiplier": 3.0, "label": "⚡⚡ Very Fast (3x)"},
}


# Color scheme for 3D animations
ANIMATION_COLORS = {
    "fake": "#FF4D6D",
    "real": "#00FF99",
    "uncertain": "#FFB800",
    "cyan": "#00E5FF",
    "purple": "#8B5CF6",
}
