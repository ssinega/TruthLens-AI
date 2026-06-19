"""
3D Theme and Animation System for TruthLens
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provides immersive 3D visual effects, particle systems, and smooth animations
that enhance the user experience without affecting model functionality.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import random


class ParticleSystem:
    """Generate floating particle effects for background ambiance."""
    
    def __init__(self, particle_count: int = 50):
        self.particle_count = particle_count
        self.particles = self._generate_particles()
    
    def _generate_particles(self) -> List[Dict]:
        """Generate random floating particles with varied properties."""
        particles = []
        for i in range(self.particle_count):
            particle = {
                'id': i,
                'x': random.uniform(0, 100),
                'y': random.uniform(0, 100),
                'size': random.uniform(2, 8),
                'opacity': random.uniform(0.1, 0.4),
                'color': random.choice([
                    'rgba(123, 92, 240, {})',
                    'rgba(6, 182, 212, {})',
                    'rgba(192, 132, 252, {})'
                ]),
                'speed_x': random.uniform(-0.02, 0.02),
                'speed_y': random.uniform(-0.05, -0.01),
                'pulse_phase': random.uniform(0, 2 * np.pi)
            }
            particles.append(particle)
        return particles
    
    def get_css_animations(self) -> str:
        """Generate CSS animations for particles."""
        css = []
        for p in self.particles[:20]:  # Limit to 20 visible particles
            color = p['color'].format(p['opacity'])
            animation_duration = random.uniform(15, 30)
            delay = random.uniform(0, 10)
            
            css.append(f"""
            .particle-{p['id']} {{
                position: fixed;
                width: {p['size']}px;
                height: {p['size']}px;
                background: {color};
                border-radius: 50%;
                left: {p['x']}vw;
                top: {p['y']}vh;
                pointer-events: none;
                z-index: 0;
                animation: floatParticle{animation_duration:.1f}s ease-in-out infinite;
                animation-delay: {delay}s;
                box-shadow: 0 0 {p['size']*2}px {color};
            }}
            """)
        
        # Add keyframes for each unique duration
        durations = set()
        for p in self.particles[:20]:
            durations.add(random.uniform(15, 30))
        
        for duration in durations:
            css.append(f"""
            @keyframes floatParticle{duration:.1f}s {{
                0%, 100% {{
                    transform: translateY(0) translateX(0) scale(1);
                    opacity: 0.3;
                }}
                25% {{
                    transform: translateY(-20px) translateX(10px) scale(1.1);
                    opacity: 0.5;
                }}
                50% {{
                    transform: translateY(-40px) translateX(-5px) scale(0.9);
                    opacity: 0.7;
                }}
                75% {{
                    transform: translateY(-20px) translateX(15px) scale(1.05);
                    opacity: 0.4;
                }}
            }}
            """)
        
        return '\n'.join(css)


class Theme3D:
    """
    3D Theme system for TruthLens.
    Provides immersive visual effects without affecting model functionality.
    """
    
    # Dark theme color palette
    COLORS = {
        'primary': '#7B5CF0',
        'secondary': '#06B6D4',
        'accent': '#C084FC',
        'success': '#10B981',
        'warning': '#F59E0B',
        'danger': '#EF4444',
        'dark_bg': '#06060C',
        'card_bg': 'rgba(255,255,255,0.07)',
    }
    
    @staticmethod
    def get_aurora_background() -> str:
        """Generate CSS for animated aurora background effect."""
        return """
        <style>
        .aurora-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -5;
            overflow: hidden;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        }
        
        .aurora-blob {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.4;
            animation: auroraMove 20s ease-in-out infinite;
        }
        
        .aurora-blob-1 {
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(123,92,240,0.6) 0%, transparent 70%);
            top: -200px;
            left: -200px;
            animation-delay: 0s;
        }
        
        .aurora-blob-2 {
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(6,182,212,0.5) 0%, transparent 70%);
            bottom: -150px;
            right: -150px;
            animation-delay: -5s;
        }
        
        .aurora-blob-3 {
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(192,132,252,0.4) 0%, transparent 70%);
            top: 50%;
            right: 10%;
            animation-delay: -10s;
        }
        
        @keyframes auroraMove {
            0%, 100% {
                transform: translate(0, 0) scale(1);
                opacity: 0.4;
            }
            25% {
                transform: translate(30px, -30px) scale(1.1);
                opacity: 0.5;
            }
            50% {
                transform: translate(-20px, 20px) scale(0.95);
                opacity: 0.35;
            }
            75% {
                transform: translate(20px, 10px) scale(1.05);
                opacity: 0.45;
            }
        }
        </style>
        
        <div class="aurora-bg">
            <div class="aurora-blob aurora-blob-1"></div>
            <div class="aurora-blob aurora-blob-2"></div>
            <div class="aurora-blob aurora-blob-3"></div>
        </div>
        """
    
    @staticmethod
    def get_particle_overlay() -> str:
        """Generate floating particle overlay HTML/CSS."""
        particle_system = ParticleSystem(particle_count=30)
        css = particle_system.get_css_animations()
        
        html = f"""
        <style>
        {css}
        </style>
        <div class="particle-overlay">
        """
        
        for p in particle_system.particles[:20]:
            html += f'<div class="particle-{p["id"]}\"></div>\n'
        
        html += "</div>"
        return html
    
    @staticmethod
    def get_glowing_grid() -> str:
        """Generate CSS for animated glowing grid background."""
        return """
        <style>
        .glow-grid {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -3;
            pointer-events: none;
            opacity: 0.15;
            background-image: 
                linear-gradient(rgba(123, 92, 240, 0.3) 1px, transparent 1px),
                linear-gradient(90deg, rgba(123, 92, 240, 0.3) 1px, transparent 1px);
            background-size: 50px 50px;
            animation: gridPulse 8s ease-in-out infinite;
        }
        
        @keyframes gridPulse {
            0%, 100% { opacity: 0.1; }
            50% { opacity: 0.2; }
        }
        </style>
        <div class="glow-grid"></div>
        """


def create_3d_globe_visualization(location_data: dict = None) -> go.Figure:
    """
    Create an animated 3D globe for geographic visualization.
    
    Parameters
    ----------
    location_data : dict
        Dictionary with location information for visualization
        
    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Create sphere surface for globe
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta_mesh, phi_mesh = np.meshgrid(theta, phi)
    
    x = np.sin(phi_mesh) * np.cos(theta_mesh)
    y = np.sin(phi_mesh) * np.sin(theta_mesh)
    z = np.cos(phi_mesh)
    
    # Dark globe surface
    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, '#1a1a2e'], [1, '#16213e']],
        showscale=False,
        hoverinfo='skip',
        opacity=0.8
    )])
    
    # Add glowing grid lines
    for i in range(0, 360, 30):
        angle = np.radians(i)
        x_circle = np.cos(angle) * np.cos(theta)
        y_circle = np.sin(angle) * np.cos(theta)
        z_circle = np.sin(theta)
        
        fig.add_trace(go.Scatter3d(
            x=x_circle, y=y_circle, z=z_circle,
            mode='lines',
            line=dict(color='rgba(123, 92, 240, 0.3)', width=1),
            hoverinfo='skip',
            showlegend=False
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor="rgba(0,0,0,0)",
            aspectmode='cube'
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    return fig
