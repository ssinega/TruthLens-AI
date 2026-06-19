"""
╔══════════════════════════════════════════════════════════════╗
║  TruthLens — AI Fake News Detector for Students             ║
║  "See Through the Noise"                                     ║
║  Author: Sinega Selvakumar                                   ║
╚══════════════════════════════════════════════════════════════╝

Main Streamlit Application
--------------------------
Launches the full TruthLens dashboard with:
    - Multi-tab article input (text / URL / image)
    - TruthScore™ gauge + verdict badge
    - Algorithm comparison chart
    - Writing style radar chart
    - LIME word-level explainability
    - Credibility component breakdown
    - Article Cluster Explorer (K-Means)
    - Student Learning Module
    - Session history + gamification
    - PDF report export

Run:
    streamlit run app.py
"""

# ── Standard library imports ───────────────────────────────────────────────────
import base64
import io
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ── Load .env for local development ───────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on os.environ

# ── Third-party imports ────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat

# ── Claude AI (Anthropic) ──────────────────────────────────────────────────────
try:
    import anthropic as _anthropic_lib
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    _anthropic_lib = None

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Project root on sys.path ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
PROFILE_PATH = ROOT / "data" / "user_profile.json"
sys.path.insert(0, str(ROOT))

# ── 3D Visualization Components ────────────────────────────────────────────────
try:
    from utils.viz_3d import (
        create_3d_truth_score_sphere,
        create_3d_algorithm_comparison,
        create_3d_credibility_breakdown,
        create_3d_cluster_explorer
    )
    HAS_3D = True
except ImportError:
    HAS_3D = False

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("truthlens.app")

# ── Page configuration (MUST be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="TruthLens — AI Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

def clean_html(html_str: str) -> str:
    """
    Remove leading indentation from each line of HTML to prevent markdown parser code-block wrapper issues.
    """
    import textwrap
    return textwrap.dedent(html_str).strip()

def inject_enhanced_3d_effects() -> None:
    """Inject enhanced 3D animations and visual effects without affecting model functionality."""
    
    # Import theme system
    from utils.theme_3d import Theme3D
    
    # Get all visual effects
    aurora_html = Theme3D.get_aurora_background()
    particle_html = Theme3D.get_particle_overlay()
    grid_html = Theme3D.get_glowing_grid()
    
    # Combine all effects
    st.markdown(aurora_html + particle_html + grid_html, unsafe_allow_html=True)


def inject_css() -> None:
    """Inject LUEIO-inspired dark agency CSS with enhanced 3D effects — all inline, no external file needed."""
    
    # Inject enhanced 3D effects first
    inject_enhanced_3d_effects()
    
    st.markdown(clean_html("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Playfair+Display:ital,wght@0,700;0,800;0,900;1,700&family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --v1: #7B5CF0;
        --v2: #9B30FF;
        --v3: #C084FC;
        --cyan: #06B6D4;
        --green: #10B981;
        --red: #EF4444;
        --amber: #F59E0B;
        --bg0: #06060C;
        --bg1: #0C0C18;
        --bg2: #12121F;
        --bg3: #1A1A2E;
        --border: rgba(123,92,240,0.15);
        --border-hi: rgba(123,92,240,0.40);
        --text: #F1F0FF;
        --muted: #8B8AA8;
        --ease: cubic-bezier(0.4,0,0.2,1);
    }

    /* ── Base / Global Typography ────────────────────────────────────── */
    html {
        font-size: 16px !important;
    }
    html, body {
        font-family: 'Inter', -apple-system, sans-serif !important;
        -webkit-font-smoothing: antialiased !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }

    /* ── Hide Streamlit sidebar collapse button label (merged text glitch) ─ */
    button[kind="header"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    .st-emotion-cache-1rtdyuf,
    [data-testid="stSidebarCollapseButton"] span,
    [aria-label="Collapse sidebar"] span,
    [aria-label="Expand sidebar"] span {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        overflow: hidden !important;
    }
    /* Hide the _arrow overlay text that bleeds from Streamlit's sidebar toggle */
    [data-testid="stSidebar"] ~ div[style*="arrow"],
    div[data-testid="stSidebarUserContent"] + div,
    .st-emotion-cache-h4xjwg { display: none !important; }
    .stApp {
        background: var(--bg0) !important;
        color: var(--text) !important;
        min-height: 100vh;
    }
    .block-container {
        padding: 0 2.5rem 3rem 2.5rem !important;
        max-width: 1380px !important;
    }

    /* ── Animated mesh background ── */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse 80% 50% at 20% 10%, rgba(123,92,240,0.12) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 80%, rgba(155,48,255,0.09) 0%, transparent 55%),
            radial-gradient(ellipse 50% 60% at 60% 30%, rgba(6,182,212,0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: -3;
        animation: meshPulse 12s ease-in-out infinite;
    }
    @keyframes meshPulse {
        0%,100% { opacity: 0.7; }
        50%      { opacity: 1.0; }
    }

    /* ── Dot grid overlay ── */
    .stApp::after {
        content: '';
        position: fixed;
        inset: 0;
        background-image: radial-gradient(circle, rgba(123,92,240,0.06) 1px, transparent 1px);
        background-size: 32px 32px;
        pointer-events: none;
        z-index: -2;
        animation: dotFade 10s ease-in-out infinite;
    }
    @keyframes dotFade {
        0%,100% { opacity: 0.4; }
        50%      { opacity: 0.9; }
    }

    /* ── Floating particles ── */
    .particle-3d {
        position: fixed;
        background: radial-gradient(circle, rgba(123,92,240,0.8) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        z-index: -1;
        animation-name: rise;
        animation-timing-function: linear;
        animation-iteration-count: infinite;
    }
    @keyframes rise {
        0%   { transform: translateY(100vh) scale(0.3); opacity: 0; }
        8%   { opacity: 0.7; }
        92%  { opacity: 0.3; }
        100% { transform: translateY(-5vh) scale(1.6); opacity: 0; }
    }

    /* ── Grid flow lines ── */
    .bg-grid-3d { position: fixed; inset: 0; pointer-events: none; z-index: -1; overflow: hidden; }
    .grid-line {
        position: absolute; top: 0; width: 1px; height: 100%;
        background: linear-gradient(180deg,
            transparent 0%,
            rgba(123,92,240,0.06) 35%,
            rgba(192,132,252,0.05) 65%,
            transparent 100%);
        animation-name: gflow;
        animation-timing-function: linear;
        animation-iteration-count: infinite;
    }
    @keyframes gflow {
        0%   { transform: translateY(-100%); opacity: 0; }
        15%  { opacity: 1; }
        85%  { opacity: 1; }
        100% { transform: translateY(100vh); opacity: 0; }
    }

    /* ── Glow orbs ── */
    .aurora-orb {
        position: fixed;
        border-radius: 50%;
        pointer-events: none;
        z-index: -4;
        filter: blur(100px);
        animation: orbFloat ease-in-out infinite;
    }
    @keyframes orbFloat {
        0%,100% { transform: translate(0,0); opacity: 0.10; }
        33%      { transform: translate(24px,-20px); opacity: 0.16; }
        66%      { transform: translate(-20px,24px); opacity: 0.08; }
    }

    /* ════════════════════════════════════
       HERO TYPOGRAPHY
    ════════════════════════════════════ */
    .lueio-hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(123,92,240,0.12);
        border: 1px solid rgba(123,92,240,0.30);
        border-radius: 999px;
        padding: 6px 18px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        color: #C084FC;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 20px;
    }
    .lueio-hero-eyebrow::before {
        content: '';
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #7B5CF0;
        box-shadow: 0 0 8px #7B5CF0;
        animation: dotBlink 1.8s ease-in-out infinite;
    }
    @keyframes dotBlink { 0%,100%{opacity:1;box-shadow:0 0 5px #7B5CF0;} 50%{opacity:0.4;box-shadow:0 0 16px #7B5CF0;} }

    .lueio-title {
        font-family: 'Playfair Display', Georgia, 'Times New Roman', serif !important;
        font-size: clamp(3.2rem, 7vw, 6rem) !important;
        font-weight: 800 !important;
        line-height: 1.0 !important;
        letter-spacing: 0.01em !important;
        color: #F1F0FF !important;
        margin: 0 0 16px 0 !important;
    }
    .lueio-title .grad {
        background: linear-gradient(135deg, #7B5CF0 0%, #C084FC 45%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradShift 5s ease-in-out infinite;
        background-size: 250% 250%;
    }
    @keyframes gradShift {
        0%,100% { background-position: 0% 50%; }
        50%      { background-position: 100% 50%; }
    }

    .lueio-subtitle {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.25rem !important;
        font-weight: 400 !important;
        color: var(--muted) !important;
        line-height: 1.75 !important;
        max-width: 620px;
        margin: 0 0 32px 0 !important;
    }

    /* ── Hero cursor ── */
    .hcursor {
        display: inline-block;
        width: 4px; height: 0.85em;
        background: #7B5CF0;
        margin-left: 4px;
        vertical-align: middle;
        border-radius: 2px;
        animation: blink 1s step-end infinite;
    }
    @keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0;} }

    /* ── Scanline ── */
    .hscan {
        height: 1px;
        max-width: 500px;
        margin: 0 auto 28px;
        background: linear-gradient(90deg, transparent, #7B5CF0, #C084FC, #06B6D4, transparent);
        animation: scanExp 2s ease-out both;
        box-shadow: 0 0 18px rgba(123,92,240,0.5);
    }
    @keyframes scanExp { from{width:0;opacity:0;} to{width:100%;opacity:1;} }

    /* ── Hero pills ── */
    .hpill {
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        padding: 8px 20px !important;
        border-radius: 999px !important;
        border: 1px solid rgba(123,92,240,0.25) !important;
        background: rgba(123,92,240,0.08) !important;
        color: #C084FC !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
        cursor: default !important;
        backdrop-filter: blur(8px) !important;
    }
    .hpill:hover {
        background: rgba(124,58,237,0.28) !important;
        border-color: rgba(124,58,237,0.65) !important;
        transform: translateY(-4px) !important;
        box-shadow: 0 10px 32px rgba(124,58,237,0.35) !important;
        color: #ffffff !important;
    }
    .fa1{animation:pf 3.0s ease-in-out infinite;}
    .fa2{animation:pf 3.4s ease-in-out infinite 0.4s;}
    .fa3{animation:pf 3.8s ease-in-out infinite 0.8s;}
    .fa4{animation:pf 3.2s ease-in-out infinite 1.2s;}
    .fa5{animation:pf 3.6s ease-in-out infinite 1.6s;}
    @keyframes pf { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-7px);} }

    /* ════════════════════════════════════
       SIDEBAR
    ════════════════════════════════════ */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"] {
        background: rgba(15,12,41,0.55) !important;
        backdrop-filter: blur(28px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
        border-right: 1px solid rgba(124,58,237,0.22) !important;
    }
    @media (min-width: 992px) {
        /* Customize expanded sidebar width */
        [data-testid="stSidebar"][aria-expanded="true"],
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 340px !important;
        }

        /* Customize collapsed sidebar width and offset to hide it fully */
        [data-testid="stSidebar"][aria-expanded="false"],
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 340px !important;
            margin-left: -340px !important;
        }

        /* Shift main content container, adjust width, and clear default left padding using sibling selector */
        [data-testid="stSidebar"][aria-expanded="true"] ~ [data-testid="stMain"] {
            margin-left: 340px !important;
            padding-left: 0px !important;
            padding-right: 0px !important;
            width: calc(100% - 340px) !important;
            max-width: calc(100% - 340px) !important;
        }

        /* Clear default padding/margin on stMainContent, prevent offset gap, and force left alignment */
        [data-testid="stSidebar"][aria-expanded="true"] ~ [data-testid="stMain"] [data-testid="stMainContent"] {
            padding-left: 0px !important;
            margin-left: 0px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: flex-start !important;
            justify-content: flex-start !important;
        }

        /* Shift header container and adjust its width when sidebar is expanded */
        [data-testid="stSidebar"][aria-expanded="true"] ~ [data-testid="stMain"] header[data-testid="stHeader"] {
            left: 340px !important;
            width: calc(100% - 340px) !important;
        }

        /* Align main content block-container to the left and allow full width usage next to sidebar */
        [data-testid="stSidebar"][aria-expanded="true"] ~ [data-testid="stMain"] .block-container,
        [data-testid="stSidebar"][aria-expanded="true"] ~ [data-testid="stMain"] .stMainBlockContainer {
            margin-left: 0px !important;
            margin-right: auto !important;
            padding-left: 0.5rem !important;
            padding-right: 2rem !important;
            max-width: 100% !important;
        }
    }
    section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.72) !important; }
    section[data-testid="stSidebar"] strong { color: #A78BFA !important; }
    section[data-testid="stSidebar"] hr {
        border: none !important; height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(124,58,237,0.45), rgba(6,182,212,0.25), transparent) !important;
        margin: 14px 0 !important;
    }

    /* ════════════════════════════════════
       LUEIO CARDS
    ════════════════════════════════════ */
    .lcard {
        background: rgba(255,255,255,0.07) !important;
        backdrop-filter: blur(24px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(160%) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 16px !important;
        padding: 28px !important;
        position: relative !important;
        overflow: hidden !important;
        transform-style: preserve-3d !important;
        will-change: transform !important;
        transition: border-color 0.4s cubic-bezier(0.4,0,0.2,1),
                    box-shadow 0.4s cubic-bezier(0.4,0,0.2,1),
                    transform 0.4s cubic-bezier(0.4,0,0.2,1) !important;
        z-index: 2 !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35),
                    0 0 0 1px rgba(124,58,237,0.10) !important;
    }
    .lcard::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 1px;
        background: linear-gradient(90deg, transparent, #7C3AED, #A78BFA, #06B6D4, transparent);
        animation: scanLine 4s ease-in-out infinite;
    }
    @keyframes scanLine { 0%{left:-100%;} 100%{left:100%;} }
    .lcard:hover {
        border-color: rgba(124,58,237,0.50) !important;
        box-shadow:
            0 0 0 1px rgba(124,58,237,0.45),
            0 20px 60px rgba(0,0,0,0.60),
            0 0 70px rgba(124,58,237,0.20),
            inset 0 1px 0 rgba(255,255,255,0.10) !important;
        transform: perspective(900px) rotateX(4deg) rotateY(-4deg) translateY(-8px) translateZ(10px) !important;
    }

    /* card icon circle */
    .lcard-icon {
        width: 52px; height: 52px;
        border-radius: 16px;
        background: rgba(124,58,237,0.18);
        border: 1px solid rgba(124,58,237,0.35);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 16px;
        transition: background 0.3s, transform 0.3s;
    }
    .lcard:hover .lcard-icon {
        background: rgba(124,58,237,0.38);
        transform: scale(1.08) rotate(-3deg);
    }
    .lcard-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.0rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }
    .lcard-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        color: rgba(255,255,255,0.65);
        line-height: 1.6;
    }

    /* ── legacy neo-card alias ── */
    .neo-card {
        background: rgba(255,255,255,0.07) !important;
        backdrop-filter: blur(24px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(160%) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.4s cubic-bezier(0.4,0,0.2,1) !important;
        z-index: 2 !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.30) !important;
    }
    .neo-card::before {
        content: '';
        position: absolute; top:0; left:-100%;
        width:100%; height:1px;
        background: linear-gradient(90deg, transparent, #7C3AED, #A78BFA, transparent);
        animation: scanLine 4s ease-in-out infinite;
    }
    .neo-card:hover {
        border-color: rgba(124,58,237,0.45) !important;
        box-shadow: 0 0 50px rgba(124,58,237,0.18), 0 20px 40px rgba(0,0,0,0.50) !important;
        transform: translateY(-4px) !important;
    }
    .glass-card {
        background: rgba(255,255,255,0.06) !important;
        backdrop-filter: blur(18px) saturate(140%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(140%) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.4s cubic-bezier(0.4,0,0.2,1) !important;
        z-index: 2 !important;
    }
    .glass-card:hover {
        border-color: rgba(124,58,237,0.40) !important;
        transform: translateY(-3px) !important;
    }

    /* ════════════════════════════════════
       BUTTONS
    ════════════════════════════════════ */
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 50%, #06B6D4 100%) !important;
        background-size: 200% 200% !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.70rem 1.8rem !important;
        transition: all 0.35s cubic-bezier(0.4,0,0.2,1) !important;
        box-shadow: 0 4px 24px rgba(124,58,237,0.45) !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 40px rgba(124,58,237,0.60) !important;
        background-position: right center !important;
    }
    .stButton > button:active { transform: translateY(-1px) !important; }

    /* ════════════════════════════════════
       TABS
    ════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.06) !important;
        backdrop-filter: blur(16px) !important;
        border-radius: 16px !important;
        padding: 5px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: rgba(255,255,255,0.60) !important;
        border-radius: 12px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 10px 22px !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(124,58,237,0.28) !important;
        color: #A78BFA !important;
        border: 1px solid rgba(124,58,237,0.50) !important;
        box-shadow: 0 0 20px rgba(124,58,237,0.25) !important;
    }

    /* ════════════════════════════════════
       INPUTS
    ════════════════════════════════════ */
    .stTextArea textarea, .stTextInput input {
        background: rgba(15,12,41,0.70) !important;
        border: 1px solid rgba(124,58,237,0.30) !important;
        color: #ffffff !important;
        border-radius: 14px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(12px) !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #7C3AED !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.25), 0 0 20px rgba(124,58,237,0.20) !important;
        outline: none !important;
    }

    /* ════════════════════════════════════
       METRICS
    ════════════════════════════════════ */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.07) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        transition: all 0.35s cubic-bezier(0.4,0,0.2,1) !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25) !important;
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(124,58,237,0.50) !important;
        box-shadow: 0 16px 40px rgba(124,58,237,0.18), 0 0 30px rgba(124,58,237,0.12) !important;
        transform: translateY(-3px) !important;
    }
    [data-testid="metric-container"] label {
        color: rgba(255,255,255,0.55) !important;
        font-size: 0.85rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #A78BFA !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 2.0rem !important;
        font-weight: 800 !important;
    }

    /* ════════════════════════════════════
       PROGRESS BAR
    ════════════════════════════════════ */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #7C3AED, #A78BFA, #06B6D4) !important;
        box-shadow: 0 0 14px rgba(124,58,237,0.60) !important;
    }
    .stProgress > div > div {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 4px !important;
    }

    /* ════════════════════════════════════
       EXPANDERS
    ════════════════════════════════════ */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.07) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
        color: #A78BFA !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .streamlit-expanderContent {
        background: rgba(15,12,41,0.55) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-top: none !important;
    }

    /* ════════════════════════════════════
       DIVIDERS / SCROLLBAR
    ════════════════════════════════════ */
    hr {
        border: none !important; height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(124,58,237,0.35), rgba(6,182,212,0.20), transparent) !important;
        margin: 18px 0 !important;
    }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0f0c29; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #7C3AED, #06B6D4);
        border-radius: 3px;
    }

    /* ════════════════════════════════════
       TYPOGRAPHY GLOBAL
    ════════════════════════════════════ */
    h1 {
        font-family: 'Playfair Display', Georgia, 'Times New Roman', serif !important;
        font-weight: 800 !important;
        color: #ffffff !important;
    }
    h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    p, li {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        color: rgba(255, 255, 255, 0.85) !important;
    }
    .stMarkdown div p {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
    }
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background: rgba(124,58,237,0.18) !important;
        color: #A78BFA !important;
        border-radius: 5px !important;
        padding: 2px 8px !important;
        font-size: 0.95rem !important;
    }

    /* ════════════════════════════════════
       VERDICT ANIMATIONS
    ════════════════════════════════════ */
    .verdict-badge-3d {
        display: inline-block; padding: 12px 32px; border-radius: 999px;
        font-family: 'Syne', sans-serif; font-weight: 800;
        font-size: 1rem; letter-spacing: 0.12em; text-transform: uppercase;
        transform: perspective(400px) translateZ(0);
        transition: transform 0.3s ease !important;
        box-shadow:
            0 0 0 1px currentColor,
            0 4px 24px rgba(0,0,0,0.45),
            inset 0 1px 0 rgba(255,255,255,0.12) !important;
    }
    .verdict-badge-3d:hover {
        transform: perspective(400px) translateZ(18px) scale(1.05) !important;
    }
    .verdict-fake-anim    { animation: shakeIn 0.5s ease-out, fakePulse 2s ease-in-out 0.5s infinite; }
    .verdict-suspicious-anim { animation: suspPulse 2.2s ease-in-out infinite; }
    .verdict-real-anim    { animation: realGrow 0.7s cubic-bezier(0.4,0,0.2,1) both, realPulse 2.5s ease-in-out 0.7s infinite; }
    @keyframes shakeIn  { 0%{transform:translateX(-8px) scale(0.9);} 25%{transform:translateX(8px);} 50%{transform:translateX(-5px);} 75%{transform:translateX(5px);} 100%{transform:translateX(0) scale(1);} }
    @keyframes fakePulse { 0%,100%{box-shadow:0 0 28px rgba(239,68,68,0.50);} 50%{box-shadow:0 0 70px rgba(239,68,68,0.95);} }
    @keyframes suspPulse { 0%,100%{box-shadow:0 0 24px rgba(245,158,11,0.50);} 50%{box-shadow:0 0 65px rgba(245,158,11,0.90);} }
    @keyframes realGrow  { 0%{transform:scale(0.6);opacity:0;} 100%{transform:scale(1);opacity:1;} }
    @keyframes realPulse { 0%,100%{box-shadow:0 0 24px rgba(16,185,129,0.50);} 50%{box-shadow:0 0 65px rgba(16,185,129,0.90);} }

    /* ════════════════════════════════════
       LOADING
    ════════════════════════════════════ */
    .analyzing-container { text-align:center; padding:56px 20px; }
    .loading-ring {
        width:72px; height:72px; border-radius:50%;
        border:4px solid transparent;
        border-top-color: #7C3AED;
        border-right-color: #A78BFA;
        border-bottom-color: #06B6D4;
        animation: spin 0.85s linear infinite;
        margin:0 auto 22px;
        box-shadow: 0 0 35px rgba(124,58,237,0.50), 0 0 60px rgba(6,182,212,0.20);
    }
    @keyframes spin { to{transform:rotate(360deg);} }
    .spinner-text  { font-family:'Space Grotesk',sans-serif; font-size:1.08rem; font-weight:700; color:#A78BFA; margin-bottom:6px; }
    .loading-stage { font-family:'Inter',sans-serif; font-size:0.80rem; color:rgba(255,255,255,0.65); margin-bottom:14px; }
    .loading-bar-track { width:300px; height:3px; background:rgba(255,255,255,0.08); border-radius:2px; overflow:hidden; margin:0 auto; }
    .loading-bar-fill  { height:100%; background:linear-gradient(90deg,#7C3AED,#A78BFA,#06B6D4,#A78BFA,#7C3AED); background-size:300% 100%; animation:barFill 2s ease-out both, barShimmer 1.8s linear infinite 0.5s; box-shadow:0 0 14px rgba(124,58,237,0.65); }
    @keyframes barFill    { from{width:0%;} to{width:100%;} }
    @keyframes barShimmer { 0%{background-position:200% 0;} 100%{background-position:-200% 0;} }

    .claude-card {
        background: rgba(109,40,217,0.12);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(124,58,237,0.28);
        border-radius: 14px;
        padding: 16px 20px;
        position: relative;
        overflow: hidden;
    }
    .claude-card::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, #7C3AED, #06B6D4, transparent);
        animation: scanLine 4.2s ease-in-out infinite;
    }
    .claude-brief-bullet {
        background: rgba(124,58,237,0.12);
        border-left: 3px solid #7C3AED;
        border-radius: 0 10px 10px 0;
        padding: 10px 14px;
        margin: 8px 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
        color: rgba(255,255,255,0.90);
        line-height: 1.58;
    }
    .ai-badge {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 3px 10px; border-radius: 999px;
        background: rgba(124,58,237,0.20);
        border: 1px solid rgba(124,58,237,0.40);
        color: #A78BFA; font-size: 0.72rem; font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.05em; margin-left: 8px;
    }

    /* ════════════════════════════════════════════════════
       TEXT HIGHLIGHTS (LIME XAI)
    ════════════════════════════════════════════════════ */
    .highlight-fake {
        background: rgba(239,68,68,0.22); color: #FCA5A5;
        padding: 1px 4px; border-radius: 3px;
        border-bottom: 2px solid rgba(239,68,68,0.55);
        font-weight: 500;
    }
    .highlight-real {
        background: rgba(16,185,129,0.18); color: #6EE7B7;
        padding: 1px 4px; border-radius: 3px;
        border-bottom: 2px solid rgba(16,185,129,0.45);
        font-weight: 500;
    }

    /* ════════════════════════════════════════════════════
       RED FLAG BULLETS
    ════════════════════════════════════════════════════ */
    .red-flag {
        background: rgba(239,68,68,0.10);
        border-left: 3px solid #EF4444;
        border-radius: 0 10px 10px 0;
        padding: 10px 14px; margin: 6px 0;
        font-size: 0.85rem; color: rgba(255,255,255,0.88);
        font-family: 'Inter', sans-serif;
    }

    /* ════════════════════════════════════════════════════
       MISC COMPONENTS
    ════════════════════════════════════════════════════ */
    .points-display {
        font-family: 'Syne', sans-serif;
        font-size: 2rem; font-weight: 700; text-align: center;
    }
    .history-item {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 10px; padding: 8px 10px; margin: 4px 0;
        transition: border-color 0.3s;
    }
    .history-item:hover { border-color: rgba(124,58,237,0.40); }
    .algo-card {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px; padding: 14px 16px; margin: 8px 0;
        transition: border-color 0.3s;
    }
    .algo-card:hover { border-color: rgba(124,58,237,0.40); }
    .algo-name { font-family:'Space Grotesk',sans-serif; font-weight:700; color:#06B6D4; font-size:0.92rem; }
    .algo-desc { color:rgba(255,255,255,0.65); font-size:0.82rem; margin-top:6px; line-height:1.5; }
    .tip-card {
        background: rgba(6,182,212,0.07);
        border: 1px solid rgba(6,182,212,0.18);
        border-left: 3px solid #06B6D4;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px; margin: 8px 0;
        font-size: 0.85rem; color: rgba(255,255,255,0.88);
    }
    .chart-3d { transition: transform 0.4s cubic-bezier(0.4,0,0.2,1); }
    .chart-3d:hover { transform: perspective(1000px) rotateX(1deg) scale(1.01); }
    .float-3d { animation: floatChart 4.2s ease-in-out infinite; }
    @keyframes floatChart { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-6px);} }

    /* ════════════════════════════════════════════════════
       HIDE STREAMLIT CHROME
    ════════════════════════════════════════════════════ */
    footer { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }
    .stDeployButton { display: none !important; }

    /* ════════════════════════════════════════════════════
       ALERTS
    ════════════════════════════════════════════════════ */
    .stAlert {
        border-radius: 14px !important;
        font-family: 'Inter', sans-serif !important;
        backdrop-filter: blur(12px) !important;
        background: rgba(255,255,255,0.07) !important;
    }
    .stCheckbox label, .stToggle label { color: rgba(255,255,255,0.65) !important; font-family: 'Inter', sans-serif !important; }

    /* ════════════════════════════════════════════════════
       REDUCED MOTION ACCESSIBILITY
    ════════════════════════════════════════════════════ */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
        }
    }

    /* ════════════════════════════════════
       GLITCH EFFECT — SIDEBAR LOGO
    ════════════════════════════════════ */
    .truthlens-logo {
        position: relative;
        display: inline-block;
    }
    .truthlens-logo:hover {
        animation: glitch 0.4s linear;
    }
    @keyframes glitch {
        0%   { transform: translate(0); }
        20%  { transform: translate(-3px, 2px); filter: hue-rotate(40deg); }
        40%  { transform: translate(3px, -2px); filter: hue-rotate(-40deg); }
        60%  { transform: translate(-2px, 1px); filter: hue-rotate(90deg); }
        80%  { transform: translate(2px, -1px); }
        100% { transform: translate(0); filter: none; }
    }

    /* ════════════════════════════════════
       ROTATING CONIC GRADIENT HERO OVERLAY
    ════════════════════════════════════ */
    .hero-section {
        position: relative;
    }
    .hero-section::after {
        content: '';
        position: absolute;
        top: 50%; left: 50%;
        width: 700px; height: 700px;
        transform: translate(-50%, -50%);
        background: conic-gradient(
            from 0deg,
            transparent 0deg,
            rgba(124,58,237,0.05) 60deg,
            rgba(6,182,212,0.07) 120deg,
            transparent 180deg,
            rgba(167,139,250,0.05) 240deg,
            transparent 360deg
        );
        border-radius: 50%;
        animation: conicSpin 20s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    @keyframes conicSpin { to { transform: translate(-50%, -50%) rotate(360deg); } }

    /* ════════════════════════════════════
       VERDICT BADGE — 3D DEPTH
    ════════════════════════════════════ */
    .verdict-badge-3d {
        transform: perspective(400px) translateZ(0);
        transition: transform 0.3s ease !important;
        box-shadow:
            0 0 0 1px currentColor,
            0 4px 24px rgba(0,0,0,0.45),
            inset 0 1px 0 rgba(255,255,255,0.12) !important;
    }
    .verdict-badge-3d:hover {
        transform: perspective(400px) translateZ(18px) scale(1.05) !important;
    }

    /* ════════════════════════════════════
       FACT-CHECK LINK BUTTONS
    ════════════════════════════════════ */
    .fc-link-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 18px;
        border-radius: 10px;
        border: 1px solid rgba(6,182,212,0.25);
        background: rgba(6,182,212,0.08);
        color: #06B6D4;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.80rem;
        font-weight: 600;
        text-decoration: none !important;
        transition: all 0.3s ease;
        margin: 4px;
        cursor: pointer;
    }
    .fc-link-btn:hover {
        background: rgba(6,182,212,0.20);
        border-color: rgba(6,182,212,0.55);
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(6,182,212,0.20);
        color: #F1F0FF;
    }

    /* ════════════════════════════════════
       MEDIA LITERACY TIP CARDS
    ════════════════════════════════════ */
    .ml-tip-fake {
        background: rgba(239,68,68,0.07);
        border: 1px solid rgba(239,68,68,0.18);
        border-left: 3px solid #EF4444;
        border-radius: 0 12px 12px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #F1F0FF;
        line-height: 1.6;
        transition: border-color 0.3s;
    }
    .ml-tip-real {
        background: rgba(16,185,129,0.07);
        border: 1px solid rgba(16,185,129,0.18);
        border-left: 3px solid #10B981;
        border-radius: 0 12px 12px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #F1F0FF;
        line-height: 1.6;
    }
    .ml-tip-warn {
        background: rgba(245,158,11,0.07);
        border: 1px solid rgba(245,158,11,0.18);
        border-left: 3px solid #F59E0B;
        border-radius: 0 12px 12px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #F1F0FF;
        line-height: 1.6;
    }

    /* ════════════════════════════════════
       SHARE PANEL
    ════════════════════════════════════ */
    .share-panel {
        background: rgba(12,12,24,0.85);
        border: 1px solid rgba(123,92,240,0.18);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 16px 0;
    }
    .share-code {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.82rem;
        color: #C084FC;
        background: rgba(123,92,240,0.10);
        border: 1px solid rgba(123,92,240,0.20);
        border-radius: 10px;
        padding: 14px 16px;
        line-height: 1.7;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* ════════════════════════════════════
       SIDEBAR LOGO ENHANCED
    ════════════════════════════════════ */
    .sidebar-logo-wrap {
        text-align: center;
        padding: 20px 8px 8px 8px;
        cursor: pointer;
    }
    .sidebar-logo-text {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 900;
        letter-spacing: 0.18em;
        background: linear-gradient(135deg, #06B6D4 0%, #7C3AED 50%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        background-size: 200% 200%;
        animation: gradShift 6s ease-in-out infinite;
        display: block;
        margin-bottom: 4px;
    }
    .sidebar-logo-sub {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.60rem;
        color: rgba(255,255,255,0.45);
        letter-spacing: 0.28em;
        text-transform: uppercase;
    }
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(124,58,237,0.50), rgba(6,182,212,0.30), transparent);
        margin: 14px 0 6px 0;
        animation: divGlow 4s ease-in-out infinite;
    }
    @keyframes divGlow {
        0%,100% { opacity: 0.5; }
        50%      { opacity: 1.0; }
    }

    </style>
    """), unsafe_allow_html=True)

    # ── Background effects injected as HTML ───────────────────────────────────
    import random as _rng_mod
    rng = _rng_mod.Random(42)

    bg_html = []

    # Aurora orbs — cosmic palette
    bg_html.append(
        '<div class="aurora-orb" style="width:580px;height:580px;'
        'background:radial-gradient(circle,rgba(124,58,237,0.22),transparent 70%);'
        'top:-140px;left:-140px;animation-delay:0s;animation-duration:14s;"></div>'
    )
    bg_html.append(
        '<div class="aurora-orb" style="width:480px;height:480px;'
        'background:radial-gradient(circle,rgba(6,182,212,0.18),transparent 70%);'
        'bottom:-110px;right:-110px;animation-delay:7s;animation-duration:18s;"></div>'
    )
    bg_html.append(
        '<div class="aurora-orb" style="width:360px;height:360px;'
        'background:radial-gradient(circle,rgba(167,139,250,0.14),transparent 70%);'
        'top:40%;right:5%;animation-delay:14s;animation-duration:22s;"></div>'
    )
    bg_html.append(
        '<div class="aurora-orb" style="width:300px;height:300px;'
        'background:radial-gradient(circle,rgba(124,58,237,0.12),transparent 70%);'
        'top:25%;left:8%;animation-delay:3s;animation-duration:16s;"></div>'
    )

    # Floating particles — z-index 9999 so they always float above everything
    bg_html.append(
        '<div style="position:fixed;inset:0;pointer-events:none;z-index:9999;overflow:hidden;">'
    )
    for _ in range(18):
        sz    = rng.randint(4, 14)
        left  = rng.randint(3, 97)
        delay = rng.uniform(0, 18)
        dur   = rng.uniform(10, 22)
        bg_html.append(
            f'<div class="particle-3d" style="'
            f'width:{sz}px;height:{sz}px;left:{left}%;bottom:-20px;'
            f'animation-delay:{delay:.1f}s;animation-duration:{dur:.1f}s;'
            f'opacity:0.90;"></div>'
        )

    # Vertical grid lines — z-index 9998 (below particles but above page)
    bg_html.append('<div class="bg-grid-3d" style="z-index:9998;pointer-events:none;">')
    for i in range(12):
        left  = i * 8.5 + 2
        delay = i * 0.65
        dur   = 8.5 + i * 0.25
        bg_html.append(
            f'<div class="grid-line" style="'
            f'left:{left:.1f}%;'
            f'animation-delay:{delay:.1f}s;'
            f'animation-duration:{dur:.1f}s;"></div>'
        )
    bg_html.append('</div></div>')

    st.markdown("".join(bg_html), unsafe_allow_html=True)

    # ── JS: Force cosmic gradient on ALL Streamlit wrappers at runtime ─────────
    st.markdown(clean_html("""
    <script>
    (function applyCosmicTheme() {
        const BG = 'linear-gradient(135deg,#0f0c29 0%,#302b63 50%,#24243e 100%)';
        const selectors = [
            'html', 'body',
            '.stApp',
            '[data-testid="stAppViewContainer"]',
            '[data-testid="stAppViewBlockContainer"]',
            '[data-testid="stMain"]',
            '[data-testid="stMainBlockContainer"]',
            '.main',
            '.appview-container'
        ];
        function paint() {
            selectors.forEach(function(sel) {
                document.querySelectorAll(sel).forEach(function(el) {
                    el.style.setProperty('background', BG, 'important');
                    el.style.setProperty('background-attachment', 'fixed', 'important');
                    el.style.setProperty('min-height', '100vh', 'important');
                });
            });
        }
        paint();
        // Re-apply after Streamlit re-renders
        var obs = new MutationObserver(paint);
        obs.observe(document.body, {childList:true, subtree:true, attributes:false});
        // Also apply after 500ms, 1s, 2s for safety
        [500, 1000, 2000, 4000].forEach(function(t){ setTimeout(paint, t); });
    })();
    </script>
    """), unsafe_allow_html=True)




# ─────────────────────────────────────────────────────────────────────────────

# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Initialize all session state variables with defaults."""
    profile = load_profile()
    defaults = {
        "analysis_history": profile.get("analysis_history", []),  # Past analysis summaries
        "current_result": None,          # Latest analysis result
        "truth_hunter_points": profile.get("truth_hunter_points", 0),  # Gamification points
        "analyses_count": profile.get("analyses_count", 0),            # Total analyses
        "analysis_in_progress": False,   # True only while a request is running
        "checklist_state": [False] * 7,  # Credibility checklist
        "pipeline": None,               # Lazy-loaded PredictionPipeline
        "models_loaded": False,          # Whether models are on disk
        "3d_mode": True,                 # Enable 3D visualizations
        "claude_qa_history": [],         # Claude Q&A conversation history
        "claude_question_input": "",     # Pre-fill for question box
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state.pop("demo_mode", None)


def load_profile() -> Dict[str, Any]:
    """Load persisted sidebar progress so analysis reruns do not crash or reset state."""
    default_profile = {
        "truth_hunter_points": 0,
        "analyses_count": 0,
        "analysis_history": [],
    }
    if not PROFILE_PATH.exists():
        return default_profile

    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {
            "truth_hunter_points": int(raw.get("truth_hunter_points", 0)),
            "analyses_count": int(raw.get("analyses_count", 0)),
            "analysis_history": raw.get("analysis_history", [])[:20],
        }
    except Exception as e:
        logger.warning(f"Could not load saved profile: {e}")
        return default_profile


def save_profile(points: int, analyses_count: int, analysis_history: List[Dict[str, Any]]) -> None:
    """Persist a lightweight user progress snapshot for sidebar gamification."""
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    sanitized_history = []
    for item in analysis_history[-20:]:
        if not isinstance(item, dict):
            continue
        sanitized_history.append({
            "title": str(item.get("title", ""))[:160],
            "verdict": str(item.get("verdict", ""))[:80],
            "truth_score": float(item.get("truth_score", 0)),
            "timestamp": str(item.get("timestamp", ""))[:40],
        })

    payload = {
        "truth_hunter_points": int(points),
        "analyses_count": int(analyses_count),
        "analysis_history": sanitized_history,
    }

    try:
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save profile: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Lazy Model Loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Load the prediction pipeline (cached so it's only loaded once).

    Parameters
    ----------
    Returns
    -------
    PredictionPipeline instance
    """
    try:
        from models.predict import PredictionPipeline
        return PredictionPipeline(demo_mode=False)
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Render the application sidebar with navigation and settings."""
    with st.sidebar:
        st.markdown(clean_html("""
        <div class="sidebar-logo-wrap truthlens-logo">
            <span class="sidebar-logo-text">TRUTHLENS</span>
            <span class="sidebar-logo-sub">See Through the Noise</span>
            <div class="sidebar-divider"></div>
        </div>
        """), unsafe_allow_html=True)

        # Always-on 3D mode
        if HAS_3D:
            st.session_state["3d_mode"] = True
            st.session_state["animation_speed"] = 1.5
            st.session_state["reduce_motion"] = False

        # ── Gamification ─────────────────────────────────────────────────────
        st.markdown("**🏆 TruthHunter Score**")
        points = st.session_state.truth_hunter_points
        _render_badge_system(points)

        st.divider()

        # ── Session History ──────────────────────────────────────────────────
        st.markdown("**📋 Recent Analyses**")
        history = st.session_state.analysis_history[-10:][::-1]

        if not history:
            st.markdown(
                "<div style='font-size:0.8rem; color:#8892A4;'>No analyses yet.</div>",
                unsafe_allow_html=True
            )
        else:
            for i, item in enumerate(history[:5]):
                score = item.get("truth_score", 0)
                verdict = item.get("verdict", "?")
                preview = item.get("text_preview", "")[:40] + "..."
                color = "#FF4444" if score < 30 else "#FFB800" if score < 55 else "#00C851"
                st.markdown(
                    f"""<div class='history-item'>
<span style='color:{color}; font-size:1rem;'>●</span>
<span style='font-size:0.88rem; color:#F0F4FF; margin-left:6px;'>{preview}</span><br>
<span style='font-size:0.80rem; color:#8892A4; margin-left:18px;'>Score: {score:.0f}/100</span>
</div>""",
                    unsafe_allow_html=True
                )

            if len(history) > 1:
                scores = [h["truth_score"] for h in history]
                _render_mini_trend(scores)

        st.divider()

        # ── About ────────────────────────────────────────────────────────────
        with st.expander("ℹ️ About TruthLens"):
            st.markdown(clean_html("""
            **TruthLens v1.0**

            🔍 TruthLens v1.0 — AI Fake News Detector

            👩‍💻 **Author:** Sinega Selvakumar

            🤖 **ML Models:** LR, RF, PAC, SVC,
            Voting Ensemble, DistilBERT,
            K-Means, Isolation Forest

            📊 **Dataset:** WELFake (72,134 articles)

            🔒 **Privacy:** All analysis is local.
            No data sent to external servers.
            """))


def _render_badge_system(points: int) -> None:
    """Render the gamification badge progress display."""
    badge_thresholds = [
        (0,   "🥚 Novice",        "#8892A4"),
        (50,  "🔍 Fact-Finder",   "#FFB800"),
        (150, "🛡️ Truth Seeker",  "#00D4FF"),
        (300, "🏆 TruthLens Expert", "#7B2FBE"),
    ]

    current_badge = badge_thresholds[0]
    next_badge = badge_thresholds[1]
    for i, (threshold, name, color) in enumerate(badge_thresholds):
        if points >= threshold:
            current_badge = (threshold, name, color)
            if i + 1 < len(badge_thresholds):
                next_badge = badge_thresholds[i + 1]
            else:
                next_badge = None

    _, badge_name, badge_color = current_badge
    st.markdown(
        f"<div class='points-display' style='color:{badge_color};'>{points} pts</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='text-align:center; font-size:0.9rem; color:{badge_color};'>{badge_name}</div>",
        unsafe_allow_html=True
    )

    if next_badge:
        next_threshold, next_name, _ = next_badge
        progress = min(points / next_threshold, 1.0)
        remaining = next_threshold - points
        st.progress(progress)
        st.markdown(
            f"<div style='font-size:0.72rem; color:#8892A4; text-align:center;'>"
            f"{remaining} pts to {next_name}</div>",
            unsafe_allow_html=True
        )


def _render_mini_trend(scores: list) -> None:
    """Render a small sparkline of recent truth scores."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=scores[::-1],
        mode="lines+markers",
        line=dict(color="#00D4FF", width=2),
        marker=dict(size=4, color="#7B2FBE"),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.05)",
    ))
    fig.update_layout(
        height=80, margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[0, 100]),
        showlegend=False,
    )
    st.plotly_chart(fig, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# Input Panel
# ─────────────────────────────────────────────────────────────────────────────

def render_input_panel() -> Tuple[str, str, str]:
    """
    Render the 3-tab input panel (Text / URL / Image).

    Returns
    -------
    (article_text, title_text, input_source)
    """
    st.markdown(clean_html("""
    <div style='margin-bottom: 8px;'>
        <h3 style="font-family: 'Space Grotesk'; color: #F0F4FF; margin: 0;">
            📥 Article Input
        </h3>
        <p style="color: #8892A4; font-size: 0.85rem; margin: 4px 0 0 0;">
            Paste text, enter a URL, or upload one or more images to analyze.
        </p>
    </div>
    """), unsafe_allow_html=True)

    tab_text, tab_url, tab_image = st.tabs(["📝 Paste Text", "🌐 Enter URL", "🖼️ Upload Images"])
    article_text, title_text, input_source = "", "", "text"

    with tab_text:
        title_text = st.text_input(
            "Article Headline (optional)",
            key="article_headline",
            placeholder="Enter the article headline for better analysis...",
            help="Including the headline improves headline-body consistency analysis."
        )
        article_text = st.text_area(
            "Article Body",
            key="article_body",
            placeholder="Paste the full article text here...\n\nTip: Longer articles give more accurate results.",
            height=250,
            help="Minimum 50 words recommended for meaningful analysis."
        )
        word_count = len(article_text.split()) if article_text else 0
        st.markdown(
            f"<div style='font-size:0.75rem; color:#8892A4; text-align:right;'>"
            f"Word count: <b style='color:#00D4FF;'>{word_count}</b></div>",
            unsafe_allow_html=True
        )
        input_source = "text"

    with tab_url:
        url_input = st.text_input(
            "Article URL",
            placeholder="https://example.com/article...",
            help="TruthLens will fetch and extract the article text from the URL."
        )
        if url_input:
            with st.spinner("Fetching article from URL..."):
                article_text, title_text = _fetch_from_url(url_input)
                if article_text:
                    st.success(f"✅ Extracted {len(article_text.split())} words.")
                    with st.expander("Preview extracted text"):
                        st.text(article_text[:800] + "...")
                else:
                    st.error("❌ Could not extract text from this URL.")
        input_source = "url"

    with tab_image:
        uploaded_files = st.file_uploader(
            "Upload news screenshots or meme images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="TruthLens uses OCR (pytesseract) to extract text from one or more images."
        )
        if uploaded_files:
            extracted_items = []
            progress = st.progress(0, text="Extracting text from images...")

            for idx, uploaded_file in enumerate(uploaded_files, start=1):
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                except Exception as e:
                    st.error(f"Could not read {uploaded_file.name}: {e}")
                    continue

                text = _ocr_extract(image)
                extracted_items.append((idx, uploaded_file.name, image, text))
                progress.progress(idx / len(uploaded_files), text=f"Processed {idx}/{len(uploaded_files)} images")

            progress.empty()

            combined_text = "\n\n".join(
                text for _, _, _, text in extracted_items if text
            ).strip()
            extracted_count = sum(1 for _, _, _, text in extracted_items if text)

            if extracted_items:
                for idx, file_name, image, text in extracted_items:
                    expanded = len(extracted_items) == 1
                    with st.expander(f"Image {idx}: {file_name}", expanded=expanded):
                        col_img, col_ocr = st.columns([1, 1])
                        with col_img:
                            st.image(image, caption=file_name, width='stretch')
                        with col_ocr:
                            if text:
                                st.success(f"Extracted {len(text.split())} words.")
                                st.text_area(
                                    "Image OCR Text",
                                    text,
                                    height=140,
                                    key=f"ocr_preview_{idx}_{file_name}",
                                    disabled=True,
                                )
                            else:
                                st.warning("No readable text detected in this image.")

            if combined_text:
                file_signature = "|".join(
                    f"{file.name}:{getattr(file, 'size', 0)}" for file in uploaded_files
                )
                ocr_edit_key = f"ocr_extracted_text_{abs(hash(file_signature))}"
                st.success(
                    f"Extracted {len(combined_text.split())} words from "
                    f"{extracted_count}/{len(uploaded_files)} image(s)."
                )
                article_text = st.text_area(
                    "Combined Extracted Text (editable)",
                    combined_text,
                    height=240,
                    key=ocr_edit_key,
                    help="Review and correct OCR text before analyzing."
                )
                title_text = " / ".join(Path(file.name).stem for file in uploaded_files[:3])
            else:
                st.warning("No text detected. Try clearer, higher-resolution images.")
        input_source = "image"

    # Language detection warning
    if article_text and len(article_text.split()) > 10:
        _check_language(article_text)

    return article_text, title_text, input_source


def _fetch_from_url(url: str) -> Tuple[str, str]:
    """
    Attempt to extract article text from a URL using trafilatura or newspaper3k.

    Parameters
    ----------
    url : str

    Returns
    -------
    (body_text, title)
    """
    title, body = "", ""

    # Try trafilatura first
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            result = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                output_format="txt"
            )
            if result and len(result.split()) > 20:
                body = result
                # Extract title from metadata
                metadata = trafilatura.extract_metadata(downloaded)
                if metadata:
                    title = metadata.title or ""
                return body, title
    except Exception as e:
        logger.warning(f"trafilatura failed: {e}")

    # Fallback: newspaper3k
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        body = article.text
        title = article.title
        return body, title
    except Exception as e:
        logger.warning(f"newspaper3k failed: {e}")

    # Last fallback: simple regex scrape
    try:
        import urllib.request
        import html
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw_html = resp.read().decode("utf-8", errors="ignore")
        # Strip tags
        text = re.sub(r"<[^>]+>", " ", raw_html)
        text = html.unescape(re.sub(r"\s+", " ", text)).strip()[:10000]
        # Extract title
        title_match = re.search(r"<title>(.*?)</title>", raw_html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""
        if len(text.split()) > 20:
            return text, title
    except Exception as e:
        logger.warning(f"Fallback URL fetch failed: {e}")

    return "", ""


def _configure_tesseract(pytesseract_module) -> bool:
    """Find a local Tesseract executable and point pytesseract at it."""
    configured_cmd = getattr(pytesseract_module.pytesseract, "tesseract_cmd", "")
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        configured_cmd,
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\Admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract_module.pytesseract.tesseract_cmd = str(candidate)
            return True

    return False


def _resize_for_ocr(image: Image.Image) -> Image.Image:
    """Upscale small screenshots enough for OCR without creating huge images."""
    width, height = image.size
    if width <= 0 or height <= 0:
        return image

    target_width = 2200
    max_width = 3600
    scale = 1.0
    if width < target_width:
        scale = min(3.0, target_width / width)
    if width * scale > max_width:
        scale = max_width / width

    if scale <= 1.05:
        return image

    return image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)


def _prepare_ocr_variants(image: Image.Image) -> List[Image.Image]:
    """Create OCR-friendly variants for light articles and dark screenshots."""
    base = _resize_for_ocr(image.convert("RGB"))
    gray = ImageOps.grayscale(base)

    # Tesseract performs best with dark text on a light background.
    if ImageStat.Stat(gray).mean[0] < 128:
        gray = ImageOps.invert(gray)

    gray = ImageOps.autocontrast(gray)
    enhanced = ImageEnhance.Contrast(gray).enhance(1.8)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
    enhanced = enhanced.filter(ImageFilter.SHARPEN)
    threshold = enhanced.point(lambda p: 255 if p > 165 else 0)

    return [base, enhanced, threshold]


def _clean_ocr_text(text: str) -> str:
    """Normalize OCR output and remove obvious scanner noise."""
    text = text.replace("\x0c", "")
    cleaned_lines = []

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue

        alnum_count = sum(ch.isalnum() for ch in line)
        alnum_ratio = alnum_count / max(len(line), 1)
        if len(line) < 3 and alnum_ratio < 0.5:
            continue
        if len(line) < 20 and alnum_ratio < 0.25:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _ocr_extract(image: Image.Image) -> str:
    """
    Extract visible text from an image using pytesseract (OCR).

    Parameters
    ----------
    image : PIL.Image

    Returns
    -------
    str
    """
    try:
        import pytesseract
    except ImportError:
        st.warning("pytesseract is not installed. Run: pip install pytesseract")
        return ""

    if not _configure_tesseract(pytesseract):
        if not st.session_state.get("_tesseract_missing_notice_shown"):
            st.error(
                "Tesseract OCR is not installed or is not on PATH. "
                "Install Tesseract OCR, then restart the Streamlit app."
            )
            st.session_state["_tesseract_missing_notice_shown"] = True
        return ""

    candidates = []
    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
    ]

    for variant in _prepare_ocr_variants(image):
        for config in configs:
            try:
                text = pytesseract.image_to_string(variant, lang="eng", config=config)
            except Exception as e:
                logger.warning(f"OCR failed for config {config}: {e}")
                continue

            cleaned = _clean_ocr_text(text)
            if cleaned:
                candidates.append(cleaned)

    if not candidates:
        return ""

    return max(candidates, key=lambda value: (len(value.split()), len(value)))


def _check_language(text: str) -> None:
    """Show warning banner if text is not in English."""
    try:
        from langdetect import detect, LangDetectException
        lang = detect(text)
        if lang != "en":
            st.warning(
                f"⚠️ **Non-English Content Detected** — Language detected: `{lang.upper()}`. "
                "TruthLens is optimized for English articles. Results may be less accurate.",
                icon="🌐"
            )
    except Exception:
        pass  # silently skip if langdetect not available


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Runner
# ─────────────────────────────────────────────────────────────────────────────

def _local_fallback_prediction(text: str, title: str = "") -> Dict[str, Any]:
    """Analyze the submitted article with local rules when trained models fail."""
    combined_text = f"{title} {text}".strip()

    try:
        from models.predict import PredictionPipeline
        rule_result = PredictionPipeline._rule_based_predict(text, title)
        lime_weights = PredictionPipeline._rule_based_weights(combined_text)
        cosine_similarity = PredictionPipeline._token_overlap_similarity(combined_text)
    except Exception:
        full = combined_text.lower()
        fake_signals = [
            "shocking", "you won't believe", "bombshell", "cover-up",
            "conspiracy", "miracle", "secret", "leaked", "elites",
            "they don't want you to know", "breaking", "scandal",
        ]
        real_signals = [
            "according to", "study", "research", "published",
            "confirmed", "official", "data", "evidence", "court documents",
        ]
        fake_score = sum(1 for signal in fake_signals if signal in full)
        real_score = sum(1 for signal in real_signals if signal in full)
        verdict = "FAKE" if fake_score > real_score else "REAL"
        signal_count = fake_score if verdict == "FAKE" else real_score
        rule_result = {
            "verdict": verdict,
            "confidence": min(0.5 + signal_count * 0.08, 0.9),
            "label": 0 if verdict == "FAKE" else 1,
        }
        lime_weights = []
        cosine_similarity = min(real_score / 5.0, 1.0)

    is_fake = rule_result.get("verdict") == "FAKE"
    confidence = float(rule_result.get("confidence", 0.5))

    return {
        "model_predictions": {"rule_based": rule_result},
        "ensemble_label": rule_result.get("verdict", "UNCERTAIN"),
        "ensemble_confidence": confidence,
        "cosine_similarity": round(float(cosine_similarity), 4),
        "anomaly_score": 0.65 if is_fake else 0.35,
        "cluster_point": {"x": 0.0, "y": 0.0, "cluster_id": 0},
        "lime_weights": lime_weights,
        "distilbert_result": None,
        "processing_time_ms": 0.0,
    }


def run_analysis(text: str, title: str) -> Optional[Dict]:
    """
    Run the full prediction pipeline and compute all credibility signals.

    Parameters
    ----------
    text : str
    title : str

    Returns
    -------
    dict with all analysis results, or None on failure
    """
    # Load pipeline
    pipeline = load_pipeline()

    # Run prediction
    try:
        if pipeline is None:
            raise RuntimeError("Prediction pipeline could not be loaded.")
        pred_result = pipeline.predict(text, title, demo_mode=False)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        pred_result = _local_fallback_prediction(text, title)

    # Compute linguistic features
    from utils.feature_extractor import FeatureExtractor
    from utils.text_preprocessor import TextPreprocessor
    from utils.credibility_scorer import CredibilityScorer

    preprocessor = TextPreprocessor()
    fe = FeatureExtractor()
    scorer = CredibilityScorer()

    try:
        ling_feats = fe.extract_linguistic_features(text, title)
        radar_scores = fe.get_radar_scores(text, title)

        ling_cred = scorer.compute_linguistic_credibility(
            passive_voice_ratio=ling_feats.get("passive_voice_ratio", 0.1),
            avg_word_length=ling_feats.get("avg_word_len", 4.5),
            caps_ratio=ling_feats.get("caps_ratio", 0.02),
            exclamation_ratio=ling_feats.get("exclamation_ratio", 0.1),
        )

        truth_score, breakdown = scorer.compute_truth_score(
            linguistic_credibility=ling_cred,
            sentiment_balance=ling_feats.get("sentiment_balance", 0.5),
            source_citation_index=ling_feats.get("source_citation_index", 0.3),
            headline_body_consistency=ling_feats.get("headline_body_similarity", 0.5),
            clickbait_probability=ling_feats.get("clickbait_probability", 0.3),
            model_confidence=pred_result.get("ensemble_confidence", 0.5),
            model_label=pred_result.get("ensemble_label", "UNCERTAIN"),
        )

        verdict_text, verdict_color = scorer.get_verdict(truth_score)
        red_flags = scorer.get_redflag_explanation(breakdown, truth_score)

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        truth_score = 50.0
        verdict_text, verdict_color = "🟡 SUSPICIOUS", "#FFB800"
        breakdown = {
            "Linguistic Credibility": 50.0, "Sentiment Balance": 50.0,
            "Source Citation Index": 30.0, "Headline–Body Consistency": 50.0,
            "Clickbait Probability": 40.0
        }
        radar_scores = {k: 50.0 for k in [
            "Sensationalism", "Emotional Bias", "Clickbait Language",
            "Source Density", "Claim Frequency", "Complexity"
        ]}
        red_flags = ["⚠️ Analysis partially completed due to an error."]

    # Top red-flag words for word cloud
    try:
        from wordcloud import WordCloud
        wc_text = text.lower()
        from utils.feature_extractor import SENSATIONAL_WORDS
        redflag_words = {
            w: wc_text.count(w)
            for w in SENSATIONAL_WORDS
            if wc_text.count(w) > 0
        }
    except Exception:
        redflag_words = {}

    # Reading Level (Flesch-Kincaid)
    try:
        words_list = text.split()
        sentences_list = re.split(r'[.!?]+', text)
        sentences_list = [s for s in sentences_list if len(s.strip()) > 3]
        total_syllables = sum(
            max(1, len(re.findall(r'[aeiouAEIOU]', w)) - len(re.findall(r'[aeiouAEIOU]{2,}', w)))
            for w in words_list
        )
        n_words = max(len(words_list), 1)
        n_sentences = max(len(sentences_list), 1)
        flesch = 206.835 - 1.015*(n_words/n_sentences) - 84.6*(total_syllables/n_words)
        flesch = max(0, min(100, flesch))
        if flesch >= 70:
            reading_level = "Easy (Grade 6-7)"
        elif flesch >= 50:
            reading_level = "Standard (Grade 8-10)"
        elif flesch >= 30:
            reading_level = "Difficult (Grade 11-13)"
        else:
            reading_level = "Very Difficult (College+)"
    except Exception:
        flesch = 50.0
        reading_level = "Standard"

    # Named Entity Extraction (simple regex-based, no spacy needed)
    try:
        # Find capitalized multi-word phrases (likely named entities)
        entity_pattern = r'\b([A-Z][a-z]+ (?:[A-Z][a-z]+ )*(?:[A-Z][a-z]+))\b'
        raw_entities = re.findall(entity_pattern, text)
        # Deduplicate and filter noise words
        noise = {"The", "This", "That", "These", "Those", "They", "He", "She",
                 "According", "Breaking", "Share", "Click", "Read"}
        entities = list(dict.fromkeys(
            e for e in raw_entities if e not in noise and len(e.split()) >= 2
        ))[:12]
    except Exception:
        entities = []

    # Emotional manipulation score
    try:
        fear_words   = ["terrifying","deadly","danger","threat","crisis","catastrophe",
                        "panic","alarming","horrifying","disaster","warning","attack"]
        anger_words  = ["outrage","disgust","shameful","corrupt","betrayal","evil",
                        "disgusting","furious","lie","liar","fraud","rigged","cheat"]
        disgust_words= ["disgusting","revolting","sick","vile","repulsive","filth",
                        "trash","garbage","awful","terrible","horrible"]
        text_lower_em = text.lower()
        fear_score   = min(sum(1 for w in fear_words   if w in text_lower_em) / 4.0, 1.0)
        anger_score  = min(sum(1 for w in anger_words  if w in text_lower_em) / 4.0, 1.0)
        disgust_score= min(sum(1 for w in disgust_words if w in text_lower_em) / 3.0, 1.0)
        emotional_score = round((fear_score * 0.4 + anger_score * 0.4 + disgust_score * 0.2) * 100, 1)
    except Exception:
        fear_score = anger_score = disgust_score = 0.0
        emotional_score = 0.0

    # Compile full result
    result = {
        # Core
        "text": text,
        "title": title,
        "text_preview": text[:100],
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        # Scores
        "truth_score": truth_score,
        "verdict": verdict_text,
        "verdict_color": verdict_color,
        "ensemble_label": pred_result.get("ensemble_label", "UNCERTAIN"),
        "ensemble_confidence": pred_result.get("ensemble_confidence", 0.5),
        # Breakdown
        "credibility_breakdown": breakdown,
        "radar_scores": radar_scores,
        # Model predictions
        "model_predictions": pred_result.get("model_predictions", {}),
        # Explainability
        "lime_weights": pred_result.get("lime_weights", []),
        "red_flags": red_flags,
        "redflag_words": redflag_words,
        # Unsupervised
        "cosine_similarity": pred_result.get("cosine_similarity", 0.5),
        "anomaly_score": pred_result.get("anomaly_score", 0.5),
        "cluster_point": pred_result.get("cluster_point", {}),
        # Misc
        "distilbert_result": pred_result.get("distilbert_result"),
        "processing_time_ms": pred_result.get("processing_time_ms", 0.0),
        "word_count": len(text.split()),
        # Advanced Linguistic Insights
        "flesch_score": flesch,
        "reading_level": reading_level,
        "named_entities": entities,
        "emotional_score":  emotional_score,
        "fear_score":       round(fear_score * 100, 1),
        "anger_score":      round(anger_score * 100, 1),
        "disgust_score":    round(disgust_score * 100, 1),
    }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard Components
# ─────────────────────────────────────────────────────────────────────────────

def render_truth_score_gauge(truth_score: float, verdict: str, verdict_color: str) -> None:
    """
    Render the TruthScore™ gauge (2D or 3D based on settings).

    Parameters
    ----------
    truth_score : float (0-100)
    verdict : str
    verdict_color : str (hex)
    """
    use_3d = st.session_state.get("3d_mode", True) and HAS_3D
    speed_mult = st.session_state.get("animation_speed", 1.0)

    if use_3d:
        # 3D rotating sphere gauge with animation
        fig = create_3d_truth_score_sphere(truth_score, verdict, speed_multiplier=speed_mult)
        st.markdown('<div class="chart-3d float-3d">', unsafe_allow_html=True)
        st.plotly_chart(fig, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Original 2D gauge
        if truth_score <= 30:
            bar_color = "#FF4444"
        elif truth_score <= 55:
            bar_color = "#FFB800"
        else:
            bar_color = "#00C851"

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=truth_score,
            number={
                "suffix": "/100",
                "font": {"size": 36, "color": bar_color, "family": "Space Grotesk"},
            },
            title={
                "text": "TruthScore™",
                "font": {"size": 16, "color": "#8892A4", "family": "Space Grotesk"}
            },
            delta={
                "reference": 50,
                "increasing": {"color": "#00C851"},
                "decreasing": {"color": "#FF4444"},
                "font": {"size": 14}
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#A0A5B5",
                    "tickfont": {"color": "#A0A5B5", "size": 10},
                },
                "bar": {"color": bar_color, "thickness": 0.25},
                "bgcolor": "rgba(18,18,22,0.5)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],  "color": "rgba(255,68,68,0.15)"},
                    {"range": [30, 55], "color": "rgba(255,184,0,0.15)"},
                    {"range": [55, 100],"color": "rgba(0,200,81,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 2},
                    "thickness": 0.7,
                    "value": truth_score
                },
            }
        ))

        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter"},
        )

        st.plotly_chart(fig, config={"displayModeBar": False})

    # Verdict Badge with animated class based on verdict type
    if truth_score < 30:
        verdict_anim_class = "verdict-fake-anim"
    elif truth_score <= 55:
        verdict_anim_class = "verdict-suspicious-anim"
    else:
        verdict_anim_class = "verdict-real-anim"

    st.markdown(
        f"""<div style="text-align:center; margin: 10px 0;">
        <span class="verdict-badge-3d {verdict_anim_class}" style="background: linear-gradient(135deg, {verdict_color}35, {verdict_color}15);
        border: 2px solid {verdict_color}; color: {verdict_color};">{verdict}</span>
        </div>""",
        unsafe_allow_html=True
    )



def render_algorithm_comparison(model_predictions: dict) -> None:
    """
    Render algorithm comparison (2D or 3D based on settings).

    Parameters
    ----------
    model_predictions : dict mapping model_key → {'verdict', 'confidence'}
    """
    if not model_predictions:
        st.info("No model predictions available. Check that the local models are trained.")
        return

    use_3d = st.session_state.get("3d_mode", True) and HAS_3D
    speed_mult = st.session_state.get("animation_speed", 1.0)

    if use_3d:
        fig = create_3d_algorithm_comparison(model_predictions, speed_multiplier=speed_mult)
        st.markdown('<div class="chart-3d">', unsafe_allow_html=True)
        st.plotly_chart(fig, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        display_names = {
            "logistic_regression": "Logistic Regression",
            "random_forest":       "Random Forest",
            "passive_aggressive":  "Passive Aggressive",
            "linear_svc":          "LinearSVC",
            "voting_ensemble":     "Voting Ensemble",
            "rule_based":          "Rule-Based",
        }

        names, confidences, verdicts, colors = [], [], [], []
        for key, pred in model_predictions.items():
            names.append(display_names.get(key, key.replace("_", " ").title()))
            conf = pred.get("confidence", 0.5) * 100
            verdict = pred.get("verdict", "UNCERTAIN")
            confidences.append(conf)
            verdicts.append(verdict)
            colors.append(
                "#FF4444" if verdict == "FAKE"
                else "#00C851" if verdict == "REAL"
                else "#FFB800"
            )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=names,
            y=confidences,
            marker_color=colors,
            marker_line=dict(color="rgba(255,255,255,0.1)", width=1),
            text=[f"{c:.1f}%<br><b>{v}</b>" for c, v in zip(confidences, verdicts)],
            textposition="outside",
            textfont=dict(color="#F0F4FF", size=10),
            hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>",
        ))

        fig.add_hline(
            y=50, line_dash="dot",
            line_color="rgba(255,255,255,0.2)",
            annotation_text="50% threshold",
            annotation_font=dict(color="#8892A4", size=10),
        )

        fig.update_layout(
            title=dict(
                text="Algorithm Comparison",
                font=dict(family="Space Grotesk", size=15, color="#F0F4FF")
            ),
            yaxis=dict(
                title=dict(text="Confidence (%)", font=dict(color="#8892A4")),
                range=[0, 110],
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(color="#8892A4"),
            ),
            xaxis=dict(tickfont=dict(color="#8892A4", size=9)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(l=40, r=20, t=50, b=60),
            showlegend=False,
        )

        st.plotly_chart(fig, config={"displayModeBar": False})


def render_radar_chart(radar_scores: dict) -> None:
    """
    Render the 6-axis writing style radar chart.

    Parameters
    ----------
    radar_scores : dict mapping axis_name → score (0-100)
    """
    categories = list(radar_scores.keys())
    values = list(radar_scores.values())
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(123,47,190,0.15)",
        line=dict(color="#7B2FBE", width=2),
        marker=dict(size=6, color="#00D4FF"),
        name="Article Profile",
    ))

    # Add reference: "ideal credible article"
    ideal_values = [20, 20, 15, 80, 40, 60]
    ideal_closed = ideal_values + [ideal_values[0]]
    fig.add_trace(go.Scatterpolar(
        r=ideal_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(0,240,255,0.05)",
        line=dict(color="#00F0FF", width=1.5, dash="dash"),
        name="Ideal Credible Article",
        opacity=0.7,
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(18,18,22,0.4)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="rgba(255,255,255,0.08)",
                tickfont=dict(color="#A0A5B5", size=8),
                tickvals=[25, 50, 75, 100],
            ),
            angularaxis=dict(
                tickfont=dict(color="#FFFFFF", size=10, family="Space Grotesk"),
                gridcolor="rgba(255,255,255,0.06)",
            ),
        ),
        showlegend=True,
        legend=dict(
            font=dict(color="#8892A4", size=9),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        title=dict(
            text="Writing Style Profile",
            font=dict(family="Space Grotesk", size=15, color="#F0F4FF")
        ),
        margin=dict(l=60, r=60, t=60, b=40),
    )

    st.plotly_chart(fig, config={"displayModeBar": False})


def render_credibility_breakdown(breakdown: dict) -> None:
    """
    Render the 5 credibility component progress bars.

    Parameters
    ----------
    breakdown : dict mapping signal_name → score (0-100)
    """
    icons = {
        "Linguistic Credibility": "📝",
        "Sentiment Balance": "⚖️",
        "Source Citation Index": "🔗",
        "Headline–Body Consistency": "🔄",
        "Clickbait Probability": "🎣",
    }
    help_texts = {
        "Linguistic Credibility": "Formality, passive voice ratio, and writing style quality.",
        "Sentiment Balance": "VADER sentiment — neutral articles score higher.",
        "Source Citation Index": "Number of verifiable source references found.",
        "Headline–Body Consistency": "Cosine similarity between headline and body.",
        "Clickbait Probability": "Lower is better. Clickbait lowers credibility.",
    }

    for signal, score in breakdown.items():
        icon = icons.get(signal, "📊")
        is_inverse = signal == "Clickbait Probability"  # Lower = better

        # Display score: for clickbait, show inverted value
        display_score = (100 - score) if is_inverse else score
        bar_color = (
            "#FF4444" if display_score < 35
            else "#FFB800" if display_score < 60
            else "#00C851"
        )

        st.markdown(clean_html(f"""
        <div style="margin: 10px 0;" title="{help_texts.get(signal, '')}">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:0.85rem; color:#F0F4FF;">{icon} {signal}</span>
                <span style="font-size:0.85rem; font-weight:600; color:{bar_color};">{display_score:.0f}/100</span>
            </div>
            <div style="background:rgba(255,255,255,0.06); border-radius:4px; height:8px; overflow:hidden;">
                <div style="width:{display_score}%; height:100%; border-radius:4px;
                            background: linear-gradient(90deg, #7B2FBE, {bar_color});
                            transition: width 0.8s ease;"></div>
            </div>
        </div>
        """), unsafe_allow_html=True)


def render_lime_explanation(lime_weights: list, text: str) -> None:
    """
    Render LIME word-level explainability with color-coded text highlighting.

    Parameters
    ----------
    lime_weights : list of (word, weight) tuples
    text : str — original article text
    """
    if not lime_weights:
        st.info("LIME explanation not available for this article.")
        return

    # Sort by absolute weight
    sorted_weights = sorted(lime_weights, key=lambda x: abs(x[1]), reverse=True)[:15]

    # LIME bar chart
    words = [w for w, _ in sorted_weights]
    weights = [float(v) for _, v in sorted_weights]
    bar_colors = ["#FF4444" if w < 0 else "#00C851" for w in weights]

    fig = go.Figure(go.Bar(
        x=weights,
        y=words,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.05)", width=1)),
        text=[f"{w:+.3f}" for w in weights],
        textposition="outside",
        textfont=dict(color="#F0F4FF", size=9),
        hovertemplate="<b>%{y}</b>: %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text="LIME — Word Influence on Prediction",
            font=dict(family="Space Grotesk", size=14, color="#F0F4FF")
        ),
        xaxis=dict(
            title=dict(text="Influence (negative→FAKE, positive→REAL)", font=dict(color="#8892A4", size=11)),
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="#8892A4"),
            zeroline=True, zerolinecolor="rgba(255,255,255,0.2)", zerolinewidth=1,
        ),
        yaxis=dict(tickfont=dict(color="#F0F4FF", size=10), autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350 + len(sorted_weights) * 8,
        margin=dict(l=10, r=80, t=50, b=40),
    )
    st.plotly_chart(fig, config={"displayModeBar": False})

    # Color-coded sentence highlighting
    st.markdown("**🔤 Sentence-Level Highlights**")
    st.markdown(
        "<div style='font-size:0.75rem; color:#8892A4; margin-bottom:8px;'>"
        "<span style='background:rgba(255,68,68,0.25); color:#FF6B6B; padding:2px 6px; "
        "border-radius:3px;'>Red words</span> → push toward FAKE &nbsp;"
        "<span style='background:rgba(0,200,81,0.25); color:#00E676; padding:2px 6px; "
        "border-radius:3px;'>Green words</span> → push toward REAL"
        "</div>",
        unsafe_allow_html=True
    )

    # Build word→color map
    word_color_map = {}
    for word, weight in sorted_weights:
        if weight < -0.05:
            word_color_map[word.lower()] = "fake"
        elif weight > 0.05:
            word_color_map[word.lower()] = "real"

    # Highlight text
    highlighted_html = _highlight_text(text[:1500], word_color_map)
    st.markdown(
        f"<div style='background:rgba(12,12,14,0.6); padding:16px; border-radius:10px; "
        f"line-height:1.8; font-size:0.9rem; color:#D0D8F0; border:1px solid rgba(0,240,255,0.08);'>"
        f"{highlighted_html}</div>",
        unsafe_allow_html=True
    )


def _highlight_text(text: str, word_color_map: dict) -> str:
    """Apply color highlights to text based on word_color_map."""
    words = text.split(" ")
    highlighted = []
    for word in words:
        clean = re.sub(r"[^a-zA-Z]", "", word).lower()
        if clean in word_color_map:
            css_class = "highlight-fake" if word_color_map[clean] == "fake" else "highlight-real"
            highlighted.append(f'<span class="{css_class}">{word}</span>')
        else:
            highlighted.append(word)
    return " ".join(highlighted)


def render_wordcloud(redflag_words: dict) -> None:
    """
    Render a wordcloud of top red-flag words.

    Parameters
    ----------
    redflag_words : dict mapping word → frequency
    """
    if not redflag_words:
        st.info("No significant red-flag words detected in this article. 🟢 This is a good sign.")
        return

    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        wc = WordCloud(
            width=600, height=280,
            background_color=None, mode="RGBA",
            colormap="RdYlGn_r",
            max_words=30,
            font_path=None,
            prefer_horizontal=0.8,
            min_font_size=10,
        ).generate_from_frequencies(redflag_words)

        fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="none")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.patch.set_alpha(0.0)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        # Fallback: text-based display
        st.markdown("**Top Red-Flag Words:**")
        sorted_words = sorted(redflag_words.items(), key=lambda x: x[1], reverse=True)[:10]
        word_html = " ".join(
            f'<span class="highlight-fake" style="font-size:{14 + v * 2}px;">{w}</span>'
            for w, v in sorted_words
        )
        st.markdown(
            f"<div style='padding:16px; line-height:2.5; text-align:center;'>{word_html}</div>",
            unsafe_allow_html=True
        )


def render_cluster_explorer(cluster_data: dict, new_point: dict) -> None:
    """
    Render the 2D or 3D K-Means article cluster scatter plot.

    Parameters
    ----------
    cluster_data : dict with 'x', 'y', 'cluster', 'true_label', 'text_preview'
    new_point : dict with 'x', 'y', 'cluster_id' for the new article
    """
    if not cluster_data or not cluster_data.get("x"):
        st.info("Cluster data not available. Run `python models/train_models.py` to generate cluster map.")
        return

    use_3d = st.session_state.get("3d_mode", True) and HAS_3D
    speed_mult = st.session_state.get("animation_speed", 1.0)

    if use_3d:
        fig = create_3d_cluster_explorer(cluster_data, new_point, speed_multiplier=speed_mult)
        st.markdown('<div class="chart-3d">', unsafe_allow_html=True)
        st.plotly_chart(fig, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        df_cluster = pd.DataFrame({
            "x": cluster_data["x"],
            "y": cluster_data["y"],
            "cluster": [str(c) for c in cluster_data["cluster"]],
            "true_label": ["Fake" if l == 0 else "Real" for l in cluster_data["true_label"]],
            "text": cluster_data.get("text_preview", [""] * len(cluster_data["x"])),
        })

        color_map = {"Fake": "#FF4444", "Real": "#00C851"}

        fig = px.scatter(
            df_cluster,
            x="x", y="y",
            color="true_label",
            color_discrete_map=color_map,
            hover_data={"text": True, "cluster": True, "x": False, "y": False},
            opacity=0.5,
            size_max=8,
        )
        fig.update_traces(marker=dict(size=5))

        new_color = (
            "#FF4444" if new_point.get("cluster_id", 0) % 2 == 0 else "#00C851"
        )
        fig.add_trace(go.Scatter(
            x=[new_point.get("x", 0)],
            y=[new_point.get("y", 0)],
            mode="markers+text",
            marker=dict(size=18, color="#FFB800", symbol="star",
                        line=dict(color="white", width=2)),
            text=["Your Article"],
            textposition="top center",
            textfont=dict(color="#FFB800", size=11, family="Space Grotesk"),
            name="Your Article",
            showlegend=True,
        ))

        fig.update_layout(
            title=dict(
                text="Article Cluster Map (PCA Visualization)",
                font=dict(family="Space Grotesk", size=14, color="#FFFFFF")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,12,14,0.4)",
            height=400,
            xaxis=dict(title="PC1", gridcolor="rgba(255,255,255,0.04)",
                       tickfont=dict(color="#A0A5B5")),
            yaxis=dict(title="PC2", gridcolor="rgba(255,255,255,0.04)",
                       tickfont=dict(color="#A0A5B5")),
            legend=dict(font=dict(color="#A0A5B5"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=40, r=20, t=50, b=40),
        )

        st.plotly_chart(fig, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# Student Learning Module
# ─────────────────────────────────────────────────────────────────────────────

def render_learning_module() -> None:
    """Render the Student Learning Module with algo explanations and tips."""
    st.markdown(clean_html("""
    <h3 style="font-family: 'Space Grotesk'; color: #F0F4FF;">🎓 Student Learning Module</h3>
    <p style="color: #8892A4; font-size: 0.85rem;">
        Learn how AI detected this and how you can apply these skills manually.
    </p>
    """), unsafe_allow_html=True)

    # ── Algorithm Explanations ─────────────────────────────────────────────────
    with st.expander("🤖 How Did AI Detect This? — Algorithm Explanations", expanded=False):
        algo_info = [
            ("📊 Logistic Regression", "supervised",
             "Uses TF-IDF word frequencies to find statistical patterns that "
             "correlate with fake vs. real news. Think of it as learning which "
             "words appear more often in fake articles."),
            ("🌲 Random Forest", "supervised",
             "Builds hundreds of decision trees and combines their votes. "
             "Each tree independently checks different word patterns, making "
             "it robust and less likely to overfit."),
            ("⚡ Passive Aggressive", "supervised",
             "An online learning algorithm — it updates itself after each mistake. "
             "Great for fast news streams where new types of fake news emerge daily."),
            ("📐 LinearSVC", "supervised",
             "Finds the best hyperplane in a high-dimensional word space to separate "
             "fake from real news. Very fast and accurate on text classification."),
            ("🗳️ Voting Ensemble", "ensemble",
             "Combines all 4 models with learned confidence weights. If 3 out of 4 "
             "models say 'Fake' — the ensemble says 'Fake'. Majority rules!"),
            ("🧬 DistilBERT", "deep learning",
             "A smaller, faster version of BERT. It reads the article like a human "
             "would — understanding context and sentence meaning, not just word counts."),
            ("🔵 K-Means Clustering", "unsupervised",
             "Groups articles by topic similarity without labels. Historical clusters "
             "reveal that certain topic groups have higher rates of misinformation."),
            ("🔍 TF-IDF + Cosine Similarity", "unsupervised",
             "Compares your article's word vectors against known credible sources. "
             "Higher similarity = more credible writing style."),
            ("🚨 Isolation Forest", "unsupervised",
             "Detects statistically anomalous writing patterns. If an article looks "
             "very different from normal journalism, it gets flagged as an outlier."),
        ]

        for algo_name, algo_type, description in algo_info:
            st.markdown(clean_html(f"""
            <div class="algo-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="algo-name">{algo_name}</span>
                    <span style="font-size:0.7rem; color:#8892A4; background:rgba(0,212,255,0.08);
                                 padding:2px 8px; border-radius:10px;">{algo_type}</span>
                </div>
                <div class="algo-desc">{description}</div>
            </div>
            """), unsafe_allow_html=True)

    # ── Fact-Check Tips ────────────────────────────────────────────────────────
    with st.expander("💡 5 Fact-Check Tips You Can Apply Right Now", expanded=False):
        tips = [
            ("🔍 Check the Source", "Who published this? Look up the website domain. "
             "Check if it has an About page and named journalists."),
            ("📅 Verify the Date", "Old news recycled as breaking news is a common tactic. "
             "Always check the publication date."),
            ("🔗 Find the Original Source", "If an article references a 'study' or 'official report', "
             "Google it directly. Don't trust the article's interpretation alone."),
            ("🖼️ Reverse Image Search", "Fake news often uses real photos out of context. "
             "Right-click any image and 'Search Image with Google'."),
            ("📰 Cross-Reference 3 Sources", "If only one outlet is reporting something shocking, "
             "it's a red flag. Credible news is confirmed by multiple independent sources."),
        ]
        for title, body in tips:
            st.markdown(clean_html(f"""
            <div class="tip-card">
                <strong style="color:#00D4FF;">{title}</strong><br>
                {body}
            </div>
            """), unsafe_allow_html=True)

    # ── Credibility Checklist ──────────────────────────────────────────────────
    with st.expander("✅ Credibility Checklist — Self-Evaluate This Article", expanded=False):
        questions = [
            "Is the source a recognized news organization?",
            "Is the author named and verifiable?",
            "Are claims backed by cited sources or data?",
            "Is the headline proportional to the story content?",
            "Is the writing calm and informative (not alarming or emotional)?",
            "Does the story appear in other credible outlets?",
            "Is the publication date recent and relevant?",
        ]
        checked_count = 0
        for i, question in enumerate(questions):
            checked = st.checkbox(question, key=f"checklist_{i}")
            if checked:
                checked_count += 1

        if checked_count == 7:
            st.success("✅ All checks passed! This article shows high credibility indicators.")
        elif checked_count >= 5:
            st.info(f"🟡 {checked_count}/7 checks passed. Moderately credible — verify further.")
        else:
            st.warning(f"⚠️ Only {checked_count}/7 checks passed. Be cautious with this article.")

        # Checklist score gauge
        checklist_score = (checked_count / 7) * 100
        st.progress(checked_count / 7)
        st.markdown(
            f"<div style='font-size:0.8rem; color:#8892A4; text-align:right;'>"
            f"Manual credibility: {checklist_score:.0f}%</div>",
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# PDF Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(result: dict) -> bytes:
    """
    Generate a downloadable PDF analysis report using fpdf2.

    Parameters
    ----------
    result : dict — full analysis result

    Returns
    -------
    bytes — PDF file content
    """
    try:
        from fpdf import FPDF

        class TruthLensPDF(FPDF):
            def header(self):
                self.set_fill_color(10, 14, 26)
                self.rect(0, 0, 210, 297, "F")
                self.set_font("Helvetica", "B", 20)
                self.set_text_color(0, 212, 255)
                self.cell(0, 15, "TruthLens - AI Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
                self.set_font("Helvetica", "", 10)
                self.set_text_color(136, 146, 164)
                self.cell(0, 6, "TruthLens — See Through the Noise | AI Credibility Analysis",
                         align="C", new_x="LMARGIN", new_y="NEXT")
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(136, 146, 164)
                self.cell(0, 10, f"Page {self.page_no()} | TruthLens v1.0 | {datetime.now().strftime('%Y-%m-%d')}")

        pdf = TruthLensPDF()
        pdf.add_page()

        # Enhanced text cleaning for PDF compatibility
        def clean_txt(s):
            if not isinstance(s, str):
                s = str(s)
            # Replace Unicode em-dash, en-dash with regular hyphen
            s = s.replace('—', '-').replace('–', '-')
            # Replace other common Unicode characters
            s = s.replace('"', '"').replace('"', '"')
            s = s.replace(''', "'").replace(''', "'")
            s = s.replace('•', '*').replace('…', '...')
            # Encode to latin-1, ignoring any remaining unsupported characters
            return s.encode("latin-1", errors="ignore").decode("latin-1")

        # Title
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 212, 255)
        pdf.cell(0, 10, clean_txt("ANALYSIS RESULT"), new_x="LMARGIN", new_y="NEXT")

        # Verdict
        score = result.get("truth_score", 0)
        verdict = result.get("verdict", "?")
        pdf.set_font("Helvetica", "B", 16)
        if score <= 30:
            pdf.set_text_color(255, 68, 68)
        elif score <= 55:
            pdf.set_text_color(255, 184, 0)
        else:
            pdf.set_text_color(0, 200, 81)
        pdf.cell(0, 12, clean_txt(f"Verdict: {verdict}"), new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(240, 244, 255)
        pdf.cell(0, 8, clean_txt(f"TruthScore: {score:.1f} / 100"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, clean_txt(f"Analyzed: {result.get('timestamp', '')}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, clean_txt(f"Word Count: {result.get('word_count', 0)}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # Article text preview
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(0, 212, 255)
        pdf.cell(0, 8, clean_txt("Article Preview:"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(200, 210, 220)
        preview = result.get("text", "")[:400] + "..."
        pdf.multi_cell(0, 5, clean_txt(preview))
        pdf.ln(4)

        # Credibility Breakdown
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(0, 212, 255)
        pdf.cell(0, 8, clean_txt("Credibility Breakdown:"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(240, 244, 255)
        for signal, score_val in result.get("credibility_breakdown", {}).items():
            is_inv = signal == "Clickbait Probability"
            display = (100 - score_val) if is_inv else score_val
            pdf.cell(0, 6, clean_txt(f"  {signal}: {display:.0f}/100"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # Red Flags
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(255, 68, 68)
        pdf.cell(0, 8, clean_txt("Red Flags Detected:"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(240, 244, 255)
        for flag in result.get("red_flags", []):
            clean_flag = re.sub(r"\*+|🚩|ℹ️", "", flag).strip()
            pdf.multi_cell(0, 5, clean_txt(f"  • {clean_flag}"))
        pdf.ln(4)

        # Footer note
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(136, 146, 164)
        pdf.multi_cell(0, 5, clean_txt(
            "Disclaimer: TruthLens is an AI tool for educational purposes. "
            "Always verify information with multiple trusted sources before making conclusions."
        ))

        return pdf.output()

    except ImportError:
        st.error("fpdf2 not installed. Run: pip install fpdf2")
        return b""
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return b""


# ─────────────────────────────────────────────────────────────────────────────
# Claude AI Integration
# ─────────────────────────────────────────────────────────────────────────────

def _claude_available() -> bool:
    """Return True if Claude integration is properly configured."""
    return HAS_ANTHROPIC and bool(ANTHROPIC_API_KEY)


def _call_claude_with_fallback(client, create_params: dict) -> str:
    """
    Call Anthropic API with model fallbacks to handle cases where a specific model
    (e.g., claude-3-5-sonnet-20241022) is not available or disabled on the API key.
    """
    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-haiku-20241022",
        "claude-3-haiku-20240307"
    ]
    
    requested_model = create_params.get("model")
    if requested_model and requested_model in models:
        models.remove(requested_model)
        models.insert(0, requested_model)
    elif requested_model:
        models.insert(0, requested_model)

    last_err = None
    for model in models:
        try:
            params = create_params.copy()
            params["model"] = model
            message = client.messages.create(**params)
            return message.content[0].text.strip()
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # Fall back on 400 errors or model availability errors
            if "400" in err_str or "model" in err_str or "not_found" in err_str:
                logger.warning(f"Claude model {model} failed. Trying next fallback model. Error: {e}")
                continue
            else:
                # Re-raise authentication or billing errors immediately (e.g. 401, 403)
                raise e
                
    raise last_err


def _extract_claude_error_message(e: Exception) -> str:
    """Extract a user-friendly error message from an Anthropic API exception."""
    error_msg = str(e)
    if hasattr(e, "body") and isinstance(e.body, dict):
        api_err = e.body.get("error", {})
        if isinstance(api_err, dict) and api_err.get("message"):
            error_msg = api_err.get("message")
    return f"⚠️ Claude API error: {error_msg}"


def claude_credibility_brief(result: dict) -> str:
    """
    Generate a concise AI Credibility Brief using Claude.

    Parameters
    ----------
    result : dict — full analysis result

    Returns
    -------
    str — Claude's brief analysis text
    """
    if not _claude_available():
        return ""

    verdict = result.get("verdict", "UNKNOWN")
    truth_score = result.get("truth_score", 50)
    red_flags = result.get("red_flags", [])
    breakdown = result.get("credibility_breakdown", {})
    text_preview = result.get("text", "")[:600]

    prompt = f"""You are TruthLens AI, an expert fact-checker and media literacy assistant.

Article excerpt (first 600 chars):
\"\"\"{text_preview}\"\"\"

AI Analysis Summary:
- TruthScore™: {truth_score:.0f}/100
- Verdict: {verdict}
- Credibility signals: {json.dumps(breakdown, indent=2)}
- Red flags detected: {'; '.join(red_flags[:3]) if red_flags else 'None'}

Provide a concise Credibility Brief in exactly 3 short bullet points (each ≤ 20 words):
1. What is the overall credibility verdict and why?
2. The most important red flag or positive indicator.
3. A specific recommendation for the reader.

Format: Start each bullet with a relevant emoji. Be direct, factual, educational."""

    try:
        client = _anthropic_lib.Anthropic(api_key=ANTHROPIC_API_KEY)
        return _call_claude_with_fallback(client, {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}]
        })
    except Exception as e:
        logger.error(f"Claude brief failed: {e}")
        return _extract_claude_error_message(e)


def claude_ask_question(article_text: str, question: str, result: dict) -> str:
    """
    Answer a user question about the article using Claude.

    Parameters
    ----------
    article_text : str
    question : str
    result : dict — analysis result for context

    Returns
    -------
    str — Claude's answer
    """
    if not _claude_available():
        return ""

    truth_score = result.get("truth_score", 50)
    verdict = result.get("verdict", "UNKNOWN")
    text_preview = article_text[:800]

    system_prompt = (
        "You are TruthLens AI, a helpful media literacy and fact-checking assistant. "
        "You help students and readers evaluate news credibility. Be concise, educational, "
        "and evidence-based. Never make up facts."
    )

    user_prompt = f"""Article (excerpt):
\"\"\"{text_preview}\"\"\"

AI analysis: TruthScore™ = {truth_score:.0f}/100, Verdict = {verdict}

User question: {question}

Answer in 2-3 sentences. Be specific to this article. Start with a relevant emoji."""

    try:
        client = _anthropic_lib.Anthropic(api_key=ANTHROPIC_API_KEY)
        return _call_claude_with_fallback(client, {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 320,
            "messages": [{"role": "user", "content": user_prompt}],
            "system": system_prompt
        })
    except Exception as e:
        logger.error(f"Claude question failed: {e}")
        return _extract_claude_error_message(e)


def render_claude_panel(result: dict) -> None:
    """
    Render the Claude AI panel: AI Credibility Brief + Ask Claude Q&A.

    Parameters
    ----------
    result : dict — full analysis result
    """
    st.markdown("---")
    st.markdown(
        '<h3 style="font-family:\'Space Grotesk\'; font-weight:800; margin-bottom:4px;">'
        '&#129302; Claude AI Analysis <span class="ai-badge">&#9679; Powered by Anthropic</span>'
        '</h3>',
        unsafe_allow_html=True
    )
    st.caption("Claude reads the article and provides expert media-literacy insights.")

    if not _claude_available():
        st.markdown(clean_html("""
        <div class="claude-card" style="border-left-color:#FFB800;">
            <div style="color:#FFB800; font-weight:600; margin-bottom:6px;">&#9888;&#65039; Claude AI not configured</div>
            <div style="color:#8892A4; font-size:0.85rem; line-height:1.6;">
                To enable Claude AI features:<br>
                1. Get an API key at <code>console.anthropic.com</code><br>
                2. Add <code>ANTHROPIC_API_KEY=your_key</code> to <code>.env</code><br>
                3. Install: <code>pip install anthropic python-dotenv</code>
            </div>
        </div>
        """), unsafe_allow_html=True)
        return

    tab_brief, tab_ask = st.tabs(["&#129300; AI Credibility Brief", "&#128172; Ask Claude"])

    # Cache the generated brief for the current article analysis so repeated
    # reruns do not trigger an API call unless the user explicitly regenerates it.
    article_signature = (
        f"{result.get('verdict', 'UNKNOWN')}|"
        f"{result.get('truth_score', 50)}|"
        f"{result.get('text', '')[:200]}"
    )
    brief_key = "claude_brief_cache"
    brief_sig_key = "claude_brief_signature"

    with tab_brief:
        st.markdown(
            '<p style="color:#8892A4; font-size:0.85rem;">'
            'A concise 3-point summary of credibility, generated from the article and TruthLens signals.</p>',
            unsafe_allow_html=True
        )

        if (
            brief_key not in st.session_state
            or st.session_state.get(brief_sig_key) != article_signature
        ):
            with st.spinner("&#129302; Claude is writing your credibility brief..."):
                st.session_state[brief_key] = claude_credibility_brief(result)
                st.session_state[brief_sig_key] = article_signature

        brief = st.session_state.get(brief_key, "")
        if brief:
            st.markdown(
                f'<div class="claude-card">{brief}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div style="font-size:0.72rem; color:#8892A4; margin-top:10px; text-align:right;">'
                '&#129302; Generated by Claude Sonnet | TruthLens AI 2025</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("Claude did not return a brief for this article.")

        if st.button("&#128260; Regenerate Brief", key="regen_brief"):
            if brief_key in st.session_state:
                del st.session_state[brief_key]
            if brief_sig_key in st.session_state:
                del st.session_state[brief_sig_key]
            st.rerun()

    # ── Tab 2: Ask Claude ─────────────────────────────────────────────────────
    with tab_ask:
        st.markdown(
            '<p style="color:#8892A4; font-size:0.85rem;">'
            'Ask Claude any question about this article — fact-checking, bias, credibility, or media literacy.</p>',
            unsafe_allow_html=True
        )

        # Quick question chips
        quick_qs = [
            "What are the main red flags in this article?",
            "Is the language biased or emotional?",
            "What sources should I check to verify this?",
            "What makes this article credible or not?",
        ]
        st.markdown("**Quick questions:**")
        q_cols = st.columns(2)
        for idx, qtext in enumerate(quick_qs):
            with q_cols[idx % 2]:
                if st.button(qtext, key=f"quick_q_{idx}"):
                    st.session_state["claude_question_input"] = qtext

        question = st.text_input(
            "Your question:",
            value=st.session_state.get("claude_question_input", ""),
            placeholder="e.g. Is this article biased? What sources are cited?",
            key="claude_question_field"
        )

        if st.button("&#128172; Ask Claude", key="ask_claude_btn") and question.strip():
            with st.spinner("&#129302; Claude is thinking..."):
                answer = claude_ask_question(result.get("text", ""), question, result)

            st.markdown(
                f'<div class="claude-card">{answer}</div>',
                unsafe_allow_html=True
            )

            # Save Q&A to history
            if "claude_qa_history" not in st.session_state:
                st.session_state.claude_qa_history = []
            st.session_state.claude_qa_history.append({"q": question, "a": answer})

        # Show previous Q&A if any
        if st.session_state.get("claude_qa_history"):
            with st.expander("&#128196; Previous Questions", expanded=False):
                for qa in reversed(st.session_state.claude_qa_history[-5:]):
                    st.markdown(
                        f'<div style="margin-bottom:10px;">'
                        f'<div style="color:#8892A4; font-size:0.78rem; margin-bottom:4px;">'
                        f'&#10067; {qa["q"]}</div>'
                        f'<div class="claude-card" style="padding:10px 14px;">{qa["a"]}</div></div>',
                        unsafe_allow_html=True
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Media Literacy Tips
# ─────────────────────────────────────────────────────────────────────────────

def render_media_literacy_tips(truth_score: float, verdict: str) -> None:
    """
    Render context-sensitive media literacy tips based on the verdict.
    Helps students understand what to do next.
    """
    st.markdown(clean_html("""
    <div style="margin: 6px 0 14px 0;">
        <span style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                     font-size:0.88rem; color:#8B8AA8; letter-spacing:0.08em;
                     text-transform:uppercase;">📚 Media Literacy Guidance</span>
    </div>
    """), unsafe_allow_html=True)

    if truth_score < 30:
        tips = [
            ("🚩 Check named sources", "This article likely uses vague phrases like 'experts say' or 'sources claim' — always demand specific, named citations."),
            ("🔍 Search for this story", "Open Google News and search the headline. If only fringe sites cover it, treat it as unverified."),
            ("⚠️ Emotional language alert", "Misinformation uses fear and outrage to bypass critical thinking. Notice how this article made you feel."),
        ]
        tip_class = "ml-tip-fake"
        header_html = '<span style="color:#EF4444; font-weight:700;">🚨 Likely Fake — Here\'s what to watch for:</span>'
    elif truth_score <= 55:
        tips = [
            ("🔗 Find the primary source", "Click any linked 'study' or 'report' and read it directly — don't rely solely on this article's interpretation."),
            ("📅 Check the date", "Is this old news recycled as breaking? Always verify the original publication date."),
            ("🌐 Cross-reference 3+ sources", "Search Google News for this story. If credible outlets aren't covering it, exercise caution."),
        ]
        tip_class = "ml-tip-warn"
        header_html = '<span style="color:#F59E0B; font-weight:700;">⚠️ Suspicious — How to verify:</span>'
    else:
        tips = [
            ("✅ Good credibility signals found", "This article shows named sources, balanced language, and verifiable claims — positive signs of quality journalism."),
            ("📖 Still verify the main claim", "Even credible-looking articles can be wrong. Search the primary claim on Google Scholar or official sources."),
            ("🏆 Share responsibly", "If you share this, consider adding context so your audience understands the source's perspective."),
        ]
        tip_class = "ml-tip-real"
        header_html = '<span style="color:#10B981; font-weight:700;">✅ Appears Credible — Good signs:</span>'

    st.markdown(f"<div style='margin-bottom:8px;'>{header_html}</div>", unsafe_allow_html=True)
    for tip_title, tip_body in tips:
        st.markdown(
            f"<div class='{tip_class}'><strong>{tip_title}:</strong> {tip_body}</div>",
            unsafe_allow_html=True
        )

    # External fact-check links
    st.markdown(clean_html("""
    <div style="margin-top:16px; margin-bottom:6px;">
        <span style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                     font-size:0.82rem; color:#8B8AA8; letter-spacing:0.08em;
                     text-transform:uppercase;">🔗 Verify with trusted tools:</span>
    </div>
    <div style="display:flex; flex-wrap:wrap; gap:4px; margin-bottom:4px;">
        <a href="https://toolbox.google.com/factcheck/explorer" target="_blank" class="fc-link-btn">🔍 Google Fact Check</a>
        <a href="https://www.snopes.com" target="_blank" class="fc-link-btn">🕵️ Snopes</a>
        <a href="https://www.factcheck.org" target="_blank" class="fc-link-btn">✅ FactCheck.org</a>
        <a href="https://www.politifact.com" target="_blank" class="fc-link-btn">⚖️ PolitiFact</a>
        <a href="https://www.boomlive.in" target="_blank" class="fc-link-btn">🇮🇳 BOOM Live</a>
    </div>
    """), unsafe_allow_html=True)


def render_share_panel(result: dict) -> None:
    """Render a copyable share text panel for the analysis result."""
    score = result.get("truth_score", 0)
    verdict = result.get("verdict", "UNKNOWN")
    title = result.get("title", "Untitled Article")[:80]
    timestamp = result.get("timestamp", "")

    if score < 30:
        emoji = "🚨"
    elif score <= 55:
        emoji = "⚠️"
    else:
        emoji = "✅"

    share_text = (
        f"{emoji} TruthLens Analysis Result\n"
        f"Article: {title or 'No title provided'}\n"
        f"TruthScore™: {score:.0f}/100\n"
        f"Verdict: {verdict}\n"
        f"Analyzed: {timestamp}\n"
        f"—\n"
        f"Analyzed with TruthLens — AI Fake News Detector\n"
        f"Always verify with multiple trusted sources."
    )

    st.markdown(clean_html("""
    <div style="margin: 6px 0 10px 0;">
        <span style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                     font-size:0.88rem; color:#8B8AA8; letter-spacing:0.08em;
                     text-transform:uppercase;">📤 Share Your Findings</span>
    </div>
    """), unsafe_allow_html=True)

    st.code(share_text, language=None)
    st.caption("Copy the text above to share your fact-check result on social media or with classmates.")


# ─────────────────────────────────────────────────────────────────────────────
# Main Dashboard Render
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard(result: dict) -> None:
    """
    Render the full credibility dashboard after analysis.

    Parameters
    ----------
    result : dict — full analysis result from run_analysis()
    """
    st.markdown("---")
    st.markdown(clean_html("""
    <p style="color:#8892A4; font-size:0.82rem; margin-bottom:4px;">
    TruthScore&#8482; &bull; 9 Models &bull; XAI &bull; Cluster Map &bull; Claude AI
    </p>
    """), unsafe_allow_html=True)

    # ── Row 1: Gauge + Key Metrics ─────────────────────────────────────────────
    col_gauge, col_metrics = st.columns([1.05, 1.35], gap="large")

    with col_gauge:
        render_truth_score_gauge(
            result["truth_score"],
            result["verdict"],
            result["verdict_color"]
        )

        # Media literacy tips
        st.markdown("---")
        render_media_literacy_tips(result["truth_score"], result["verdict"])

    with col_metrics:
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3, gap="small")
        with m1:
            st.metric(
                "Confidence",
                f"{result['ensemble_confidence'] * 100:.1f}%",
                help="Ensemble model confidence in the predicted verdict"
            )
        with m2:
            st.metric(
                "Cosine Similarity",
                f"{result['cosine_similarity'] * 100:.1f}%",
                help="Similarity vs. credible source writing patterns"
            )
        with m3:
            anomaly = result["anomaly_score"]
            anomaly_delta = "normal" if anomaly < 0.5 else "anomalous"
            st.metric(
                "Anomaly Score",
                f"{anomaly * 100:.1f}%",
                delta=anomaly_delta,
                delta_color="inverse",
                help="Isolation Forest: how unusual are the writing patterns?"
            )

        m4, m5, m6 = st.columns(3, gap="small")
        with m4:
            st.metric("Word Count", result.get("word_count", 0))
        with m5:
            st.metric("Processing Time", f"{result['processing_time_ms']:.0f}ms")
        with m6:
            bert = result.get("distilbert_result")
            bert_val = f"{bert['score']*100:.0f}%" if bert else "N/A"
            st.metric("DistilBERT", bert_val, help="DistilBERT prediction confidence")

        # Red flags
        st.markdown("**🚩 Key Findings:**")
        key_findings = result.get("red_flags", [])[:3]
        if key_findings:
            for flag in key_findings:
                st.markdown(
                    f"<div class='red-flag'>{flag}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No major red flags were detected for this article.")

    # ── Row 2: Algorithm Comparison + Radar ────────────────────────────────────
    st.markdown("---")
    col_bar, col_radar = st.columns([1, 1], gap="large")

    with col_bar:
        render_algorithm_comparison(result.get("model_predictions", {}))

    with col_radar:
        render_radar_chart(result.get("radar_scores", {}))

    # ── Row 3: Credibility Breakdown + Word Cloud ──────────────────────────────
    st.markdown("---")
    col_break, col_wc = st.columns([1, 1], gap="large")

    with col_break:
        st.markdown("**📊 Credibility Signal Breakdown**",
                    help="5 individual credibility signals — higher is better (except Clickbait)")
        render_credibility_breakdown(result.get("credibility_breakdown", {}))

    with col_wc:
        st.markdown("**☁️ Top Red-Flag Word Cloud**",
                    help="Words most commonly associated with misinformation")
        render_wordcloud(result.get("redflag_words", {}))

    # ── Row 4: LIME Explainability ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🧠 Explainability Panel (XAI)**")
    st.caption("Which words influenced the prediction? Why did the AI say this?")
    render_lime_explanation(result.get("lime_weights", []), result.get("text", ""))

    # ── Row 4b: Claude AI Panel ────────────────────────────────────────────────
    render_claude_panel(result)

    # ── Row 5: Cluster Explorer ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🗺️ Article Cluster Explorer**")
    st.caption(
        "Your article plotted alongside 500 historical articles (blue = real clusters, red = fake clusters)"
    )

    pipeline = load_pipeline()
    cluster_data = pipeline.get_cluster_data() if pipeline else {}
    render_cluster_explorer(cluster_data, result.get("cluster_point", {}))

    # ── Row 5b: Advanced Insights ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🔬 Advanced Linguistic Insights**")
    adv_col1, adv_col2, adv_col3 = st.columns(3, gap="large")

    with adv_col1:
        flesch = result.get("flesch_score", 50)
        reading_level = result.get("reading_level", "Standard")
        color = "#FF4444" if flesch < 40 else "#FFB800" if flesch < 65 else "#00C851"
        st.markdown(clean_html(f"""
        <div class="neo-card" style="padding:18px; height:100%;">
            <div style="font-family:'Space Grotesk'; font-weight:700; color:#00F0FF;
                        font-size:0.88rem; margin-bottom:10px;">📚 Reading Level</div>
            <div style="font-family:'Orbitron'; font-size:1.8rem; font-weight:700;
                        color:{color}; text-align:center;">{flesch:.0f}</div>
            <div style="text-align:center; color:#8892A4; font-size:0.78rem;">{reading_level}</div>
            <div style="font-size:0.72rem; color:#8892A4; margin-top:8px;">
                Fake news is often written at a lower reading level to maximize emotional impact and shareability.
            </div>
        </div>
        """), unsafe_allow_html=True)

    with adv_col2:
        emotional_score = result.get("emotional_score", 0)
        fear  = result.get("fear_score", 0)
        anger = result.get("anger_score", 0)
        disgust = result.get("disgust_score", 0)
        em_color = "#FF4444" if emotional_score > 60 else "#FFB800" if emotional_score > 30 else "#00C851"
        st.markdown(clean_html(f"""
        <div class="neo-card" style="padding:18px; height:100%;">
            <div style="font-family:'Space Grotesk'; font-weight:700; color:#FF6B6B;
                        font-size:0.88rem; margin-bottom:10px;">😡 Emotional Manipulation</div>
            <div style="font-family:'Orbitron'; font-size:1.8rem; font-weight:700;
                        color:{em_color}; text-align:center;">{emotional_score:.0f}%</div>
            <div style="margin-top:8px; font-size:0.78rem; color:#8892A4;">
                😨 Fear: <b style="color:#FF8080;">{fear:.0f}%</b> &nbsp;
                😠 Anger: <b style="color:#FFB800;">{anger:.0f}%</b> &nbsp;
                🤢 Disgust: <b style="color:#C084FC;">{disgust:.0f}%</b>
            </div>
            <div style="font-size:0.72rem; color:#8892A4; margin-top:8px;">
            High emotional triggers (fear, anger) are classic signs of manipulative content designed to bypass critical thinking.
            </div>
        </div>
        """), unsafe_allow_html=True)

    with adv_col3:
        entities = result.get("named_entities", [])
        if entities:
            entity_html = " ".join(
                f'<span style="display:inline-block; background:rgba(155,48,255,0.12); '
                f'border:1px solid rgba(155,48,255,0.25); border-radius:6px; '
                f'padding:3px 8px; font-size:0.72rem; color:#C084FC; margin:2px;">{e}</span>'
                for e in entities[:10]
            )
            st.markdown(
                clean_html(f"""
                <div class="neo-card" style="padding:18px; height:100%;">
                    <div style="font-family:'Space Grotesk'; font-weight:700; color:#9B30FF;
                                font-size:0.88rem; margin-bottom:10px;">🏷️ Named Entities</div>
                    <div style="min-height:74px;">{entity_html}</div>
                    <div style="font-size:0.72rem; color:#8892A4; margin-top:8px;">
                        People, organizations, and places mentioned. Verify these exist and are quoted accurately.
                    </div>
                </div>
                """),
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                clean_html("""
                <div class="neo-card" style="padding:18px; height:100%;">
                    <div style="font-family:'Space Grotesk'; font-weight:700; color:#9B30FF;
                                font-size:0.88rem; margin-bottom:10px;">🏷️ Named Entities</div>
                    <div style="color:#8892A4; font-size:0.82rem;">No named entities detected.</div>
                </div>
                """),
                unsafe_allow_html=True
            )

    # ── Row 6: Student Learning Module ────────────────────────────────────────
    st.markdown("---")
    render_learning_module()

    # ── PDF Export ─────────────────────────────────────────────────────────────
    st.markdown("---")
    col_export, col_share = st.columns([1, 2])
    with col_export:
        with st.spinner("Preparing report..."):
            pdf_bytes = generate_pdf_report(result)
        if pdf_bytes:
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"truthlens_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                help="Download a complete analysis report as PDF",
            )
    with col_share:
        st.markdown(
            "<div style='color:#8892A4; font-size:0.85rem; padding-top:8px;'>"
            "TruthLens — AI Fake News Detector | Built for Media Literacy &amp; Student Fact-Checking | "
            "<code>streamlit run app.py</code>"
            "</div>",
            unsafe_allow_html=True
        )

    # Share panel
    st.markdown("---")
    render_share_panel(result)


# ─────────────────────────────────────────────────────────────────────────────
# Gamification Update
# ─────────────────────────────────────────────────────────────────────────────

def update_gamification(truth_score: float) -> None:
    """
    Update TruthHunter points based on analysis results.

    Parameters
    ----------
    truth_score : float — detected truth score
    """
    points_earned = 10  # base points per analysis
    if truth_score < 30:
        # Bonus for catching high-confidence fake
        points_earned += 15
        st.toast("🏆 Fake Caught! +25 TruthHunter Points!", icon="🚨")
    else:
        st.toast(f"+{points_earned} TruthHunter Points!", icon="⭐")

    st.session_state.truth_hunter_points += points_earned
    st.session_state.analyses_count += 1
    save_profile(
        st.session_state.truth_hunter_points,
        st.session_state.analyses_count,
        st.session_state.analysis_history
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main App Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main Streamlit application entry point."""
    inject_css()
    init_session_state()

    # ── LUEIO-Style Hero Banner ────────────────────────────────────────────────
    st.markdown(clean_html("""
    <div class="hero-section" style="text-align:left; padding:64px 0 32px 0; position:relative; z-index:2;">
    <h1 class="lueio-title">
    Detecting Misinformation<br>
    with <span class="grad">AI Precision</span>
    </h1>
    <p class="lueio-subtitle">
    TruthLens analyzes articles using <b style="color:#C084FC;">9 ML models</b> +
    <b style="color:#06B6D4;">Claude AI</b> to deliver an explainable credibility
    verdict in seconds — built for <b style="color:#F1F0FF;">students, educators &amp;
    fact-checkers</b>.
    </p>
    <p style="font-family:'Space Grotesk',sans-serif; font-size:0.78rem;
    color:rgba(139,138,168,0.7); letter-spacing:0.15em; text-transform:uppercase;
    margin-bottom:20px;">
    Media Literacy · Fact-Checking · Misinformation Detection
    </p>
    <div class="hscan"></div>
    <div style="display:flex; justify-content:flex-start; gap:10px; flex-wrap:wrap;">
    <span class="hpill fa1">🤖 AI Detection</span>
    <span class="hpill fa2">🛡️ Source Verification</span>
    <span class="hpill fa3">🔀 Explainable AI</span>
    <span class="hpill fa4">🎥 Deepfake Detection</span>
    <span class="hpill fa5">⚡ Real-Time Analysis</span>
    </div>
    </div>
    <hr>
    """), unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    # ── Input Panel ────────────────────────────────────────────────────────────
    article_text, title_text, input_source = render_input_panel()

    # ── Analyze Button ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, col_hint = st.columns([1, 3])
    with col_btn:
        analyze_clicked = st.button(
            "🔍 Analyze Article",

            help="Run all local analysis models on the article",
            disabled=st.session_state.get("analysis_in_progress", False),
        )
    with col_hint:
        st.markdown(
            "<div style='color:#8892A4; font-size:0.85rem; padding-top:10px;'>"
            "Paste an article, enter a URL, or upload images, then click Analyze.</div>",
            unsafe_allow_html=True
        )

    # ── Run Analysis ───────────────────────────────────────────────────────────
    status_placeholder = st.empty()
    results_placeholder = st.empty()

    st.session_state.pop("_auto_analyze", None)
    if analyze_clicked:
        if not article_text or len(article_text.split()) < 10:
            st.warning(
                "⚠️ Please provide at least 10 words of article text for a meaningful analysis.",
                icon="⚠️"
            )
        else:
            st.session_state.analysis_in_progress = True
            st.session_state.current_result = None
            results_placeholder.empty()
            status_placeholder.markdown(clean_html("""
                <div class="analyzing-container">
                    <div class="loading-ring"></div>
                    <div class="spinner-text">&#129504; Analyzing Article...</div>
                    <div class="loading-stage">Running 9 AI models &bull; extracting features &bull; computing TruthScore™</div>
                    <div class="loading-bar-track"><div class="loading-bar-fill"></div></div>
                </div>
                """), unsafe_allow_html=True)
            try:
                result = run_analysis(article_text, title_text)
            finally:
                status_placeholder.empty()
                st.session_state.analysis_in_progress = False

            if result:
                st.session_state.current_result = result
                # Add to history
                st.session_state.analysis_history.append({
                    "truth_score": result["truth_score"],
                    "verdict": result["verdict"],
                    "text_preview": article_text[:80],
                    "timestamp": result["timestamp"],
                })
                # Update gamification
                update_gamification(result["truth_score"])

    # ── Show Dashboard ─────────────────────────────────────────────────────────
    if st.session_state.current_result:
        with results_placeholder.container():
            render_dashboard(st.session_state.current_result)
    else:
        # Welcome state — show feature overview
        with results_placeholder.container():
            _render_welcome()

    render_sidebar()


def _render_welcome() -> None:
    """LUEIO-style welcome — feature cards in a clean dark grid."""
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(clean_html("""
    <div style="text-align:center; padding:16px 0 28px 0; z-index:2; position:relative;">
        <h2 style="font-family:'Syne',sans-serif; font-weight:800; font-size:2rem;
                   color:#F1F0FF; margin-bottom:10px;">
            What TruthLens Gives You
        </h2>
        <p style="color:#8B8AA8; font-size:0.92rem; max-width:520px; margin:0 auto; line-height:1.65;">
            Paste any article, enter a URL, or upload a screenshot — let
            <b style="color:#C084FC;">9 AI models</b> + <b style="color:#06B6D4;">Claude AI</b>
            analyze its credibility instantly.
        </p>
    </div>
    """), unsafe_allow_html=True)

    features = [
        ("📊", "TruthScore™ Gauge",     "0–100 animated credibility score with 3D depth visualization"),
        ("🤖", "9 ML Models",           "LR, RF, PAC, SVC, Ensemble, BERT, K-Means, Isolation Forest, CosSim"),
        ("✨", "Claude AI",             "Expert credibility brief & interactive Ask-Claude Q&A panel"),
        ("🔍", "XAI Explainability",    "LIME word-level highlights — understand exactly why AI flagged content"),
        ("🗺️", "Cluster Explorer",      "See where your article lands in a 3D news landscape map"),
        ("🏆", "TruthHunter Badges",    "Earn points & badges for every article you fact-check"),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(clean_html(f"""
            <div class="lcard" style="text-align:left; margin-bottom:18px;">
                <div class="lcard-icon">{icon}</div>
                <div class="lcard-title">{title}</div>
                <div class="lcard-desc">{desc}</div>
            </div>
            """), unsafe_allow_html=True)

    st.session_state.pop("_sample_type", None)
    return

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(clean_html("""
    <div style="text-align:center; margin-bottom:10px;">
        <span style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                     font-size:0.88rem; color:#8B8AA8; letter-spacing:0.08em;
                     text-transform:uppercase;">Try a sample article</span>
    </div>
    """), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📰 Sample: Likely REAL Article", key="btn_sample_real"):
            st.session_state["_sample_type"] = "real"
    with col2:
        if st.button("🚨 Sample: Likely FAKE Article", key="btn_sample_fake"):
            st.session_state["_sample_type"] = "fake"

    # ── Annotated sample article display ──────────────────────────────────────
    sample_type = st.session_state.get("_sample_type")

    if sample_type == "real":
        st.markdown(clean_html("""
        <div style="margin-top:22px; border-radius:16px; overflow:hidden;
                    border:1px solid rgba(16,185,129,0.35);
                    background:rgba(16,185,129,0.05);">
            <div style="background:rgba(16,185,129,0.18); padding:12px 20px;
                        display:flex; align-items:center; gap:10px;
                        border-bottom:1px solid rgba(16,185,129,0.30);">
                <span style="font-size:1.2rem;">&#10003;</span>
                <span style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                             color:#6EE7B7; font-size:0.95rem; letter-spacing:0.03em;">
                    LIKELY CREDIBLE ARTICLE &mdash; Sample with credibility annotations
                </span>
            </div>
            <div style="padding:20px 24px; font-family:'Inter',sans-serif;
                        font-size:0.95rem; line-height:1.85; color:#D1D0F0;">
                <div style="font-size:1.05rem; font-weight:700; color:#F1F0FF; margin-bottom:14px;">
                    &#128240; Nature Study Confirms Exercise Reduces Cardiovascular Disease Risk
                </div>
                A
                <span style="background:rgba(16,185,129,0.22); color:#6EE7B7;
                             padding:1px 6px; border-radius:4px; font-weight:600;
                             border-bottom:2px solid rgba(16,185,129,0.60);">
                    peer-reviewed study published in the journal <em>Nature</em>
                </span>
                confirms that regular physical exercise significantly reduces the risk of cardiovascular disease.
                <span style="background:rgba(16,185,129,0.22); color:#6EE7B7;
                             padding:1px 6px; border-radius:4px; font-weight:600;
                             border-bottom:2px solid rgba(16,185,129,0.60);">
                    Researchers at Harvard Medical School
                </span>
                analyzed data from
                <span style="background:rgba(6,182,212,0.18); color:#67E8F9;
                             padding:1px 6px; border-radius:4px; font-weight:600;
                             border-bottom:2px solid rgba(6,182,212,0.55);">
                    50,000 participants over 10 years
                </span>.
                According to
                <span style="background:rgba(16,185,129,0.22); color:#6EE7B7;
                             padding:1px 6px; border-radius:4px; font-weight:600;
                             border-bottom:2px solid rgba(16,185,129,0.60);">
                    WHO guidelines
                </span>,
                adults should engage in at least
                <span style="background:rgba(6,182,212,0.18); color:#67E8F9;
                             padding:1px 6px; border-radius:4px; font-weight:600;
                             border-bottom:2px solid rgba(6,182,212,0.55);">
                    150 minutes of moderate aerobic activity per week
                </span>.
                The study has been
                <span style="background:rgba(16,185,129,0.22); color:#6EE7B7;
                             padding:1px 6px; border-radius:4px; font-weight:600;
                             border-bottom:2px solid rgba(16,185,129,0.60);">
                    corroborated by independent research teams at Oxford and Stanford
                </span>.
            </div>
            <div style="padding:12px 24px 18px; border-top:1px solid rgba(16,185,129,0.20);
                        display:flex; flex-wrap:wrap; gap:12px; align-items:center;">
                <span style="font-family:'Space Grotesk',sans-serif; font-size:0.75rem;
                             color:#8B8AA8; font-weight:600; letter-spacing:0.05em;
                             text-transform:uppercase;">Key:</span>
                <span style="background:rgba(16,185,129,0.22); color:#6EE7B7;
                             padding:3px 10px; border-radius:6px; font-size:0.78rem;
                             font-family:'Inter',sans-serif; font-weight:600;">
                    &#10003; Named Source / Institution
                </span>
                <span style="background:rgba(6,182,212,0.18); color:#67E8F9;
                             padding:3px 10px; border-radius:6px; font-size:0.78rem;
                             font-family:'Inter',sans-serif; font-weight:600;">
                    &#128202; Specific Statistic
                </span>
            </div>
            <div style="padding:0 24px 18px; display:flex; flex-wrap:wrap; gap:8px;">
                <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.30);
                            border-radius:8px; padding:7px 14px; font-size:0.80rem;
                            font-family:'Inter',sans-serif; color:#6EE7B7;">
                    &#10003; Cites peer-reviewed journal
                </div>
                <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.30);
                            border-radius:8px; padding:7px 14px; font-size:0.80rem;
                            font-family:'Inter',sans-serif; color:#6EE7B7;">
                    &#10003; Named, verifiable institutions
                </div>
                <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.30);
                            border-radius:8px; padding:7px 14px; font-size:0.80rem;
                            font-family:'Inter',sans-serif; color:#6EE7B7;">
                    &#10003; Specific data &amp; sample size
                </div>
                <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.30);
                            border-radius:8px; padding:7px 14px; font-size:0.80rem;
                            font-family:'Inter',sans-serif; color:#6EE7B7;">
                    &#10003; Independent corroboration
                </div>
                <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.30);
                            border-radius:8px; padding:7px 14px; font-size:0.80rem;
                            font-family:'Inter',sans-serif; color:#6EE7B7;">
                    &#10003; Calm, factual tone
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    elif sample_type == "fake":
        st.markdown(clean_html("""
        <div style="margin-top:22px; border-radius:16px; overflow:hidden;
                    border:1px solid rgba(239,68,68,0.40);
                    background:rgba(239,68,68,0.04);">
            <div style="background:rgba(239,68,68,0.18); padding:12px 20px;
                        display:flex; align-items:center; gap:10px;
                        border-bottom:1px solid rgba(239,68,68,0.30);">
                <span style="font-size:1.2rem;">&#128680;</span>
                <span style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                             color:#FCA5A5; font-size:0.95rem; letter-spacing:0.03em;">
                    LIKELY MISINFORMATION &mdash; Sample with red-flag annotations
                </span>
            </div>
            <div style="padding:20px 24px; font-family:'Inter',sans-serif;
                        font-size:0.95rem; line-height:1.85; color:#D1D0F0;">
                <div style="font-size:1.05rem; font-weight:700; color:#FCA5A5; margin-bottom:14px;">
                    &#128680; SHOCKING: Miracle Cure Exposed and Hiding from Public!
                </div>
                <span style="background:rgba(239,68,68,0.22); color:#FCA5A5;
                             padding:1px 6px; border-radius:4px; font-weight:700;
                             border-bottom:2px solid rgba(239,68,68,0.60);">
                    SHOCKING BOMBSHELL
                </span>
                <span style="background:rgba(239,68,68,0.10); color:#F87171;
                             padding:1px 5px; border-radius:3px; font-size:0.72rem;
                             font-weight:700; vertical-align:middle; margin-left:2px;">
                    &#128681; ALL CAPS clickbait
                </span>: Scientists
                <span style="background:rgba(239,68,68,0.22); color:#FCA5A5;
                             padding:1px 6px; border-radius:4px; font-weight:700;
                             border-bottom:2px solid rgba(239,68,68,0.60);">
                    EXPOSED
                </span>
                for hiding miracle cure that
                <span style="background:rgba(245,158,11,0.22); color:#FCD34D;
                             padding:1px 6px; border-radius:4px; font-weight:600;
                             border-bottom:2px solid rgba(245,158,11,0.55);">
                    Big Pharma DOESN'T WANT YOU TO KNOW
                </span>
                <span style="background:rgba(245,158,11,0.10); color:#FBBF24;
                             padding:1px 5px; border-radius:3px; font-size:0.72rem;
                             font-weight:700; vertical-align:middle; margin-left:2px;">
                    &#9888; Conspiracy framing
                </span>!
                <span style="background:rgba(239,68,68,0.22); color:#FCA5A5;
                             padding:1px 6px; border-radius:4px; font-weight:700;
                             border-bottom:2px solid rgba(239,68,68,0.60);">
                    Leaked documents
                </span>
                <span style="background:rgba(239,68,68,0.10); color:#F87171;
                             padding:1px 5px; border-radius:3px; font-size:0.72rem;
                             font-weight:700; vertical-align:middle; margin-left:2px;">
                    &#128681; No verifiable source
                </span>
                reveal massive government cover-up that elites have been hiding for decades! You WON'T BELIEVE what
                <span style="background:rgba(239,68,68,0.22); color:#FCA5A5;
                             padding:1px 6px; border-radius:4px; font-weight:700;
                             border-bottom:2px solid rgba(239,68,68,0.60);">
                    anonymous whistleblowers
                </span>
                <span style="background:rgba(239,68,68,0.10); color:#F87171;
                             padding:1px 5px; border-radius:3px; font-size:0.72rem;
                             font-weight:700; vertical-align:middle; margin-left:2px;">
                    &#128681; Unnamed source
                </span>
                revealed about the
                <span style="background:rgba(245,158,11,0.22); color:#FCD34D;
                             padding:1px 6px; border-radius:4px; font-weight:600;
                             border-bottom:2px solid rgba(245,158,11,0.55);">
                    deep state conspiracy
                </span>
                <span style="background:rgba(245,158,11,0.10); color:#FBBF24;
                             padding:1px 5px; border-radius:3px; font-size:0.72rem;
                             font-weight:700; vertical-align:middle; margin-left:2px;">
                    &#9888; Conspiracy label
                </span>
                involving vaccines.
                <span style="background:rgba(239,68,68,0.22); color:#FCA5A5;
                             padding:1px 6px; border-radius:4px; font-weight:700;
                             border-bottom:2px solid rgba(239,68,68,0.60);">
                    SHARE before they delete this!
                </span>
                <span style="background:rgba(239,68,68,0.10); color:#F87171;
                             padding:1px 5px; border-radius:3px; font-size:0.72rem;
                             font-weight:700; vertical-align:middle; margin-left:2px;">
                    &#128681; Urgency manipulation
                </span>
                The mainstream media is
                <span style="background:rgba(239,68,68,0.22); color:#FCA5A5;
                             padding:1px 6px; border-radius:4px; font-weight:700;
                             border-bottom:2px solid rgba(239,68,68,0.60);">
                    LYING to you!
                </span>
                <span style="background:rgba(239,68,68,0.10); color:#F87171;
                             padding:1px 5px; border-radius:3px; font-size:0.72rem;
                             font-weight:700; vertical-align:middle; margin-left:2px;">
                    &#128681; Blanket accusation, no evidence
                </span>
            </div>
            <div style="padding:12px 24px 18px; border-top:1px solid rgba(239,68,68,0.20);
                        display:flex; flex-wrap:wrap; gap:12px; align-items:center;">
                <span style="font-family:'Space Grotesk',sans-serif; font-size:0.75rem;
                             color:#8B8AA8; font-weight:600; letter-spacing:0.05em;
                             text-transform:uppercase;">Key:</span>
                <span style="background:rgba(239,68,68,0.22); color:#FCA5A5;
                             padding:3px 10px; border-radius:6px; font-size:0.78rem;
                             font-family:'Inter',sans-serif; font-weight:600;">
                    &#128681; High-risk red flag
                </span>
                <span style="background:rgba(245,158,11,0.22); color:#FCD34D;
                             padding:3px 10px; border-radius:6px; font-size:0.78rem;
                             font-family:'Inter',sans-serif; font-weight:600;">
                    &#9888; Suspicious pattern
                </span>
            </div>
            <div style="padding:0 24px 18px; display:flex; flex-wrap:wrap; gap:8px;">
                <div style="background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.30);
                            border-radius:8px; padding:7px 14px; font-size:0.80rem;
                            font-family:'Inter',sans-serif; color:#FCA5A5;">
                    &#128681; Excessive ALL CAPS &amp; clickbait
                </div>
                <div style="background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.30);
                            border-radius:8px; padding:7px 14px; font-size:0.80rem;
                            font-family:'Inter',sans-serif; color:#FCA5A5;">
                    &#128681; Anonymous, unverifiable sources
                </div>
                <div style="background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.30);
                            border-radius:8px; padding:7px 14px; font-size:0.80rem;
                            font-family:'Inter',sans-serif; color:#FCA5A5;">
                    &#128681; Conspiracy framing
                </div>
                <div style="background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.30);
                            border-radius:8px; padding:7px 14px; font-size:0.80rem;
                            font-family:'Inter',sans-serif; color:#FCA5A5;">
                    &#128681; Urgency &amp; fear manipulation
                </div>
                <div style="background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.30);
                            border-radius:8px; padding:7px 14px; font-size:0.80rem;
                            font-family:'Inter',sans-serif; color:#FCA5A5;">
                    &#128681; Zero verifiable evidence
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
