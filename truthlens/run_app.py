import sys
import asyncio

# Set the selector event loop policy on Windows to prevent WinError 64 crashes.
# This prevents the asyncio loop from raising a fatal OSError when a client closes
# a connection abruptly.
if sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Shadow torchvision to prevent RuntimeError: operator torchvision::nms does not exist
# caused by a mismatch between pre-installed torch and torchvision packages.
sys.modules['torchvision'] = None

import streamlit.web.bootstrap as bootstrap

if __name__ == "__main__":
    flag_options = {
        "server.port": 8501,
        "server.headless": True,
        "server.enableCORS": True,
        "server.enableXsrfProtection": True
    }
    
    # Run streamlit bootstrap
    bootstrap.run('app.py', False, [], flag_options)
