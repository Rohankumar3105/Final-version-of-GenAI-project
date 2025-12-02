# Main entry point to run the Streamlit app
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Import and run the Streamlit UI
from ui.streamlit_app import main

if __name__ == "__main__":
    main()
