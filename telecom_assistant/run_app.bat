@echo off
echo ============================================================
echo   TELECOM SERVICE ASSISTANT - STREAMLIT APPLICATION
echo ============================================================
echo.
echo Starting the Streamlit application...
echo.
echo The application will open in your default web browser.
echo Press Ctrl+C to stop the server.
echo.
echo ============================================================
echo.

cd /d "%~dp0"
streamlit run ui/streamlit_app.py

pause
