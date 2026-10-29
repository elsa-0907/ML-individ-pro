@echo off
title Fake News Detector Web App
cd /d "%~dp0"
echo Starting Fake News Detector on http://localhost:8501 ...
"C:\Users\elsar\AppData\Local\Programs\Python\Python314\python.exe" -m streamlit run app.py --server.port 8501
pause
