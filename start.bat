@echo off
echo =============================================
echo   FactGuard AI — Starting Server
echo =============================================
echo.

if not exist ".env" (
    echo ERROR: .env file not found!
    echo Please copy .env.example to .env and add your API keys:
    echo   GOOGLE_API_KEY=your_key
    echo   TAVILY_API_KEY=your_key
    echo.
    pause
    exit /b 1
)

echo Starting FastAPI server on http://localhost:8000
echo Press Ctrl+C to stop.
echo.
python api_server.py
