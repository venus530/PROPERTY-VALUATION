@echo off
echo 🚀 Harare Property Valuation System - Deployment Script
echo ======================================================

REM Check if Git is initialized
if not exist ".git" (
    echo ❌ Git repository not found. Please run 'git init' first.
    pause
    exit /b 1
)

REM Check current status
echo 📊 Current Git Status:
git status

echo.
echo 📋 Next Steps for Deployment:
echo =============================
echo.
echo 1. 🐙 Create GitHub Repository:
echo    - Go to https://github.com
echo    - Click 'New repository'
echo    - Name: harare-property-valuation
echo    - Make it PUBLIC (required for free Render deployment)
echo    - Don't initialize with README (we already have one)
echo.
echo 2. 🔗 Connect to GitHub:
echo    git remote add origin https://github.com/YOUR_USERNAME/harare-property-valuation.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo 3. 🌐 Deploy on Render:
echo    - Go to https://render.com
echo    - Sign up with GitHub account
echo    - Click 'New +' → 'Web Service'
echo    - Connect your GitHub repository
echo    - Configure:
echo      * Name: harare-property-valuation
echo      * Environment: Python
echo      * Build Command: pip install -r requirements.txt
echo      * Start Command: gunicorn app:app
echo      * Plan: Free
echo.
echo 4. ✅ Verify Deployment:
echo    - Wait for build to complete (2-5 minutes)
echo    - Visit your Render URL
echo    - Test all features
echo.
echo 📁 Files Ready for Deployment:
echo ==============================
echo ✅ app.py - Flask application
echo ✅ requirements.txt - Python dependencies
echo ✅ render.yaml - Render configuration
echo ✅ templates/ - HTML templates
echo ✅ models/ - Trained ML models
echo ✅ .gitignore - Git ignore rules
echo ✅ README.md - Project documentation
echo.
echo 🎯 Your application will be available at:
echo    https://your-app-name.onrender.com
echo.
echo 📚 For detailed instructions, see DEPLOYMENT_GUIDE.md
echo.
echo Good luck with your dissertation! 🎓
pause 