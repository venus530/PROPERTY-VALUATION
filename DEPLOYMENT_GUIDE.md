# Harare Property Valuation Application - Deployment Guide

## 🚀 Deployment Options

### Option 1: Local Development (Current Setup)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Access at: http://localhost:5000
```

### Option 2: Heroku Deployment (Recommended for Dissertation)

#### Step 1: Prepare for Heroku
```bash
# Install Heroku CLI
# Download from: https://devcenter.heroku.com/articles/heroku-cli

# Login to Heroku
heroku login

# Create Heroku app
heroku create your-property-valuation-app

# Add buildpack for Python
heroku buildpacks:set heroku/python
```

#### Step 2: Create Procfile
Create a file named `Procfile` (no extension) in your project root:
```
web: gunicorn app:app
```

#### Step 3: Update requirements.txt
Add gunicorn to your requirements:
```
Flask==2.3.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
geopy==2.3.0
numpy==1.24.3
gunicorn==21.2.0
```

#### Step 4: Deploy to Heroku
```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial commit"

# Deploy to Heroku
git push heroku main

# Open the app
heroku open
```

### Option 3: Python Anywhere (Free Hosting)

#### Step 1: Sign up at PythonAnywhere
- Go to www.pythonanywhere.com
- Create a free account

#### Step 2: Upload your files
- Use the Files tab to upload your project files
- Or use git clone if your code is in a repository

#### Step 3: Set up a web app
- Go to Web tab
- Click "Add a new web app"
- Choose "Flask" and Python 3.9
- Set the source code directory to your project folder
- Set the WSGI configuration file to point to your app.py

#### Step 4: Install dependencies
```bash
pip install --user -r requirements.txt
```

### Option 4: AWS EC2 (Professional Deployment)

#### Step 1: Launch EC2 Instance
- Launch Ubuntu 20.04 LTS instance
- Configure security groups (allow HTTP, HTTPS, SSH)

#### Step 2: Connect and Setup
```bash
# Connect via SSH
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3-pip python3-venv nginx -y

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
pip install gunicorn
```

#### Step 3: Configure Nginx
Create `/etc/nginx/sites-available/property-valuation`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Step 4: Run with Gunicorn
```bash
# Create systemd service
sudo nano /etc/systemd/system/property-valuation.service
```

Add this content:
```ini
[Unit]
Description=Property Valuation Gunicorn daemon
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/property_valuation_project
Environment="PATH=/home/ubuntu/property_valuation_project/venv/bin"
ExecStart=/home/ubuntu/property_valuation_project/venv/bin/gunicorn --workers 3 --bind 127.0.0.1:8000 app:app

[Install]
WantedBy=multi-user.target
```

#### Step 5: Start Services
```bash
sudo systemctl start property-valuation
sudo systemctl enable property-valuation
sudo ln -s /etc/nginx/sites-available/property-valuation /etc/nginx/sites-enabled
sudo systemctl restart nginx
```

## 📊 Dashboard and Spatial Analysis Setup

### Dashboard Features
- Interactive charts using Chart.js
- Real-time property statistics
- Property value distribution analysis
- Location-based insights

### Spatial Analysis Features
- Interactive map using Leaflet.js
- Property value hotspots
- Location-based clustering
- Spatial statistics

## 🔧 Configuration

### Environment Variables
Create a `.env` file for production:
```env
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
DEBUG=False
```

### Model Files
Ensure your trained model files are in the `models/` directory:
- `best_model_gradient_boosting.joblib`
- `preprocessor.joblib`
- `model_metadata.json`

## 📈 Performance Optimization

### For Production
1. **Use Gunicorn** instead of Flask development server
2. **Enable caching** for API responses
3. **Compress static files**
4. **Use CDN** for external libraries
5. **Database optimization** for large datasets

### Monitoring
- Set up logging
- Monitor response times
- Track user interactions
- Set up error alerts

## 🛡️ Security Considerations

1. **HTTPS**: Always use HTTPS in production
2. **Input Validation**: Validate all user inputs
3. **Rate Limiting**: Prevent abuse
4. **CORS**: Configure Cross-Origin Resource Sharing
5. **Secrets Management**: Use environment variables for sensitive data

## 📝 Dissertation Integration

### For Your Dissertation:
1. **Document the deployment process**
2. **Include performance metrics**
3. **Show user interface screenshots**
4. **Demonstrate real-world usage**
5. **Include scalability considerations**

### Presentation Tips:
- Show live demo during presentation
- Prepare backup screenshots
- Have deployment statistics ready
- Demonstrate all three objectives (web app, ML model, spatial analysis)

## 🆘 Troubleshooting

### Common Issues:
1. **Port already in use**: Change port in app.py
2. **Model not found**: Ensure models/ directory exists
3. **Geolocation not working**: Check HTTPS requirement
4. **Charts not loading**: Check internet connection for CDN

### Support:
- Check Flask logs for errors
- Verify all dependencies are installed
- Ensure file permissions are correct
- Test locally before deploying

---

**Your application is now ready for dissertation presentation and real-world deployment!** 🎉

# Deployment Guide: Render + GitHub

This guide will walk you through deploying your Harare Property Valuation System to Render using GitHub.

## 🚀 Quick Start (Recommended)

### Step 1: Prepare Your Repository

1. **Initialize Git** (if not already done)
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Create GitHub Repository**
   - Go to [GitHub.com](https://github.com)
   - Click "New repository"
   - Name it: `harare-property-valuation`
   - Make it public (for free Render deployment)
   - Don't initialize with README (we already have one)

3. **Push to GitHub**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/harare-property-valuation.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Render

1. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with your GitHub account

2. **Create New Web Service**
   - Click "New +" button
   - Select "Web Service"
   - Connect your GitHub repository

3. **Configure the Service**
   - **Name**: `harare-property-valuation`
   - **Environment**: `Python`
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free (for testing)

4. **Advanced Settings** (Optional)
   - **Health Check Path**: `/`
   - **Auto-Deploy**: Enabled
   - **Environment Variables**: None required for basic deployment

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - Wait for the build to complete (usually 2-5 minutes)

### Step 3: Verify Deployment

1. **Check Build Logs**
   - Monitor the build process in Render dashboard
   - Ensure all dependencies install successfully

2. **Test Your Application**
   - Visit your Render URL: `https://your-app-name.onrender.com`
   - Test all features:
     - Home page
     - Property prediction
     - Dashboard
     - Spatial analysis

## 🔧 Configuration Files

### render.yaml (Auto-deployment)
```yaml
services:
  - type: web
    name: harare-property-valuation
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
    healthCheckPath: /
    autoDeploy: true
```

### requirements.txt
```
Flask==2.3.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
gunicorn==21.2.0
Werkzeug==2.3.7
geopy==2.3.0
numpy==1.24.3
```

### .gitignore
```
# Python
__pycache__/
*.py[cod]
*.so
.Python
build/
dist/
*.egg-info/

# Virtual environments
venv/
env/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Flask
instance/
.webassets-cache

# Logs
*.log
```

## 🛠️ Troubleshooting

### Common Issues

1. **Build Fails**
   - Check requirements.txt for correct versions
   - Ensure all dependencies are listed
   - Verify Python version compatibility

2. **Application Won't Start**
   - Check start command: `gunicorn app:app`
   - Verify app.py has the correct Flask app instance
   - Check logs for specific error messages

3. **Static Files Not Loading**
   - Ensure templates/ directory is included
   - Check file paths in HTML templates
   - Verify Bootstrap and other CDN links

4. **Model Files Missing**
   - Ensure models/ directory is committed to Git
   - Check file paths in app.py
   - Verify model loading code

### Debug Steps

1. **Check Render Logs**
   ```bash
   # In Render dashboard, go to your service
   # Click on "Logs" tab
   # Look for error messages
   ```

2. **Test Locally First**
   ```bash
   pip install -r requirements.txt
   python app.py
   # Ensure it works locally before deploying
   ```

3. **Verify File Structure**
   ```bash
   # Ensure all necessary files are present
   ls -la
   # Should include: app.py, requirements.txt, templates/, models/
   ```

## 🔄 Continuous Deployment

### Automatic Updates
- Every push to the `main` branch triggers a new deployment
- Render automatically rebuilds and deploys your application
- No manual intervention required

### Manual Deployment
```bash
# Make changes to your code
git add .
git commit -m "Update feature"
git push origin main
# Render will automatically deploy the changes
```

## 📊 Monitoring

### Render Dashboard
- **Build Status**: Monitor build success/failure
- **Deployment History**: Track all deployments
- **Logs**: View application logs
- **Metrics**: Monitor performance (paid plans)

### Health Checks
- Render automatically checks your application health
- Health check path: `/` (home page)
- Failed health checks trigger alerts

## 🔒 Security Considerations

### Environment Variables
```bash
# For sensitive data, use Render environment variables
# In Render dashboard: Environment → Environment Variables
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-secret-key
```

### HTTPS
- Render automatically provides HTTPS
- All traffic is encrypted
- No additional configuration needed

## 📈 Scaling

### Free Tier Limitations
- **Build Time**: 500 minutes/month
- **Runtime**: 750 hours/month
- **Sleep**: App sleeps after 15 minutes of inactivity
- **Bandwidth**: 100GB/month

### Paid Plans
- **Starter**: $7/month - No sleep, more resources
- **Standard**: $25/month - Better performance
- **Pro**: $50/month - High performance

## 🎯 Best Practices

### Code Organization
- Keep your code clean and well-documented
- Use meaningful commit messages
- Test locally before pushing

### Performance
- Optimize your ML model loading
- Use efficient data structures
- Minimize external API calls

### Maintenance
- Regularly update dependencies
- Monitor application logs
- Keep your GitHub repository organized

## 📞 Support

### Render Support
- [Render Documentation](https://render.com/docs)
- [Render Community](https://community.render.com)
- [Render Status](https://status.render.com)

### GitHub Support
- [GitHub Guides](https://guides.github.com)
- [GitHub Community](https://github.community)

## 🎉 Success Checklist

- [ ] Repository created on GitHub
- [ ] Code pushed to GitHub
- [ ] Render account created
- [ ] Web service configured
- [ ] Application deployed successfully
- [ ] All features working
- [ ] Health checks passing
- [ ] HTTPS working
- [ ] Auto-deployment enabled

---

**Congratulations!** Your Harare Property Valuation System is now live on the internet and ready for your dissertation presentation! 🚀 