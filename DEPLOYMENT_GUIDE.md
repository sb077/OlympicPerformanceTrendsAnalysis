# Olympic Performance Trends Analysis - Streamlit Deployment Guide

## Overview
This guide will help you deploy your Olympic Performance Trends Analysis app on Streamlit Cloud.

## Prerequisites
1. A GitHub account
2. Your code pushed to a GitHub repository
3. A Streamlit Cloud account (free at https://share.streamlit.io/)

## Deployment Steps

### 1. Prepare Your Repository
Make sure your repository structure looks like this:
```
OlympicPerformanceTrendsAnalysis/
├── app/
│   ├── app.py                 # Main Streamlit app
│   ├── requirements.txt       # Python dependencies
│   ├── .streamlit/
│   │   └── config.toml       # Streamlit configuration
│   ├── csv_files/            # Your CSV data files
│   ├── dbcrud.py
│   ├── hypotheses.py
│   ├── preprocess.py
│   └── streamlit_helper.py
└── README.md
```

### 2. Push to GitHub
```bash
git add .
git commit -m "Prepare for Streamlit deployment"
git push origin main
```

### 3. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Click "New app"
4. Fill in the details:
   - **Repository**: Select your repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `app/app.py`
   - **App URL**: Choose a unique URL (optional)
5. Click "Deploy!"

### 4. Configuration Details

#### App Path
- **Main file path**: `app/app.py`
- This tells Streamlit where your main application file is located

#### Requirements
- Streamlit will automatically install packages from `app/requirements.txt`
- All dependencies are pinned to specific versions for stability

#### Data Files
- Your CSV files in `app/csv_files/` will be available to the app
- The database will be created automatically when the app runs

## Post-Deployment

### First Run
1. After deployment, your app will be available at the provided URL
2. The first time you run the app, you'll need to:
   - Click "Upload" to populate the database with CSV data
   - Click "Run Preprocessing" to create processed tables
3. This may take a few minutes on the first run

### Using the App
1. **Upload Data**: Use the "Upload" button to load CSV files into the database
2. **View Tables**: Select tables from the dropdown to view and modify data
3. **Run Analysis**: Execute the four hypotheses to see Olympic trends analysis
4. **Preprocessing**: Run preprocessing steps to create analysis-ready tables

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all required packages are in `requirements.txt`
2. **File Not Found**: Check that CSV files are in the correct `csv_files/` directory
3. **Database Issues**: The app will create the database automatically on first run
4. **Memory Issues**: Large datasets might cause memory problems - consider data sampling

### Performance Tips
- The app uses SQLite for data storage
- Large CSV files may take time to upload initially
- Consider adding data caching for better performance

## Support
If you encounter issues:
1. Check the Streamlit Cloud logs in your app dashboard
2. Verify all file paths are correct
3. Ensure all dependencies are properly specified

## App Features
- **Data Upload**: Upload Olympic data from CSV files
- **Data Exploration**: View and modify raw and processed tables
- **Trend Analysis**: Four comprehensive hypotheses about Olympic trends
- **Interactive Visualizations**: Charts and graphs for data insights
- **Real-time Processing**: Dynamic data analysis and visualization 