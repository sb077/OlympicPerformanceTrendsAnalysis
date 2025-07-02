#!/usr/bin/env python3
"""
Setup script for Olympic Performance Trends Analysis
This script initializes the database and ensures all required files are present.
"""

import os
import sys
from dbcrud import create_tables, upload_data_in_db
from preprocess import preprocess_data

def check_files():
    """Check if all required CSV files are present."""
    csv_dir = os.path.join(os.path.dirname(__file__), "csv_files")
    required_files = [
        'Olympic_Athlete_Event_Details.csv',
        'Olympic_Event_Results.csv',
        'Olympic_Athlete_Biography.csv',
        'Olympic_Medal_Tally_History.csv',
        'Olympic_Games_Summary.csv',
        'population_total_long.csv',
        'Olympic_Country_Profiles.csv'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(csv_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required CSV files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required CSV files found")
    return True

def setup_database():
    """Initialize the database with tables and data."""
    try:
        print("Creating database tables...")
        create_tables()
        print("✅ Database tables created successfully")
        
        print("Uploading CSV data to database...")
        upload_data_in_db()
        print("✅ CSV data uploaded successfully")
        
        print("Running preprocessing...")
        preprocess_data()
        print("✅ Preprocessing completed successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        return False

def main():
    """Main setup function."""
    print("🚀 Olympic Performance Trends Analysis - Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Please run this script from the app directory")
        sys.exit(1)
    
    # Check required files
    if not check_files():
        print("\n❌ Setup failed: Missing required files")
        sys.exit(1)
    
    # Setup database
    if setup_database():
        print("\n✅ Setup completed successfully!")
        print("\n🎉 Your app is ready to run!")
        print("Run: streamlit run app.py")
    else:
        print("\n❌ Setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 