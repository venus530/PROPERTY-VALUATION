import pandas as pd
import numpy as np
import random

def add_coordinates_to_csv():
    """Add latitude and longitude coordinates to the property dataset."""
    
    # Load the existing CSV
    df = pd.read_csv('PROPERTYVALUATIONS.csv')
    
    # Define coordinate ranges for different areas in Harare
    # These are approximate coordinates for different suburbs in Harare
    location_coordinates = {
        'medium density': {
            'lat_range': (-17.85, -17.75),  # Harare latitude range
            'lon_range': (30.95, 31.15)     # Harare longitude range
        },
        'high density': {
            'lat_range': (-17.90, -17.80),
            'lon_range': (30.90, 31.10)
        },
        'low density': {
            'lat_range': (-17.80, -17.70),
            'lon_range': (31.00, 31.20)
        }
    }
    
    # Add coordinate columns
    latitudes = []
    longitudes = []
    
    for _, row in df.iterrows():
        location_type = row['Location']
        
        if location_type in location_coordinates:
            lat_range = location_coordinates[location_type]['lat_range']
            lon_range = location_coordinates[location_type]['lon_range']
            
            # Generate coordinates with some randomness within the range
            lat = random.uniform(lat_range[0], lat_range[1])
            lon = random.uniform(lon_range[0], lon_range[1])
        else:
            # Default coordinates for unknown locations
            lat = random.uniform(-17.85, -17.75)
            lon = random.uniform(30.95, 31.15)
        
        latitudes.append(lat)
        longitudes.append(lon)
    
    # Add the coordinate columns
    df['Latitude'] = latitudes
    df['Longitude'] = longitudes
    
    # Save the updated CSV
    df.to_csv('PROPERTYVALUATIONS.csv', index=False)
    
    print(f"✅ Added coordinates to {len(df)} properties")
    print(f"📊 Coordinate ranges:")
    print(f"   Latitude: {df['Latitude'].min():.4f} to {df['Latitude'].max():.4f}")
    print(f"   Longitude: {df['Longitude'].min():.4f} to {df['Longitude'].max():.4f}")
    
    return df

if __name__ == '__main__':
    add_coordinates_to_csv() 