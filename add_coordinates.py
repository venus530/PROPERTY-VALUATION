import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

# Define Harare bounding box (approximate)
HARARE_BOUNDS = {
    'min_lat': -17.95,
    'max_lat': -17.65,
    'min_lon': 30.95,
    'max_lon': 31.25
}

def is_valid_coordinate(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
        return (HARARE_BOUNDS['min_lat'] <= lat <= HARARE_BOUNDS['max_lat'] and
                HARARE_BOUNDS['min_lon'] <= lon <= HARARE_BOUNDS['max_lon'])
    except Exception:
        return False

def main():
    df = pd.read_csv('PROPERTYVALUATIONS.csv')
    geolocator = Nominatim(user_agent="property_valuation_geocoder")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    updated = 0
    for idx, row in df.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        if not is_valid_coordinate(lat, lon):
            # Build a location string for better accuracy
            location_query = f"{row['Location']}, Harare, Zimbabwe"
            try:
                location = geocode(location_query)
                if location:
                    df.at[idx, 'Latitude'] = location.latitude
                    df.at[idx, 'Longitude'] = location.longitude
                    updated += 1
                    print(f"Updated row {idx}: {location_query} -> ({location.latitude}, {location.longitude})")
                else:
                    print(f"Could not geocode: {location_query}")
            except Exception as e:
                print(f"Error geocoding {location_query}: {e}")
            time.sleep(1)  # Be nice to the API

    if updated > 0:
        df.to_csv('PROPERTYVALUATIONS_geocoded.csv', index=False)
        print(f"\nUpdated {updated} rows. Saved to PROPERTYVALUATIONS_geocoded.csv.")
    else:
        print("No updates were made. All coordinates appear valid.")

if __name__ == "__main__":
    main() 