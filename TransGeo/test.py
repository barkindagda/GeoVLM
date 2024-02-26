import numpy as np
import pandas as pd
import math
import scipy.spatial
K = 1 # Number of top features to consider
DISTANCE_THRESHOLD = 0.0 
ref_features_path='./'
query_features_path = './'
ref_filenames_path = './'
query_filenames_path = './'


ref_features = np.load(ref_features_path)
ref_filenames_df = pd.read_csv(ref_filenames_path)
query_features = np.load(query_features_path)
query_filenames_df = pd.read_csv(query_filenames_path)

# Helper functions
def haversine(lon1, lat1, lon2, lat2):
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers. Use 3956 for miles
    return c * r

def extract_gps_from_filename(filename):
    parts = filename.split('_')
    lat = float(parts[1])
    lon = float(parts[2].split('.')[0])  # Remove the file extension
    return lat, lon

# GeoImageMatcher class
class GeoImageMatcher:
    def __init__(self, ref_features, ref_filenames_df, query_features, query_filenames_df):
        self.reference_features = ref_features
        self.query_features = query_features
        self.reference_filenames_df = ref_filenames_df
        self.query_filenames_df = query_filenames_df

    def find_top_k_features(self, query_feature, k):
        if len(query_feature.shape) != 1:
            query_feature = query_feature.flatten()
        similarities = [1 - scipy.spatial.distance.cosine(query_feature, ref_feature)
                        for ref_feature in self.reference_features]
        top_k_indices = np.argsort(similarities)[-k:]
        return top_k_indices

    def calculate_top_k_accuracy(self, k, distance_threshold):
        correct_in_top_k_count = 0
        for i, query_feature in enumerate(self.query_features):
            top_k_indices = self.find_top_k_features(query_feature, k)
            query_filename = self.query_filenames_df.iloc[i]['filename']
            query_lat, query_lon = extract_gps_from_filename(query_filename)

            for idx in top_k_indices:
                ref_filename = self.reference_filenames_df.iloc[idx]['filename']
                ref_lat, ref_lon = extract_gps_from_filename(ref_filename)
                distance = haversine(query_lon, query_lat, ref_lon, ref_lat)
                if distance <= distance_threshold:
                    correct_in_top_k_count += 1
                    break

        accuracy = (correct_in_top_k_count / len(self.query_features)) * 100
        return accuracy

# Usage
matcher = GeoImageMatcher(ref_features, ref_filenames_df, query_features, query_filenames_df)
top_k_accuracy = matcher.calculate_top_k_accuracy(K, DISTANCE_THRESHOLD)
print(f"Top-{K} Accuracy: {top_k_accuracy}%")
