import numpy as np
import pandas as pd
import math
import scipy.spatial
import argparse

# Setup argparse
parser = argparse.ArgumentParser(description='GeoImage Matching Script')
parser.add_argument('--ref_features_path', type=str, required=True, help='Path to the reference features .npy file')
parser.add_argument('--query_features_path', type=str, required=True, help='Path to the query features .npy file')
parser.add_argument('--ref_filenames_path', type=str, required=True, help='Path to the reference filenames .csv file')
parser.add_argument('--query_filenames_path', type=str, required=True, help='Path to the query filenames .csv file')
parser.add_argument('--k', type=int, default=1, help='Number of top features to consider')
parser.add_argument('--distance_threshold', type=float, default=0.0, help='Distance threshold for matching')
args = parser.parse_args()


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
def find_top_k_features(query_feature, ref_features, k):
    if len(query_feature.shape) != 1:
        query_feature = query_feature.flatten()
    similarities = [1 - scipy.spatial.distance.cosine(query_feature, ref_feature)
                    for ref_feature in ref_features]
    top_k_indices = np.argsort(similarities)[-k:]
    return top_k_indices

def calculate_top_k_accuracy(query_features, query_filenames_df, ref_features, ref_filenames_df, k, distance_threshold):
    correct_in_top_k_count = 0
    for i, query_feature in enumerate(query_features):
        top_k_indices = find_top_k_features(query_feature, ref_features, k)
        query_filename = query_filenames_df.iloc[i]['filename']
        query_lat, query_lon = extract_gps_from_filename(query_filename)

        for idx in top_k_indices:
            ref_filename = ref_filenames_df.iloc[idx]['filename']
            ref_lat, ref_lon = extract_gps_from_filename(ref_filename)
            distance = haversine(query_lon, query_lat, ref_lon, ref_lat)
            if distance <= distance_threshold:
                correct_in_top_k_count += 1
                break

    accuracy = (correct_in_top_k_count / len(query_features)) * 100
    return accuracy

# Usage
top_k_accuracy = calculate_top_k_accuracy(query_features, query_filenames_df, ref_features, ref_filenames_df, args.K, args.distance_threshold)
print(f"Top-{args.K} Accuracy: {top_k_accuracy}%")
