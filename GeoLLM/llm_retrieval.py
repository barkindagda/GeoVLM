from sentence_transformers import SentenceTransformer, util
import torch
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, Markdown
import argparse

# Setup argparse for command line arguments
parser = argparse.ArgumentParser(description='Calculate accuracy based on most similar satellite descriptions.')
parser.add_argument('--query_csv_path', type=str, required=True, help='Path to your Query image descriptions CSV file.')
parser.add_argument('--satellite_csv_path', type=str, required=True, help='Path to your Satellite image descriptions CSV file.')
parser.add_argument('--top_k', type=int, default=1, help='Number of top K similar descriptions to consider.')
parser.add_argument('--distance_threshold', type=float, default=0.0, help='Distance threshold for matching in kilometers.')
args = parser.parse_args()


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees).
    """
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r
  
def find_most_similar_satellite_descriptions(query_description, satellite_csv_path, top_k):

    satellite_descriptions = pd.read_csv(satellite_csv_path, index_col='Filename')['Description'].to_dict()
    embedder = SentenceTransformer('all-MiniLM-L6-v2',device='cuda')
    corpus_embeddings = embedder.encode(list(satellite_descriptions.values()), convert_to_tensor=True)
    query_embedding = embedder.encode(query_description, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    return [list(satellite_descriptions.keys())[i] for i in top_results.indices]

def calculate_accuracy(query_csv_path, satellite_csv_path, top_k, distance_threshold):
    query_descriptions = pd.read_csv(query_csv_path, index_col='Filename')['Description'].to_dict()
    correct_matches = 0
    total_queries = 0


    for query_file, query_description in query_descriptions.items():
        total_queries += 1

        # Correctly extract latitude and longitude from the query filename
        query_file_parts = query_file.replace("ground_", "").replace(".jpg", "").split("_")
        query_lat, query_lon = map(float, query_file_parts[:2])  # Assuming the first two parts are lat and lon

        top_satellite_files = find_most_similar_satellite_descriptions(query_description, satellite_csv_path, top_k)
        correct_match_found = False

        for satellite_file in top_satellite_files:
            # Correctly extract latitude and longitude from the satellite filename
            satellite_file_parts = satellite_file.replace("satellite_", "").replace(".png", "").split("_")
            sat_lat, sat_lon = map(float, satellite_file_parts[:2])  # Assuming the first two parts are lat and lon

            distance = haversine(query_lon, query_lat, sat_lon, sat_lat)
            if distance <= distance_threshold:
                correct_match_found = True
                break

        if correct_match_found:
            correct_matches += 1

    return (correct_matches / total_queries) if total_queries > 0 else 0


def main():
    accuracy = calculate_accuracy(query_csv_path=args.query_csv_path, satellite_csv_path=args.satellite_csv_path, top_k=args.top_k, distance_threshold=args.distance_threshold)
    print(f"The overall accuracy is {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
