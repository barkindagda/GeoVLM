import argparse
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from tqdm import tqdm
import time

def load_descriptions(csv_path):
    return pd.read_csv(csv_path, index_col='Filename')['Description'].to_dict()

def embed_descriptions(descriptions, embedder_model):
    embedder = SentenceTransformer(embedder_model)
    embeddings = embedder.encode(list(descriptions.values()), convert_to_tensor=True)
    return embeddings, list(descriptions.keys())

def find_most_similar_index(query_embedding, satellite_embeddings):
    cos_scores = util.cos_sim(query_embedding, satellite_embeddings)[0]
    top_result_index = torch.argmax(cos_scores).item()
    return top_result_index

def calculate_accuracy(main_csv_path, query_descriptions_csv_path, satellite_descriptions_csv_path, embedder_model):
    main_df = pd.read_csv(main_csv_path)
    query_descriptions = load_descriptions(query_descriptions_csv_path)
    satellite_descriptions = load_descriptions(satellite_descriptions_csv_path)
    embedder = SentenceTransformer(embedder_model)

    correct_matches = 0
    processed_queries = 0
    fully_processed_queries = 0
    query_times = []

    for idx, row in tqdm(main_df.iterrows(), total=main_df.shape[0], desc="Processing Query Images"):
        start_time = time.time()

        if row['Query Image'] not in query_descriptions or not all(fn in satellite_descriptions for fn in [row['Top 1'], row['Top 2'], row['Top 3'], row['Top 4'], row['Top 5']]):
            continue

        query_description = query_descriptions[row['Query Image']]
        satellite_filenames = [row['Top 1'], row['Top 2'], row['Top 3'], row['Top 4'], row['Top 5']]
        fully_processed_queries += 1

        query_embedding = embedder.encode(query_description, convert_to_tensor=True)
        satellite_descs = {fn: satellite_descriptions[fn] for fn in satellite_filenames}
        satellite_embeddings, ordered_filenames = embed_descriptions(satellite_descs, embedder_model)

        most_similar_index = find_most_similar_index(query_embedding, satellite_embeddings)
        most_similar_filename = ordered_filenames[most_similar_index]

        processed_queries += 1
        if most_similar_filename == row['GT']:
            correct_matches += 1

        end_time = time.time()
        query_time = end_time - start_time
        query_times.append(query_time)

    average_time = sum(query_times) / len(query_times) if query_times else 0
    accuracy = (correct_matches / processed_queries) * 100 if processed_queries > 0 else 0
    return accuracy, average_time, fully_processed_queries

def main():
    parser = argparse.ArgumentParser(description="Calculate image matching accuracy based on descriptions.")
    parser.add_argument("--main_csv_path", type=str, required=True, help="Path to the main CSV file containing image matches.")
    parser.add_argument("--query_descriptions_csv_path", type=str, required=True, help="Path to the CSV file with query image descriptions.")
    parser.add_argument("--satellite_descriptions_csv_path", type=str, required=True, help="Path to the CSV file with satellite image descriptions.")
    parser.add_argument("--embedder_model", type=str, default="all-MiniLM-L6-v2", help="Model used for embedding descriptions.")

    args = parser.parse_args()

    accuracy, average_time, fully_processed_queries = calculate_accuracy(
        args.main_csv_path,
        args.query_descriptions_csv_path,
        args.satellite_descriptions_csv_path,
        args.embedder_model
    )

    print(f"Accuracy: {accuracy}%")
    print(f"Average processing time per query: {average_time:.2f} seconds.")
    print(f"Fully processed queries: {fully_processed_queries}")

if __name__ == "__main__":
    main()
