# GeoLLM

This guide outlines how to use GeoLLM approach for cross-view image retrieval and geolocalisation.

## Overview

There are three main steps:
1. **Description Generation:** Use the `VQA_to_Description.py` to generate descriptions for both aerial and ground-level images for your chosen data (e.g., CVUK, VIGOR, etc.).
2. **Description-Based Retrieval and Reranking:** The `VIGOR_rerank.py` is a reranking method that uses cross-view image descriptions to rerank the top-5 retrieval of TransGeo.
3. **Geographical and Semantic Image Retrieval:** Use `llm_retrieval.py` to evaluate the performance of GeoLLM approach for the descriptions generated in Step 1.

### Step 1: Generate Descriptions
First, generate descriptions for your dataset (e.g., VIGOR, CVUK, or your custom dataset) using the `VQA_to_Description.py` script. This will process aerial and ground-level images to create descriptive language outputs.

```bash
python VQA_to_Description.py --image_folder <path_to_images> --final_output_csv <path_to_output_csv>
```
### Step 2: Generate Descriptions
Use the VIGOR_rerank.py to rerank the top-5 images retrieved by a model such as TransGeo. This step uses GeoLLM as a reranking method based on the descriptions generated in the previous step. This step requires the top 5 ranks by transgeo which can be found in data/VIGOR_top5.csv .

```bash
python VIGOR_rerank.py --main_csv_path <path_to_VIGOR_top5.csv> --query_csv_path <path_to_query_csv> --satellite_csv_path <path_to_satellite_csv> --top_k 5
```
### Step 3: Evaluate Retrieval and Geolocalisation Performance 
To run GeoLLM as an image retrieval model for geo-tagged cross-view images, run the `llm_retrieval.py` with the descriptions generated in Step 1.

```bash
python llm_retrieval.py --query_csv_path <path_to_query_csv> --satellite_csv_path <path_to_satellite_csv> --top_k <number_of_top_matches> --distance_threshold <max_distance_in_km>
```
