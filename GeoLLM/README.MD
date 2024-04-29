# GeoLLM

This guide outlines how to use GeoLLM approach for cross-view image retrieval and geolocalisation.

## Overview

There are three main steps:
1. **Description Generation:** Use the `VQA_to_Description.py` to generate descriptions for both aerial and ground-level images for your chosen data (e.g., CVUK, VIGOR, etc.).
2. **Description-Based Retrieval and Reranking:** The `VIGOR_rerank.py` is a reranking method that uses cross-view image descriptions to rerank the top-5 retrieval of TransGeo.
3. **Geographical and Semantic Image Retrieval:** Use `llm_retrieval.py` to evaluate the performance of GeoLLM approach for the descriptions generated in Step 1.

## Step-by-Step Guide

### Step 1: Generate Descriptions
First, generate descriptions for your dataset (e.g., VIGOR, CVUK, or your custom dataset) using the `VQA_to_Description.py` script. This will process aerial and ground-level images to create descriptive language outputs.

**Usage:**
```bash
python VQA_to_Description.py --image_folder <path_to_images> --final_output_csv <path_to_output_csv>
```
