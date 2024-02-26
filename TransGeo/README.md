# Evaluating TrasnGeo performance on CVUK dataset

This guide provides instructions on how to use the `feature_extractor.py` and `test.py` scripts for feature extraction and accuracy testing with the TransGeo model.

## Prerequisites

- Python 3.6 or newer
- Required Python packages: `numpy`, `pandas`, `torch`, `torchvision`, `scipy`
- The TransGeo model weights placed in the "weights" folder

## Setup

1. **Install Python Packages**: Ensure that you have the required Python packages installed. You can install these packages using `pip`:

    ```bash
    pip install numpy pandas torch torchvision scipy
    ```

2. **Model Weights**: Place the TransGeo model weights file into a folder named `weights` in the `TransGeo` folder. The expected path should be `./weights/your_model_weights.pth`.

## Feature Extraction

The `feature_extractor.py` script extracts features from a specified dataset using the TransGeo model. **It is necessary to run this script once for the query dataset and once for the satellite dataset before proceeding to accuracy testing.**

Run the script from the terminal, specifying the mode (`--query` or `--satellite`), the directory for the dataset, and the path where extracted features and filenames will be saved.

```bash
python feature_extractor.py --mode MODE --dataset_dir PATH_TO_DATASET --features_path PATH_TO_SAVE_FEATURES --filenames_path PATH_TO_SAVE_FILENAMES
```

## Testing

After extracting features for both query and satellite datasets, you can evaluate the performance of TransGeo using the `test.py` script. This script calculates the top-K accuracy between the query and reference (satellite) datasets based on the extracted features.

Run the `test.py` script from the terminal, specifying the paths to the reference and query features and filenames, as well as the values for K (number of top features to consider) and the distance threshold for matching.

```bash
python test.py --ref_features_path PATH_TO_REF_FEATURES --query_features_path PATH_TO_QUERY_FEATURES --ref_filenames_path PATH_TO_REF_FILENAMES --query_filenames_path PATH_TO_QUERY_FILENAMES --K VALUE --distance_threshold VALUE
```
