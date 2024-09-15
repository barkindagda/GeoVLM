

# GeoVLM: Improving Autonomous Vehicle Geolocalisation using Vision-Language Model Reranking
![GeoVLM Pipeline](geovlm_pipeline.png)

## Updates

- **September, 2024**: GeoVLM released.

## Overview
GeoVLM is a novel vision-language based reranking approach for cross-view geolocalisation. It improves the top-match accuracy of SOTA approaches.

## Key Contributions
- **Interpretable Cross-View Geolocalisation**: The first ex-
plainable reranking approach used for cross-view geolocalisation
- **Improvement on SOTA performance**:The GeoVLM improves the SOTA performance by reranking top-10 cross-view images. It uses human level of visual reasoning. 
- **CVUK Dataset**: We introduce a comprehensive dataset encompassing diverse environmental conditions, including day-night cycles and seasonal changes, across major UK cities, to better assess CVGL systems.

### CVUK Dataset
The CVUK dataset comprises ~8 hours of driving footage across Liverpool, London, and Woking, UK, captured using a dashboard-mounted stereo camera to ensure a wide field-of-view. Recorded across August 2023, December 2023, and January 2024, the dataset encapsulates a variety of seasonal, environmental, and lighting conditions by including day-to-night transitions. Alongside the footage, we provide GPS coordinates and corresponding aerial views obtained via the Google Maps Static API, with a spatial alignment to the ground footage despite potential GPS positioning errors up to 10 meters.

<p align="center">
  <img src="query_gif.gif" width="400" />
  <img src="satellite_gif.gif" width="400" />
</p>

## Research Paper
For detailed insights and the methodology of GeoVLM, please refer to the paper
