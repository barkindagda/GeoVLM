## Updates

- **April 28, 2024**: We are currently working on making the CVUK dataset publicly available.


# GeoLLM: Evaluating Autonomous Vehicle Geolocalisation through Cross-View Language
![GeoLLM Pipeline](pipeline.png)

## Overview
GeoLLM introduces an innovative approach to cross-view geo-localisation (CVGL), addressing the challenge of geo-positioning autonomous vehicles by correlating ground-level images with geo-tagged satellite imagery. Unlike traditional methods reliant on complex domain projections and low-level geometric features, GeoLLM harnesses the power of zero-shot vision language models (VLMs) to generate interpretable scene descriptions. These descriptions bridge the gap between views, enhancing generalisability across different geographic regions and offering resilience against the variability of real-world driving conditions.

## Key Contributions
- **Interpretable Cross-View Language Descriptions**: Our method bypasses the need for geometric projections by using natural language processing to describe key features of a scene, similar to how a human might interpret a view.
- **Robustness to Environmental Dynamics**: By leveraging VLMs, GeoLLM provides consistent performance despite changes in lighting, weather, and urban activity.
- **CVUK Dataset**: We introduce a comprehensive dataset encompassing diverse environmental conditions, including day-night cycles and seasonal changes, across major UK cities, to better assess CVGL systems.

### CVUK Dataset
The CVUK dataset comprises ~8 hours of driving footage across Liverpool, London, and Woking, UK, captured using a dashboard-mounted stereo camera to ensure a wide field-of-view. Recorded across August 2023, December 2023, and January 2024, the dataset encapsulates a variety of seasonal, environmental, and lighting conditions by including day-to-night transitions. Alongside the footage, we provide GPS coordinates and corresponding aerial views obtained via the Google Maps Static API, with a spatial alignment to the ground footage despite potential GPS positioning errors up to 10 meters.

<p align="center">
  <img src="query_gif.gif" width="400" />
  <img src="satellite_gif.gif" width="400" />
</p>

## Research Paper
For detailed insights and the methodology of GeoLLM, please refer to the paper
