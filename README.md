# segmentation-MRI-curvefitting
A web-based application for MRI DICOM medical image analysis designed to process ZIP files containing DICOM datasets, perform automated ROI (Region of Interest) segmentation, and execute tri-exponential curve fitting to calculate T2 relaxation times.
Main Features
The application follows a structured three-step analytical pipeline:

1. **Interactive Viewer & Windowing**: 
   - Visualize DICOM slices with real-time **Window Level (WL)** and **Window Width (WW)** adjustments.
   - Automatically extracts **Echo Time (TE)** and **Repetition Time (TR)** parameters from ZIP filenames using regex patterns.

2. **Automated ROI Segmentation**: 
   - Detects material grids (Carbomer, PEG, Alginate, etc.) automatically using the **Hough Circle Transform**.
   - Calculates signal intensity across all slices and generates a summary table.
   - Export results directly to **Excel (.xlsx)** format.

3. **Tri-Exponential Curve Fitting**: 
   - Analyzes T2 decay signals using a **Tri-Exponential model**.
   - Compares models **with and without offset** to ensure high precision.
   - Provides statistical metrics including **R²** and **RMSE**, with estimations for $T2_1, T2_2,$ and $T2_3$.
