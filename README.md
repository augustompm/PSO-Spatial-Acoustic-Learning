# PSO-Spatial-Acoustic-Learning

  HRTF prediction from 3D head meshes using machine learning.

  ## Overview

  This project implements a complete pipeline for predicting Head-Related Transfer Functions (HRTF) from 3D head
  models, achieving 89.8% R² accuracy using only geometric features extracted from meshes.

  ## Structure

  ├── 1-hrtf_prediction-V3.ipynb          # Baseline HRTF prediction with full anthropometric data
  ├── 2-ridge_hrtf_prediction_V2.1.ipynb  # Automatic feature extraction from 3D meshes
  ├── 3-photo_hrtf_prediction.ipynb      # Validation with photogrammetric reconstruction
  ├── data/                               # HUTUBS dataset (meshes, SOFA files, measurements)*
  ├── blender/                            # 3D reconstruction files
  ├── documentation/                      # Articles
  ├── article-images/                     # Additional visualizations
  └── images/                             # Logo (not used)

*aditional HUTUBS PLY (3D models) data avaliable at [https://sofacoustics.org/data/database/hutubs/]([URL](https://sofacoustics.org/data/database/hutubs/))

  ## Key Results

  - **Baseline**: R² = 89.8% with complete anthropometric measurements
  - **Mesh extraction**: R² = 89.3% with automatic feature extraction (18.9% measurement error)
  - **Photogrammetry**: R² = 69.6% validating real-world applicability

  ## Requirements

  - Python 3.12+
  - NumPy, pandas, scikit-learn, trimesh
  - HUTUBS dataset at /data read LICENCE for details

  ## License

  MIT License

  Copyright (c) 2024 PSO-Spatial-Acoustic-Learning team: 
  
  Abraão, Augusto, Jan e Kedson Prof. Dra. Mariza Ferro

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
