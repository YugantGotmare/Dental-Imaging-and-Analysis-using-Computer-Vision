# Dental Imaging and Analysis using Computer Vision

## Overview
Teeth and Nerve Segmentation is a web-based application built using Streamlit that helps users segment teeth and nerve structures in medical images. The application allows users to upload images and apply pre-trained machine learning models for segmentation.

## Features
- **ML-Based Image Segmentation**: Uses pre-trained models to segment teeth and nerve structures.
- **Image Upload & Processing**: Users can upload images in various formats for analysis.
- **Confidence Control**: Allows users to adjust model confidence for segmentation results.
- **Detection Results**: Displays segmented images and provides detailed analysis.
- **Expandable Results View**: Users can explore detection details with bounding box data.

## Technologies Used
- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning Models**: Pre-trained segmentation models (YoloV8 Segmentation)
- **APIs & Libraries**:
  - PIL (for image processing)
  - pathlib (for handling file paths)
  - SQLite (for storing history)
  - dotenv (for environment management)

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/YugantGotmare/Dental-Imaging-and-Analysis-using-Computer-Vision.git
   cd /Dental-Imaging-and-Analysis-using-Computer-Vision
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   pip install -r requirements.txt
   ```
3. Set up the `.env` file with the required configurations:
   ```
   SEGMENTATION_MODEL=path_to_segmentation_model
   NERVE_MODEL=path_to_nerve_model
   ```
4. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage
- Open the application in the browser.
- Select the task: Teeth segmentation or Nerve segmentation.
- Adjust model confidence using the slider.
- Upload an image for analysis.
- Click on "Segment" to apply the model.
- View, analyze, and expand detection results.

## Contribution
Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request.

## License
This project is licensed under the MIT License.

