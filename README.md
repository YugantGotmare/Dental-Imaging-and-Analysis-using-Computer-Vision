# 🦷 Dental Imaging Analysis System

## 📋 Overview
The **Dental Imaging Analysis System** is an advanced web-based application built with Streamlit that leverages AI and computer vision for comprehensive dental image analysis. The system provides three specialized analysis modes: teeth segmentation, nerve segmentation, and caries detection, enabling dental professionals and researchers to perform detailed diagnostic assessments.

## ✨ Features

### 🎯 Three Analysis Modes
1. **Teeth Segmentation** - Identifies and segments individual teeth structures using YOLOv8
2. **Nerve Segmentation** - Detects and maps dental nerve pathways
3. **Caries Detection** - Identifies cavities, crowns, and fillings using VGG16-UNet architecture

### 🛠️ Core Capabilities
- **Multiple Model Support**: Seamlessly switch between YOLO-based and U-Net based models
- **Image Upload & Processing**: Support for JPG, PNG, BMP, and WEBP formats
- **Adjustable Confidence Threshold**: Fine-tune detection sensitivity (25-100%)
- **Dual Visualization Modes**: Toggle between segmentation masks only or full detection with bounding boxes
- **Real-time Analysis**: Instant processing and visualization of results
- **Detailed Statistics**: Comprehensive breakdown of detected regions with pixel counts and percentages
- **Interactive Results**: Expandable sections for in-depth analysis
- **Professional UI**: Modern medical-themed interface with gradient styling

### 📊 Advanced Caries Detection
- Multi-class segmentation (Caries, Crown, Filling, Normal tissue)
- Percentage-based pathology coverage calculation
- Color-coded severity indicators
- Detailed pixel-level analysis
- Visual overlay with legend

## 🖼️ User Interface
- **Clean Layout**: Two-column design for original and analyzed images
- **Sidebar Configuration**: Easy access to all settings and controls
- **Status Indicators**: Real-time feedback on model loading and processing
- **Medical Theme**: Professional blue-teal gradient color scheme
- **Responsive Design**: Adapts to different screen sizes

## 🔧 Technologies Used

### Frontend
- **Streamlit** - Web application framework
- **Custom CSS** - Modern medical-themed styling

### Backend & Processing
- **Python 3.8+** - Core programming language
- **YOLOv8** - Teeth and nerve segmentation
- **TensorFlow/Keras** - Caries detection model
- **OpenCV** - Image processing and manipulation

### Libraries & APIs
- **PIL (Pillow)** - Image loading and preprocessing
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization and plotting
- **Pathlib** - File path handling
- **python-dotenv** - Environment management

## 📁 Project Structure
```
Dental-Analysis-System/
│
├── Notebooks/             # Jupyter notebooks for experiments
├── __pycache__/          # Python cache files
├── images/               # Input images directory
├── output/               # Output results directory
├── weights/              # Model weights directory
│   ├── best_seg_200ep_YOLOv8l.pt    # Teeth segmentation model
│   ├── best_nerve.pt                 # Nerve segmentation model
│   └── vgg16_unet_model.h5          # Caries detection model
│
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
├── app.py                # Main application file
├── data.yaml             # Dataset configuration
├── helper.py             # Helper functions for YOLO models
├── packages.txt          # System packages
├── requirements.txt      # Python dependencies
├── settings.py           # Configuration and paths
└── Python.gitignore      # Python-specific gitignore
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YugantGotmare/Dental-Imaging-and-Analysis-using-Computer-Vision.git
   cd Dental-Imaging-and-Analysis-using-Computer-Vision
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure model paths:**
   
   Update `settings.py` with your model paths:
   ```python
   SEGMENTATION_MODEL = MODEL_DIR / 'best_seg_200ep_YOLOv8l.pt'
   NERVE_MODEL = MODEL_DIR / 'best_nerve.pt'
   ```
   
   Update `app.py` for caries model path (line ~280):
   ```python
   caries_model_path = r'path/to/your/vgg16_unet_model.h5'
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

6. **Access the application:**
   - Open your browser
   - Navigate to `http://localhost:8501`

## 📖 Usage Guide

### Basic Workflow

1. **Select Analysis Type**
   - Choose from: Teeth segmentation, Nerve segmentation, or Caries Detection
   - View model information in the sidebar

2. **Adjust Settings**
   - Set confidence threshold (25-100%)
   - For teeth/nerve: Toggle bounding box display on/off

3. **Upload Image**
   - Click "Browse files" or drag and drop
   - Supported formats: JPG, JPEG, PNG, BMP, WEBP

4. **Analyze**
   - Click "🔬 Analyze Image" button
   - Wait for processing (typically 2-5 seconds)

5. **Review Results**
   - View segmented/detected regions
   - Check detailed statistics (for caries detection)
   - Expand result sections for more information

### Caries Detection Results
- **Metric Cards**: Shows percentage of each pathology type
- **Status Badge**: Indicates severity (Minimal/Moderate/Significant)
- **Total Pathology**: Combined coverage of all detected issues
- **Detailed Breakdown**: Pixel-level statistics for each region

### Visualization Modes (Teeth/Nerve)
- **Masks Only** (Default): Shows colored segmentation regions without boxes
- **Bounding Boxes**: Displays detection boxes with labels and confidence scores

## 🎨 Model Information

### 1. Teeth Segmentation
- **Model**: YOLOv8-Large
- **Type**: Instance Segmentation
- **Input**: RGB Images
- **Output**: Segmented teeth with masks and boxes

### 2. Nerve Segmentation
- **Model**: YOLOv8 (Custom trained)
- **Type**: Instance Segmentation
- **Input**: RGB Images
- **Output**: Detected nerve pathways

### 3. Caries Detection
- **Model**: VGG16-UNet
- **Type**: Semantic Segmentation
- **Classes**: 4 (Normal, Caries, Crown, Filling)
- **Input**: 256x256 RGB Images
- **Output**: Multi-class segmentation mask

## 🔒 Important Disclaimers

⚠️ **Medical Disclaimer**: This AI analysis tool is designed for educational and research purposes only. It should **NOT** be used as a substitute for professional dental diagnosis or treatment recommendations. Always consult with qualified dental professionals for medical advice and treatment decisions.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

## 🐛 Known Issues & Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Verify model paths in `settings.py` and `app.py`
   - Ensure model files exist in the `weights/` directory

2. **Import Errors**
   - Run `pip install -r requirements.txt` again
   - Check Python version compatibility (3.8+)

3. **Image Upload Issues**
   - Verify file format is supported
   - Check file size (recommended < 10MB)

4. **Slow Processing**
   - Large images take longer to process
   - Consider using GPU for faster inference

## 📊 Performance Notes

- **Processing Time**: 2-5 seconds per image (CPU)
- **GPU Acceleration**: Significantly faster with CUDA-enabled GPU
- **Memory Usage**: ~2-4GB RAM depending on model

## 🔮 Future Enhancements

- [ ] Batch processing for multiple images
- [ ] Export results to PDF reports
- [ ] Integration with DICOM format
- [ ] Real-time video stream analysis
- [ ] Cloud deployment option
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Model comparison feature

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors & Acknowledgments

- **Yugant Gotmare** - Initial work and development

## 📞 Contact & Support

For questions, suggestions, or issues:
- GitHub Issues: [Create an issue](https://github.com/YugantGotmare/Dental-Imaging-and-Analysis-using-Computer-Vision/issues)
- Email: yugantgotmare123@gmail.com
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/yugantgotmare/)

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐!

---

**Note**: This system is continuously being improved. Check back for updates and new features!

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [OpenCV Documentation](https://docs.opencv.org/)

---