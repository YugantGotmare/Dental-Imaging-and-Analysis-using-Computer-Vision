# Python In-built packages
from pathlib import Path
import PIL
import numpy as np
import matplotlib.pyplot as plt
import io

# External packages
import streamlit as st
from tensorflow.keras.models import load_model

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Dental Analysis System",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern medical theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2C5F7C;
        --secondary-color: #4A90A4;
        --accent-color: #1E88E5;
        --success-color: #43A047;
        --warning-color: #FB8C00;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2C5F7C 0%, #4A90A4 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 600;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Card styling */
    .analysis-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .image-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 2px solid #e8eef2;
        text-align: center;
        height: 100%;
    }
    
    .image-container img {
        border-radius: 8px;
    }
    
    .image-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2C5F7C;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4A90A4;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 1px solid #A5D6A7;
    }
    
    .status-warning {
        background-color: #FFF3E0;
        color: #E65100;
        border: 1px solid #FFCC80;
    }
    
    .status-info {
        background-color: #E3F2FD;
        color: #1565C0;
        border: 1px solid #90CAF9;
    }
    
    /* Metric cards */
    .metric-box {
        background: linear-gradient(135deg, #2C5F7C 0%, #4A90A4 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-box h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-box p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.95;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2C5F7C 0%, #4A90A4 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Info box */
    .info-box {
        background: #F5F7FA;
        border-left: 4px solid #4A90A4;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: #2C5F7C;
        margin-top: 0;
        font-size: 1rem;
    }
    
    /* Results section */
    .results-header {
        background: #F5F7FA;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #4A90A4;
    }
    
    .results-header h3 {
        color: #2C5F7C;
        margin: 0;
        font-size: 1.3rem;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: #FFF3E0;
        border: 1px solid #FFB74D;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 2rem;
    }
    
    .disclaimer-title {
        color: #E65100;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Adjust spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🦷 Dental Analysis System</h1>
    <p>AI-Powered Teeth, Nerve & Caries Detection Platform</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🔧 Analysis Configuration")

# Model Options
model_type = st.sidebar.radio(
    "Select Analysis Type",
    ['Teeth segmentation', 'Nerve segmentation', 'Caries Detection'],
    help="Choose the type of dental analysis to perform"
)

confidence = float(st.sidebar.slider(
    "Model Confidence Threshold", 25, 100, 40)) / 100

# Model info box
model_info = {
    'Teeth segmentation': '🦷 Identifies and segments individual teeth structures',
    'Nerve segmentation': '🔴 Detects and maps dental nerve pathways',
    'Caries Detection': '🔍 Identifies cavities, crowns, and fillings'
}

st.sidebar.markdown(f"""
<div class="info-box">
    <h4>📋 Selected Analysis</h4>
    <p>{model_info[model_type]}</p>
</div>
""", unsafe_allow_html=True)

# Load appropriate model based on selection
model = None
if model_type == 'Teeth segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.sidebar.error(f"⚠️ Unable to load model: {ex}")
        
elif model_type == 'Nerve segmentation':
    model_path = Path(settings.NERVE_MODEL)
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.sidebar.error(f"⚠️ Unable to load model: {ex}")
        
elif model_type == 'Caries Detection':
    # Load Keras model for caries detection
    caries_model_path = Path(settings.CARIES_MODEL)
    try:
        with st.spinner('🔄 Loading Caries Detection Model...'):
            model = load_model(caries_model_path)
            st.sidebar.success("✅ Model ready!")
    except Exception as ex:
        st.sidebar.error(f"⚠️ Unable to load model: {ex}")

# Display options for YOLO models
if model_type in ['Teeth segmentation', 'Nerve segmentation']:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Display Options")
    
    show_boxes = st.sidebar.toggle(
        "Show Bounding Boxes",
        value=False,
        help="Toggle to show/hide bounding boxes with labels and confidence scores"
    )
    
    st.sidebar.markdown(f"""
    <div class="info-box" style="margin-top: 1rem;">
        <p><small>{'📦 Boxes: ON - Shows detection boxes with labels' if show_boxes else '🎯 Masks Only - Shows segmentation regions only'}</small></p>
    </div>
    """, unsafe_allow_html=True)
else:
    show_boxes = False  # Not applicable for caries detection

# Sidebar divider
st.sidebar.markdown("---")
st.sidebar.markdown("### 📤 Upload X-Ray Image")

source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST, label_visibility="collapsed")

source_img = None

# Helper functions for caries detection
def preprocess_image_caries(image, target_size=(256, 256)):
    """Preprocess image for caries detection model"""
    image = image.convert('RGB')
    image = image.resize(target_size, PIL.Image.Resampling.LANCZOS)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_caries(model, image):
    """Predict caries segmentation"""
    try:
        pred = model.predict(image, verbose=0)
        if pred.shape[-1] == 1:
            return (pred[0] >= 0.5).astype(np.int32)
        else:
            return np.argmax(pred[0], axis=-1)
    except Exception as e:
        st.error(f"Error predicting: {e}")
        return np.zeros((256, 256), dtype=np.int32)

def create_caries_visualization(original_image, prediction):
    """Create visualization for caries detection"""
    category_names = {0: 'Normal/Background', 1: 'Caries', 2: 'Crown', 3: 'Filling'}
    
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    ax.imshow(original_image[0])
    masked_prediction = np.ma.masked_where(prediction == 0, prediction)
    ax.imshow(masked_prediction, cmap='tab10', alpha=0.6, vmin=0, vmax=3)
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.tab10(i/3), label=category_names[i+1]) 
                      for i in range(3)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
             framealpha=0.9, edgecolor='gray')
    
    plt.tight_layout()
    return fig

def get_caries_statistics(prediction):
    """Calculate statistics for caries detection"""
    category_names = {0: 'Normal/Background', 1: 'Caries', 2: 'Crown', 3: 'Filling'}
    unique, counts = np.unique(prediction, return_counts=True)
    total_pixels = prediction.size
    
    stats = {}
    for cls, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        stats[category_names.get(cls, 'Unknown')] = {
            'pixels': count,
            'percentage': percentage
        }
    
    return stats

# Main content area
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", 
        type=("jpg", "jpeg", "png", 'bmp', 'webp'),
        help="Upload a dental X-ray image for analysis"
    )
    
    # Add segment button in sidebar
    st.sidebar.markdown("---")
    analyze_button = st.sidebar.button('🔬 Analyze Image', use_container_width=True)

    # Create two equal columns for images
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.markdown('<div class="image-title">📷 Original X-Ray Image</div>', unsafe_allow_html=True)
        
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, use_container_width=True)
                st.caption("Default sample image")
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, use_container_width=True)
                st.caption(f"Uploaded: {source_img.name}")
        except Exception as ex:
            st.error(f"⚠️ Error loading image: {ex}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.markdown('<div class="image-title">🎯 Analysis Results</div>', unsafe_allow_html=True)
        
        if source_img is None:
            try:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(default_detected_image_path)
                st.image(default_detected_image_path, use_container_width=True)
                st.caption("Default detection result")
            except:
                st.info("👆 Upload an image and click 'Analyze Image' to see results")
        else:
            if analyze_button:
                if model is None:
                    st.error("⚠️ Model not loaded. Please check the configuration.")
                else:
                    if model_type in ['Teeth segmentation', 'Nerve segmentation']:
                        # Original YOLO segmentation logic
                        with st.spinner('🔄 Processing...'):
                            res = model.predict(uploaded_image, conf=confidence)
                            boxes = res[0].boxes
                            
                            # Create visualization based on toggle
                            if show_boxes:
                                # Show with bounding boxes, labels and confidence
                                res_plotted = res[0].plot()[:, :, ::-1]
                            else:
                                # Show only segmentation masks without boxes
                                import cv2
                                from PIL import Image as PILImage
                                
                                # Get original image as array
                                img_array = np.array(uploaded_image)
                                
                                # Check if masks exist
                                if res[0].masks is not None:
                                    masks = res[0].masks.data.cpu().numpy()
                                    
                                    # Create overlay image
                                    overlay = img_array.copy()
                                    
                                    # Apply each mask with different colors
                                    for i, mask in enumerate(masks):
                                        # Resize mask to match image size
                                        mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]))
                                        
                                        # Generate color for this mask (using different colors)
                                        color = plt.cm.tab10(i % 10)[:3]
                                        color = tuple(int(c * 255) for c in color)
                                        
                                        # Create colored mask
                                        colored_mask = np.zeros_like(img_array)
                                        colored_mask[mask_resized > 0.5] = color
                                        
                                        # Blend with overlay
                                        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
                                    
                                    res_plotted = overlay
                                else:
                                    # No masks found, use original image
                                    res_plotted = img_array
                            
                            st.image(res_plotted, use_container_width=True)
                        
                        detection_count = len(boxes) if boxes is not None else 0
                        st.caption(f"✅ Detected {detection_count} regions")
                    
                    elif model_type == 'Caries Detection':
                        # Caries detection logic
                        processed_image = preprocess_image_caries(uploaded_image)
                        
                        with st.spinner('🧠 AI is analyzing...'):
                            prediction = predict_caries(model, processed_image)
                        
                        # Create and display visualization
                        fig = create_caries_visualization(processed_image, prediction)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        
                        st.caption("✅ Analysis complete")
            else:
                st.info("👈 Click 'Analyze Image' in the sidebar to start analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Results section (only show after analysis for caries detection)
    if source_img is not None and analyze_button and model_type == 'Caries Detection' and model is not None:
        st.markdown("---")
        
        # Results header
        st.markdown("""
        <div class="results-header">
            <h3>📊 Detailed Analysis Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Get statistics
        stats = get_caries_statistics(prediction)
        
        # Display metrics in columns
        metric_cols = st.columns(4)
        
        for i, (region, data) in enumerate(stats.items()):
            if region != 'Normal/Background':
                with metric_cols[i-1]:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{data['percentage']:.1f}%</h3>
                        <p>{region}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Calculate total pathology
        total_pathology = sum(
            data['percentage'] 
            for region, data in stats.items() 
            if region in ['Caries', 'Crown', 'Filling']
        )
        
        # Status indicator
        col_status1, col_status2 = st.columns([1, 2])
        
        with col_status1:
            if total_pathology > 20:
                st.markdown("""
                <div class="status-badge status-warning">
                    ⚠️ Significant Findings
                </div>
                """, unsafe_allow_html=True)
            elif total_pathology > 5:
                st.markdown("""
                <div class="status-badge status-info">
                    ⚡ Moderate Findings
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-badge status-success">
                    ✅ Minimal Findings
                </div>
                """, unsafe_allow_html=True)
        
        with col_status2:
            st.metric("Total Pathology Coverage", f"{total_pathology:.2f}%")
        
        # Detailed breakdown
        with st.expander("📋 View Detailed Breakdown", expanded=False):
            for region, data in stats.items():
                if region != 'Normal/Background':
                    st.write(f"**{region}:** {data['percentage']:.2f}% ({data['pixels']} pixels)")
        
        # Disclaimer
        st.markdown("""
        <div class="disclaimer">
            <div class="disclaimer-title">⚠️ Important Medical Disclaimer</div>
            <p>This AI analysis is for educational and research purposes only. 
            Results should not replace professional dental diagnosis or treatment recommendations. 
            Always consult with qualified dental professionals for medical advice and treatment decisions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show results for other segmentation types
    elif source_img is not None and analyze_button and model_type in ['Teeth segmentation', 'Nerve segmentation'] and model is not None:
        st.markdown("---")
        st.markdown("""
        <div class="results-header">
            <h3>📊 Detection Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔍 View Detection Details", expanded=True):
            try:
                for i, box in enumerate(boxes):
                    st.write(f"**Detection {i+1}:**", box.data)
            except:
                st.info("No specific detections to display")

else:
    st.error("⚠️ Please select a valid source type!")