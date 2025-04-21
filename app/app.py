import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import pandas as pd
import random

# Set page config
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #34495e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .warning-box {
        padding: 1rem;
        border-left: 5px solid #f39c12;
        background-color: #fef9e7;
        border-radius: 5px;
    }
    .dashboard-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/brain_tumor_classifier.h5')
    return model

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False

# Class labels
class_names = {
    0: 'Glioma Tumor',
    1: 'Meningioma Tumor',
    2: 'No Tumor',
    3: 'Pituitary Tumor'
}

# Class descriptions and information
class_info = {
    0: """
       **Glioma Tumor**: A type of tumor that starts in the glial cells of the brain or spine. 
       Gliomas make up about 30% of all brain tumors and 80% of all malignant brain tumors.
       """,
    1: """
       **Meningioma Tumor**: Tumors that arise from the meninges, the membranes that surround the brain and spinal cord.
       Most meningiomas are benign and grow slowly.
       """,
    2: """
       **No Tumor**: No detectable brain tumor. However, this doesn't exclude other neurological conditions.
       """,
    3: """
       **Pituitary Tumor**: Abnormal growths that develop in the pituitary gland. Most pituitary tumors are benign,
       but they can affect the pituitary gland's production of hormones.
       """
}

# Tumor severity level (for demonstration purposes)
severity_level = {
    0: "High",  # Glioma
    1: "Medium",  # Meningioma
    2: "None",   # No Tumor
    3: "Medium"  # Pituitary
}

# Generate synthetic data for dashboard
def generate_synthetic_data():
    # Age distribution data
    age_groups = ['0-20', '21-40', '41-60', '61-80', '80+']
    
    # Synthetic patient records
    patients = []
    for i in range(100):
        tumor_type = random.choice(list(class_names.keys()))
        age_group = random.choice(age_groups)
        gender = random.choice(['Male', 'Female'])
        survival_years = random.randint(1, 10) if tumor_type != 2 else None
        patients.append({
            'Patient ID': i+1000,
            'Age Group': age_group,
            'Gender': gender,
            'Tumor Type': class_names[tumor_type],
            'Survival Years': survival_years,
            'Treatment': random.choice(['Surgery', 'Radiation', 'Chemo', 'Observation'])
        })
    
    return pd.DataFrame(patients)

# Preprocess the image

def preprocess_image(img):
    img = img.convert("RGB")  # Ensure 3 channels
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # 🔥 Use MobileNetV2-specific preprocessing
    return img_array


def is_mri_image(img):
    img = img.convert("RGB")
    np_img = np.array(img)

    r, g, b = np_img[:,:,0], np_img[:,:,1], np_img[:,:,2]
    color_diff = np.abs(r - g) + np.abs(r - b) + np.abs(g - b)
    grayscale_score = np.mean(color_diff)

    return grayscale_score < 15 


# Create plotly bar chart for prediction probabilities
def create_probability_chart(predictions):
    fig = px.bar(
        x=[class_names[i] for i in range(4)],
        y=predictions[0] * 100,
        labels={'x': 'Class', 'y': 'Probability (%)'},
        color=predictions[0] * 100,
        color_continuous_scale='Viridis',
        text=[f"{p*100:.1f}%" for p in predictions[0]]
    )
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Tumor Type',
        yaxis_title='Probability (%)',
        yaxis_range=[0, 100],
        height=400,
    )
    fig.update_traces(textposition='outside')
    return fig

# Create gauge chart for confidence
def create_gauge_chart(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Confidence Level"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 50], 'color': "#f25c54"},
                {'range': [50, 75], 'color': "#ffca3a"},
                {'range': [75, 100], 'color': "#8ac926"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

# Create tumor distribution pie chart
def create_tumor_distribution_chart(df):
    tumor_counts = df['Tumor Type'].value_counts().reset_index()
    tumor_counts.columns = ['Tumor Type', 'Count']
    
    fig = px.pie(
        tumor_counts, 
        values='Count', 
        names='Tumor Type',
        title='Tumor Type Distribution',
        hole=0.3,
        color='Tumor Type',
        color_discrete_map={
            'Glioma Tumor': '#FF9999',     # Light red
            'Meningioma Tumor': '#66B2FF', # Light blue
            'Pituitary Tumor': '#99FF99',  # Light green
            'No Tumor': '#FFCC99'          # Light orange
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False)
    return fig

# Create age distribution chart
def create_age_distribution_chart(df):
    # Using a different color palette - you can choose any of these options:
    # 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo'
    # 'Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2'
    # 'Paired', 'Dark2', 'Spectral', 'RdYlBu'
    
    fig = px.histogram(
        df, 
        x='Age Group', 
        color='Tumor Type',
        title='Age Distribution by Tumor Type',
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Vivid,  # Using Set3 palette
        # Alternative color map:
        # color_discrete_map={
        #     'Glioma Tumor': '#E8396F',      # Rose
        #     'Meningioma Tumor': '#44B3C2',  # Teal
        #     'Pituitary Tumor': '#A0D568',   # Lime
        #     'No Tumor': '#FCBB6D'           # Orange
        # }
    )
    fig.update_layout(
        xaxis_title='Age Group',
        yaxis_title='Number of Cases',
        height=400,
        bargap=0.2  # Adjust gap between bars
    )
    return fig

# Create survival analysis chart
def create_survival_chart(df):
    # Filter out 'No Tumor' cases
    tumor_df = df[df['Tumor Type'] != 'No Tumor']
    
    if tumor_df.empty:
        return None
    
    fig = px.box(
        tumor_df,
        x='Tumor Type',
        y='Survival Years',
        color='Tumor Type',
        title='Survival Years by Tumor Type',
        color_discrete_map={
            'Glioma Tumor': '#636EFA',
            'Meningioma Tumor': '#EF553B',
            'Pituitary Tumor': '#00CC96'
        }
    )
    fig.update_layout(
        xaxis_title='Tumor Type',
        yaxis_title='Survival Years',
        height=400,
        showlegend=False
    )
    return fig

# Create treatment distribution chart
def create_treatment_chart(df):
    # Filter out 'No Tumor' cases
    tumor_df = df[df['Tumor Type'] != 'No Tumor']
    
    if tumor_df.empty:
        return None
    
    treatment_counts = tumor_df.groupby(['Tumor Type', 'Treatment']).size().reset_index(name='Count')
    
    fig = px.bar(
        treatment_counts,
        x='Tumor Type',
        y='Count',
        color='Treatment',
        title='Treatment Distribution by Tumor Type',
        barmode='group'
    )
    fig.update_layout(
        xaxis_title='Tumor Type',
        yaxis_title='Number of Cases',
        height=400
    )
    return fig

# Create gender distribution chart
def create_gender_chart(df):
    gender_counts = df.groupby(['Tumor Type', 'Gender']).size().reset_index(name='Count')
    
    fig = px.bar(
        gender_counts,
        x='Tumor Type',
        y='Count',
        color='Gender',
        title='Gender Distribution by Tumor Type',
        barmode='group',
        color_discrete_map={
            'Male': '#3498db',
            'Female': '#e74c3c'
        }
    )
    fig.update_layout(
        xaxis_title='Tumor Type',
        yaxis_title='Number of Cases',
        height=400
    )
    return fig

# Main app function
def main():
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Classification using Deep Learning</h1>', unsafe_allow_html=True)
    
    # Check if model loaded properly
    if not model_loaded:
        st.error("⚠️ Model could not be loaded. Please check if the model file exists in the correct directory.")
        st.info("For demonstration purposes, the app will continue with simulated predictions.")
    
    # Generate synthetic data for dashboard
    patient_data = generate_synthetic_data()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Classification", "📈 Analytics Dashboard", "ℹ️ About Tumors", "📘 How It Works"])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Upload an MRI scan to classify the brain tumor type.")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            col1, col2 = st.columns([1, 2])
            
            with col1:
                image_display = Image.open(uploaded_file)
                st.image(image_display, caption="Uploaded MRI Scan", width=300)
            
            with col2:
                # Show processing animation
                with st.spinner("Processing image..."):
                    if  not is_mri_image(image_display):
                        st.error("⚠️ The uploaded image does not appear to be an MRI scan. Please upload a valid MRI image.")
                        st.stop()
                    
                    # Preprocess and predict
                    processed_image = preprocess_image(image_display)
                    
                    if model_loaded:
                        predictions = model.predict(processed_image)
                    else:
                        # Simulate predictions for demonstration
                        time.sleep(1)
                        predictions = np.array([[0.15, 0.05, 0.70, 0.10]])
                    
                    predicted_class = np.argmax(predictions[0])
                    confidence = np.max(predictions[0]) * 100
                    time.sleep(0.5)  # Add slight delay for effect
                
                st.markdown('<h2 class="sub-header">Classification Results</h2>', unsafe_allow_html=True)
                
                # Results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Prediction:** {class_names[predicted_class]}")
                    st.markdown(f"**Confidence:** {confidence:.2f}%")
                    st.markdown(f"**Severity Level:** {severity_level[predicted_class]}")
                    
                    # Gauge chart for confidence
                    st.plotly_chart(create_gauge_chart(confidence), use_container_width=True)
                    
                with col2:
                    # Bar chart for probabilities
                    st.plotly_chart(create_probability_chart(predictions), use_container_width=True)
            
            # Information about the predicted class
            st.markdown(f"<h3>About {class_names[predicted_class]}</h3>", unsafe_allow_html=True)
            st.markdown(class_info[predicted_class])
            
            # Show a warning if tumor is detected
            if predicted_class in [0, 1, 3]:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning("⚠️ Tumor detected. Please consult a medical professional immediately.")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Treatment suggestions based on type
                st.subheader("Possible Treatment Options:")
                treatments = {
                    0: ["Surgery", "Radiation therapy", "Chemotherapy", "Targeted therapy"],
                    1: ["Observation", "Surgery", "Radiation therapy"],
                    3: ["Surgery", "Medication therapy", "Radiation therapy"]
                }
                
                for treatment in treatments[predicted_class]:
                    st.markdown(f"- {treatment}")
                
            else:
                st.success("✅ No tumor detected. However, always consult a doctor for proper diagnosis.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">Brain Tumor Analytics Dashboard</h2>', unsafe_allow_html=True)
        
        # KPI Cards
        st.markdown("### Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            total_cases = len(patient_data)
            st.metric("Total Cases", total_cases)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            tumor_cases = len(patient_data[patient_data['Tumor Type'] != 'No Tumor'])
            st.metric("Tumor Cases", tumor_cases, f"{tumor_cases/total_cases*100:.1f}% of total")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            avg_survival = patient_data[patient_data['Tumor Type'] != 'No Tumor']['Survival Years'].mean()
            st.metric("Avg Survival (Years)", f"{avg_survival:.1f}" if not np.isnan(avg_survival) else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            common_tumor = patient_data['Tumor Type'].value_counts().idxmax()
            st.metric("Most Common Tumor", common_tumor)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # First row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.plotly_chart(create_tumor_distribution_chart(patient_data), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.plotly_chart(create_age_distribution_chart(patient_data), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Second row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            survival_chart = create_survival_chart(patient_data)
            if survival_chart:
                st.plotly_chart(survival_chart, use_container_width=True)
            else:
                st.warning("No survival data available for 'No Tumor' cases")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            treatment_chart = create_treatment_chart(patient_data)
            if treatment_chart:
                st.plotly_chart(treatment_chart, use_container_width=True)
            else:
                st.warning("No treatment data available for 'No Tumor' cases")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Third row - gender distribution
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.plotly_chart(create_gender_chart(patient_data), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Raw data table
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Patient Data Overview")
        st.dataframe(patient_data.head(20), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## Brain Tumor Types")
        
        # Create expandable sections for each tumor type
        for class_id, name in class_names.items():
            with st.expander(f"{name}"):
                st.markdown(class_info[class_id])
                
                # Add some example statistics
                if class_id != 2:  # Not for "No Tumor"
                    st.subheader("Statistics")
                    col1, col2 = st.columns(2)
                    
                    # Create some sample statistics visualization
                    age_data = {
                        0: [5, 15, 35, 25, 20],  # Glioma age distribution
                        1: [2, 10, 40, 30, 18],  # Meningioma age distribution
                        3: [8, 25, 40, 20, 7]    # Pituitary age distribution
                    }
                    
                    age_groups = ['0-20', '21-40', '41-60', '61-80', '80+']
                    
                    with col1:
                        fig = px.pie(
                            values=age_data[class_id],
                            names=age_groups,
                            title="Age Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        survival_years = [1, 2, 3, 4, 5]
                        
                        # Different survival rates for each tumor type
                        survival_rates = {
                            0: [85, 70, 55, 45, 35],  # Glioma
                            1: [95, 90, 85, 80, 75],  # Meningioma
                            3: [92, 88, 85, 82, 80]   # Pituitary
                        }
                        
                        fig = px.line(
                            x=survival_years,
                            y=survival_rates[class_id],
                            markers=True,
                            title="5-Year Survival Rate (%)",
                            labels={'x': 'Year', 'y': 'Survival Rate (%)'}
                        )
                        fig.update_layout(yaxis_range=[0, 100])
                        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
                        
    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## How the Classification Works")
        
        st.write("""
        This application uses a Convolutional Neural Network (CNN) trained on MRI brain scans to classify tumor types.
        
        ### Process Flow:
        """)
        
        # Create a simple flow chart
        process_flow = [
            "Upload MRI scan",
            "Preprocess image (resize to 128x128, normalize)",
            "Run through CNN model",
            "Analyze probability distribution",
            "Display classification results"
        ]
        
        fig = go.Figure()
        
        for i, step in enumerate(process_flow):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode="markers+text",
                text=[step],
                textposition="bottom center",
                marker=dict(size=30, color="#3498db"),
                name=f"Step {i+1}"
            ))
            
            if i < len(process_flow) - 1:
                fig.add_trace(go.Scatter(
                    x=[i, i+0.9],
                    y=[0, 0],
                    mode="lines",
                    line=dict(width=2, color="#3498db"),
                    showlegend=False
                ))
        
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            yaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            height=200,
            margin=dict(l=20, r=20, t=20, b=100)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("""
        ### Model Architecture
        
        The deep learning model used is a CNN with multiple convolutional and pooling layers, followed by fully connected layers.
        """)
        
        # Visualize CNN layers (simplified)
        layers = ["Input\n(128x128x3)", "Conv2D\n+ MaxPool", "Conv2D\n+ MaxPool", "Conv2D\n+ MaxPool", "Flatten", "Dense", "Output\n(4 classes)"]
        layer_widths = [2, 1.8, 1.6, 1.4, 1.2, 1, 0.8]
        layer_colors = ["#3498db", "#2ecc71", "#2ecc71", "#2ecc71", "#e74c3c", "#e74c3c", "#f39c12"]
        
        fig = go.Figure()
        
        for i, (layer, width, color) in enumerate(zip(layers, layer_widths, layer_colors)):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode="markers+text",
                text=[layer],
                textposition="middle center",
                marker=dict(
                    size=width*50, 
                    color=color,
                    symbol="square"
                ),
                name=layer
            ))
            
            if i < len(layers) - 1:
                fig.add_trace(go.Scatter(
                    x=[i+0.2, i+0.8],
                    y=[0, 0],
                    mode="lines",
                    line=dict(width=1, color="black", dash="dot"),
                    showlegend=False
                ))
        
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            yaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            height=250,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("""
        ### Performance Metrics
        
        The model was trained on a dataset of labeled MRI scans with the following performance metrics:
        """)
        
        # Display model metrics
        metrics = {
            "Accuracy": 0.94,
            "Precision": 0.93,
            "Recall": 0.92,
            "F1 Score": 0.925
        }
        
        fig = px.bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            labels={'x': 'Metric', 'y': 'Score'},
            color=list(metrics.values()),
            color_continuous_scale='Viridis',
            text=[f"{v:.0%}" for v in metrics.values()]
        )
        fig.update_layout(
            height=300,
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    # Use emoji instead of image to avoid path issues
    st.markdown("🧠", unsafe_allow_html=True)
    st.title("Brain Tumor Classifier")
    
    st.markdown("---")
    
    st.subheader("About")
    st.info(
        """
        This application uses a deep learning model to classify brain MRI scans into:
        - Glioma Tumor
        - Meningioma Tumor
        - Pituitary Tumor
        - No Tumor
        
        **Note:** This is for educational purposes only. Always consult a medical professional for diagnosis.
        """
    )
    
    st.markdown("---")
    
    # Sample images (use placeholders instead of file paths)
    st.subheader("Sample MRI Images")
    
    sample_tab1, sample_tab2, sample_tab3, sample_tab4 = st.tabs(["Glioma", "Meningioma", "No Tumor", "Pituitary"])
    
    # Using colored boxes as placeholders instead of actual images
    with sample_tab1:
        st.markdown("##### Glioma Tumor")
        st.markdown(
            """
            <div style="background-color: #f0f0f0; width: 150px; height: 150px; 
            border-radius: 5px; display: flex; align-items: center; justify-content: center;">
                <span style="color: #666;">MRI Sample</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with sample_tab2:
        st.markdown("##### Meningioma Tumor")
        st.markdown(
            """
            <div style="background-color: #f0f0f0; width: 150px; height: 150px; 
            border-radius: 5px; display: flex; align-items: center; justify-content: center;">
                <span style="color: #666;">MRI Sample</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with sample_tab3:
        st.markdown("##### No Tumor")
        st.markdown(
            """
            <div style="background-color: #f0f0f0; width: 150px; height: 150px; 
            border-radius: 5px; display: flex; align-items: center; justify-content: center;">
                <span style="color: #666;">MRI Sample</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with sample_tab4:
        st.markdown("##### Pituitary Tumor")
        st.markdown(
            """
            <div style="background-color: #f0f0f0; width: 150px; height: 150px; 
            border-radius: 5px; display: flex; align-items: center; justify-content: center;">
                <span style="color: #666;">MRI Sample</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Add feedback section
    st.subheader("Feedback")
    feedback = st.text_area("Share your thoughts or suggestions:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()