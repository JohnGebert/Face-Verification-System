# Face Verification System

A comprehensive machine learning project that implements both traditional predictive models and advanced Siamese neural networks for face verification tasks using the Labeled Faces in the Wild (LFW) dataset.

## 📋 Project Overview

This project focuses on developing a robust face verification system that can determine whether two face images belong to the same person or different persons. The system implements multiple approaches including traditional machine learning (Logistic Regression) and advanced deep learning (Siamese neural networks) with comprehensive scenario analysis and interactive dashboards.

### Key Features

- **Face Verification**: Determines if two face images belong to the same person
- **Multiple Model Approaches**: Traditional ML (Logistic Regression) and Advanced Neural Networks
- **Siamese Neural Network**: Uses twin neural networks to learn similarity metrics
- **LFW Dataset**: Utilizes the Labeled Faces in the Wild dataset for training and testing
- **Comprehensive Data Preprocessing**: Data preparation, normalization, and quality assessment
- **Exploratory Data Analysis**: Detailed analysis of pixel intensity distributions
- **Scenario Analysis**: Robustness testing under various real-world conditions
- **Interactive Dashboard**: Power BI dashboard for stakeholder communication
- **Decision Support**: Comprehensive analysis for practical deployment decisions

## 🏗️ Project Structure

```
Face Recognition System/
├── data/                          # Dataset and analysis data
│   ├── lfw_preprocessed_data.npz  # Preprocessed LFW dataset
│   ├── SDC486L_Project4.2_Gebert.xlsx  # Analysis results data
│   └── Book1.xlsx                 # Additional analysis data
├── dashboards/
│   └── SDC486L_Project4.2_Gebert.pbix  # Power BI interactive dashboard
├── notebooks/
│   ├── SDC486L_Project2.2_Gebert.ipynb  # Data preparation and EDA
│   ├── SDC486L_Project3.2_Gebert.ipynb  # Traditional ML and neural network implementation
│   └── SDC486LProject5.2_Gebert.ipynb   # Scenario analysis and decision support
└── reports/
    ├── SDC486L_Project1.4_Gebert.docx    # Project requirements and planning
    ├── SDC486L_Project2.2_Gebert.docx    # Data preparation and EDA report
    ├── SDC486L_Project3.2_Gebert.docx    # Model implementation report
    ├── SDC486L_Project4.2_Gebert.docx    # Dashboard and visualization report
    └── SDC486L_Project5.2_Gebert.docx    # Comprehensive final report
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Power BI Desktop (for dashboard viewing)
- Required Python packages (see Installation section)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/JohnGebert/Face-Verification-System.git
cd Face-Verification-System
```

2. Install required dependencies:
```bash
pip install numpy matplotlib seaborn scikit-learn tensorflow keras pandas openpyxl
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open notebooks in sequence:
   - `notebooks/SDC486L_Project2.2_Gebert.ipynb` - Data preparation and EDA
   - `notebooks/SDC486L_Project3.2_Gebert.ipynb` - Model implementation
   - `notebooks/SDC486LProject5.2_Gebert.ipynb` - Scenario analysis

5. View the interactive dashboard:
   - Open `dashboards/SDC486L_Project4.2_Gebert.pbix` in Power BI Desktop

## 📊 Dataset

The project uses the **Labeled Faces in the Wild (LFW)** dataset, which is specifically designed for face verification tasks. The dataset includes:

- **Training subset**: 2,200 pairs of face images
- **Testing subset**: 1,000 pairs of face images
- **Binary classification**: Same person (1) vs. Different persons (0)
- **Grayscale images**: Resized to 62x47 pixels for computational efficiency

### Data Characteristics

- **Image format**: Grayscale (1 channel)
- **Pixel range**: 0-255 (normalized to 0-1 for neural network input)
- **Image dimensions**: 62x47 pixels (consistent sizing)
- **Class distribution**: Balanced between same/different person pairs

## 🔧 Technical Implementation

### Model Approaches

#### 1. Traditional Machine Learning
- **Logistic Regression**: Baseline model for comparison
- **Feature Engineering**: Flattened image pairs as input features
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score

#### 2. Advanced Neural Networks
- **Siamese Neural Network**: Twin CNN architecture for similarity learning
- **Convolutional Layers**: Feature extraction from image data
- **Distance Learning**: Euclidean distance-based similarity measurement
- **Optimization**: Adam optimizer with early stopping

### Data Preprocessing Pipeline

1. **Data Loading**: Fetch LFW pairs dataset using scikit-learn
2. **Missing Value Detection**: Verify data integrity and quality
3. **Normalization**: Scale pixel values from [0, 255] to [0, 1]
4. **Reshaping**: Prepare images for CNN input format (62x47x1)
5. **Pair Separation**: Split image pairs for Siamese network architecture

### Key Components

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of pixel intensity distributions
- **Data Visualization**: Histograms, box plots, and statistical analysis
- **Statistical Analysis**: Summary statistics, outlier detection, and data quality assessment
- **Preprocessing Pipeline**: Automated data preparation for multiple model types

## 📈 Results and Analysis

### Model Performance Comparison

- **Logistic Regression Baseline**: Traditional ML approach for comparison
- **Siamese Neural Network**: Advanced deep learning with improved accuracy
- **Performance Metrics**: Comprehensive evaluation across multiple criteria

### Scenario Analysis

The project includes comprehensive scenario analysis to test model robustness:

#### Scenario 1: Image Quality Degradation
- **Description**: Simulate poor image quality with noise and contrast reduction
- **Rationale**: Real-world conditions from surveillance cameras and mobile devices
- **Impact**: Tests model performance under suboptimal image conditions

#### Scenario 2: Facial Occlusion
- **Description**: Simulate partial face occlusion (masks, sunglasses, etc.)
- **Rationale**: Common real-world challenges in facial recognition
- **Impact**: Evaluates robustness when key features are obscured

#### Scenario 3: Lighting Variation
- **Description**: Simulate extreme lighting conditions
- **Rationale**: Environmental variations affecting recognition accuracy
- **Impact**: Tests generalization across different lighting environments

### Decision Support Analysis

- **Performance Degradation Quantification**: Percentage changes in accuracy and F1-score
- **Robustness Assessment**: Model behavior under varying conditions
- **Optimization Recommendations**: Threshold tuning and model improvements
- **Deployment Considerations**: Real-world applicability and limitations

## 📊 Interactive Visualization and Dashboarding

### Power BI Dashboard Features

- **Model Performance Metrics**: Interactive charts showing accuracy, precision, recall, and F1-score
- **Scenario Analysis Results**: Visual comparison of performance under different conditions
- **Data Quality Insights**: Interactive exploration of dataset characteristics
- **Decision Support Tools**: Stakeholder-friendly visualizations for deployment decisions

### Dashboard Components

- **Performance Comparison Charts**: Side-by-side model evaluation
- **Scenario Impact Analysis**: Visual representation of robustness testing
- **Interactive Filters**: Dynamic exploration of different analysis dimensions
- **Executive Summary Views**: High-level insights for stakeholders

## 📚 Documentation

### Comprehensive Reports

- **SDC486L_Project1.4_Gebert.docx**: Project requirements and initial planning
- **SDC486L_Project2.2_Gebert.docx**: Data preparation and exploratory analysis
- **SDC486L_Project3.2_Gebert.docx**: Model implementation and evaluation
- **SDC486L_Project4.2_Gebert.docx**: Dashboard development and visualization
- **SDC486L_Project5.2_Gebert.docx**: Comprehensive final report with scenario analysis

### Technical Implementation

- **SDC486L_Project2.2_Gebert.ipynb**: Complete data preparation pipeline
- **SDC486L_Project3.2_Gebert.ipynb**: Traditional ML and neural network implementation
- **SDC486LProject5.2_Gebert.ipynb**: Advanced scenario analysis and decision support

## 🎯 Key Achievements

### Requirements Met

✅ **Dataset Selection and Problem Identification**: Clear face verification problem with LFW dataset
✅ **Data Cleaning, EDA, and Preprocessing**: Comprehensive data preparation pipeline
✅ **Predictive Modeling and Neural Network Implementation**: Both traditional ML and advanced neural networks
✅ **Interactive Visualization and Dashboarding**: Power BI dashboard for stakeholder communication
✅ **Scenario Analysis and Decision Support**: Comprehensive robustness testing and decision guidance
✅ **Comprehensive Reporting and Communication**: Professional reports with actionable insights

### Technical Highlights

- **Multi-Model Approach**: Comparison of traditional and advanced methods
- **Robustness Testing**: Real-world scenario simulation and analysis
- **Interactive Dashboards**: Stakeholder-friendly visualization tools
- **Comprehensive Documentation**: Complete project lifecycle documentation
- **Decision Support**: Practical insights for deployment and optimization

## 🤝 Contributing

This is an academic project for SDC486L. For questions or contributions, please contact the project maintainer.

## 📄 License

This project is part of an academic course (SDC486L) and is intended for educational purposes.

## 👨‍💻 Author

**John Gebert** - Student ID: johgeb8270

## 📞 Contact

For questions about this project, please refer to the project documentation or contact the author through the academic institution.

---

*This project was developed as part of SDC486L coursework, demonstrating comprehensive machine learning implementation from data preparation through deployment-ready decision support systems.*