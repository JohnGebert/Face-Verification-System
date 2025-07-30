# Face Verification System

A machine learning project that implements a Siamese neural network for face verification tasks using the Labeled Faces in the Wild (LFW) dataset.

## Project Overview

This project focuses on developing a face verification system that can determine whether two face images belong to the same person or different persons. The system uses a Siamese neural network architecture, which is particularly effective for similarity-based learning tasks.

### Key Features

- **Face Verification**: Determines if two face images belong to the same person
- **Siamese Neural Network**: Uses twin neural networks to learn similarity metrics
- **LFW Dataset**: Utilizes the Labeled Faces in the Wild dataset for training and testing
- **Data Preprocessing**: Comprehensive data preparation and normalization pipeline
- **Exploratory Data Analysis**: Detailed analysis of pixel intensity distributions

## Project Structure

```
Face Recognition System/
├── data/                          # Dataset storage directory
├── notebooks/
│   └── SDC486L_Project2.2_Gebert.ipynb  # Main implementation notebook
└── reports/
    ├── SDC486L_Project1.4_Gebert.docx    # Project documentation
    └── SDC486L_Project2.2_Gebert.docx    # Technical report
```

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Python packages (see Installation section)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/JohnGebert/Face-Verification-System.git
cd Face-Verification-System
```

2. Install required dependencies:
```bash
pip install numpy matplotlib seaborn scikit-learn tensorflow keras
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open the main notebook: `notebooks/SDC486L_Project2.2_Gebert.ipynb`

## Dataset

The project uses the **Labeled Faces in the Wild (LFW)** dataset, which is specifically designed for face verification tasks. The dataset includes:

- **Training subset**: 1,100 pairs of face images
- **Testing subset**: 1,000 pairs of face images
- **Binary classification**: Same person (1) vs. Different persons (0)
- **Grayscale images**: Resized to 50% of original size for computational efficiency

### Data Characteristics

- **Image format**: Grayscale (1 channel)
- **Pixel range**: 0-255 (normalized to 0-1 for neural network input)
- **Image dimensions**: Variable (resized for consistency)
- **Class distribution**: Slightly imbalanced between same/different person pairs

## Technical Implementation

### Data Preprocessing Pipeline

1. **Data Loading**: Fetch LFW pairs dataset using scikit-learn
2. **Missing Value Detection**: Verify data integrity
3. **Normalization**: Scale pixel values from [0, 255] to [0, 1]
4. **Reshaping**: Prepare images for CNN input format
5. **Pair Separation**: Split image pairs for Siamese network architecture

### Model Architecture

The system implements a **Siamese Neural Network** with the following characteristics:

- **Twin Networks**: Two identical CNN branches
- **Feature Extraction**: Convolutional layers for hierarchical feature learning
- **Similarity Learning**: Distance-based comparison between feature vectors
- **Binary Classification**: Sigmoid output for same/different person prediction

### Key Components

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of pixel intensity distributions
- **Data Visualization**: Histograms and box plots for data understanding
- **Statistical Analysis**: Summary statistics and outlier detection
- **Preprocessing Pipeline**: Automated data preparation for model training

## Results and Analysis

The project includes detailed analysis of:

- **Pixel Intensity Distribution**: Understanding of image characteristics
- **Data Quality Assessment**: Missing value detection and data integrity
- **Feature Engineering**: Preparation for deep learning models
- **Model Readiness**: Verification of data suitability for neural networks

## Documentation

- **SDC486L_Project1.4_Gebert.docx**: Initial project documentation and requirements
- **SDC486L_Project2.2_Gebert.docx**: Technical implementation report
- **SDC486L_Project2.2_Gebert.ipynb**: Complete implementation with detailed comments

## Contributing

This is an academic project for SDC486L. For questions or contributions, please contact the project maintainer.

## License

This project is part of an academic course (SDC486L) and is intended for educational purposes.

## Author

**John Gebert** - Student ID: johgeb8270

## Contact

For questions about this project, please refer to the project documentation or contact the author through the academic institution.

---

*This project was developed as part of SDC486L coursework, focusing on machine learning applications in computer vision and face recognition technologies.*