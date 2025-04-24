# Skin Cancer Classification using Deep Learning

## Project Overview
This project implements a **Convolutional Neural Network (CNN)** for multi-class classification of skin cancer using the ISIC (International Skin Imaging Collaboration) dataset. The model classifies **9 types of skin lesions** from dermoscopic images, enabling automated screening of potential skin cancer types.

## Project Structure
```
Skin Cancer Classification/
├── Skin cancer ISIC The International Skin Imaging Collaboration/
│   ├── code.ipynb              # Main implementation notebook
│   ├── Train/                  # Training dataset
│   ├── Test/                   # Testing dataset
│   └── skin_cancer_cnn_model.h5 # Trained model weights
└── README.md
```

## Dataset
The dataset used is from the **International Skin Imaging Collaboration (ISIC)**. It contains thousands of dermoscopic images labeled across 9 skin lesion categories:
- Melanocytic nevi
- Melanoma
- Benign keratosis-like lesions
- Basal cell carcinoma
- Actinic keratoses
- Vascular lesions
- Dermatofibroma
- Squamous cell carcinoma
- Unknown (or additional)

## Data Pipeline

The data pipeline includes the following steps:

1. **Data Loading and Label Mapping**
   - Images are loaded using the `ImageDataGenerator` class from Keras.
   - The `flow_from_directory` method organizes them based on folder names (which represent labels).

2. **Preprocessing**
   - Images are resized to a standard size (typically 224x224).
   - Rescaling (`rescale=1./255`) is applied to normalize pixel values.
   - Data augmentation includes:
     - Rotation
     - Zoom
     - Horizontal and vertical flipping
     - Shearing and shifting

3. **Data Splitting**
   - Training and validation split is handled directly via `ImageDataGenerator`'s `validation_split` parameter.
   - A separate test set is loaded similarly using a different directory.

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
```
### Handling Class Imbalance 
To address class imbalance in the dataset, the following strategies were implemented:

1.Data Augmentation:
The ImageDataGenerator was configured with transformations such as rotation, zoom, shift, and flipping to artificially expand the dataset, particularly benefitting underrepresented classes. This enhances generalization and reduces overfitting.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.1
)

2.Class Weighting:
Class weights were computed and passed to the model during training. These weights are inversely proportional to class frequencies, ensuring that minority classes receive more attention during loss calculation.
class_counts = np.bincount(train_data.classes)
total_samples = np.sum(class_counts)
num_classes = len(class_counts)
class_weights = {
    i: total_samples / (num_classes * count)
    for i, count in enumerate(class_counts)
}

3.Class Distribution Visualization:
   ![Alt Text](assets/Class%20Distribution.png)

## Model Architecture

The CNN is custom-built with:
- **Convolutional Layers**: For hierarchical feature extraction
- **MaxPooling Layers**: For dimensionality reduction
- **Dropout**: To combat overfitting
- **Fully Connected Layers (Dense)**: For classification
- **Softmax Output**: 9-node output layer for multiclass classification

Additional Features:
- **Batch Normalization** for faster convergence
- **Early Stopping and Model Checkpoint** for training optimization

## Technologies & Libraries
- **Python**
- **TensorFlow/Keras** for deep learning model development
- **OpenCV** for image handling
- **Pandas/Numpy** for data manipulation
- **Matplotlib** for EDA and visualization

## Performance Summary
- **Training Accuracy**: 68%
- **Validation Accuracy**: 67%
- **Evaluation Metrics**:
  - Loss
  Training Loss 
   ![Alt Text](assets/Training%20Loss.png)

  Testing Loss
   ![Alt Text](assets/Testing%20Loss.png)
  
  - Confusion matrix for class-wise insights
    ![Alt Text](assets/Confusion%20Matrix.png)

## How to Run

1. **Install Required Libraries**
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn opencv-python
   ```

2. **Launch the Notebook**
   - Open `code.ipynb` using Jupyter
   - Execute cells to:
     - Load and preprocess data
     - Train the model
     - Visualize training history
     - Evaluate and predict

3. **Using the Saved Model**
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('skin_cancer_cnn_model.h5')
   ```

## Highlights 
- Built and optimized a CNN from scratch using **Keras Functional API**
- Hands-on experience with **data augmentation** and **model regularization**
- Handled **imbalanced multiclass classification** using real-world medical image data
- Implemented **class imbalancing handling ** **visualization of learning curves**
- Practical use of `ImageDataGenerator` for efficient image loading and preprocessing
- Project demonstrates strong skills in **deep learning**, **image analysis**, and **model deployment**

## Future Enhancements
- Add **lesion segmentation** using U-Net or Mask R-CNN
- Try **transfer learning** using pretrained models like EfficientNet or MobileNet
- Develop a **Flask web app** for real-time skin lesion prediction
- Evaluate model on additional datasets for generalization

## Dataset Source
[ISIC Archive](https://www.isic-archive.com/): World’s largest publicly available collection of quality-controlled dermoscopic images of skin lesions.

---

**Disclaimer**: This model is intended for research and educational use only. It is not a substitute for professional medical diagnosis.
