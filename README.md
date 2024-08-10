# Video Classification with Conv3D and EfficientNet

This project demonstrates video classification using two different approaches: a Conv3D-based model and a model based on the EfficientNet architecture. 
The models are trained to classify videos into different categories.

## Project Overview

This project is structured as follows:

1. **Data Preparation**: Loading and preprocessing [video data](https://www.kaggle.com/datasets/pevogam/ucf101) from [Kaggle](https://www.kaggle.com/).
2. **Modeling**:
   - A simple Conv3D model.
   - An EfficientNet-based model using transfer learning.
3. **Training**: Training the models and saving the best-performing versions.
4. **Evaluation**: Evaluating the performance of the models on the validation dataset.
5. **Visualization**: Visualizing the learning curves and model architecture.

## Dataset

The dataset used in this project is a subset of the UCF101 dataset, focusing on three classes: "ApplyEyeMakeup", "ApplyLipstick", and "Archery". Each video is processed to extract a fixed number of frames, which are then resized to a uniform shape.

## Project Files

- `video_classification.ipynb`: The Jupyter notebook containing the complete code for data processing, model training, and evaluation.

## Installation

To run this project, you'll need to install the necessary dependencies. This can be done using the following commands:

```bash
pip install tensorflow
pip install opencv-python
pip install scikit-learn
pip install imageio
pip install tqdm
```

## Running the Project

1. **Import the necessary packages**:
   Ensure all required libraries are installed and imported.

2. **Data Preparation**:
   - Load the videos from the dataset.
   - Extract frames from each video and resize them to 224x224 pixels.
   - Split the data into training and validation sets.

3. **Modeling**:
   - Define a simple Conv3D model for video classification.
   - Define a transfer learning model using EfficientNetB0.

4. **Training**:
   - Train the Conv3D model for 30 epochs, saving the best model based on validation accuracy.
   - Train the EfficientNet model for 10 epochs, also saving the best model.

5. **Evaluation**:
   - Evaluate the performance of both models on the validation dataset.
   - Plot the learning curves for both models.

6. **Visualization**:
   - Visualize the architecture of the models using `plot_model`.

## Model Architectures

### Conv3D Model
A simple 3D Convolutional Neural Network with three Conv3D layers followed by MaxPooling3D layers and a dropout layer.

### EfficientNet Model
A more sophisticated model leveraging the EfficientNetB0 architecture, pre-trained on ImageNet, followed by a GlobalAveragePooling3D layer and a dense layer for classification.

## Results

The project highlights the effectiveness of transfer learning in video classification tasks, especially when the dataset size is small. The EfficientNet-based model achieved better accuracy than the Conv3D model.

## Conclusion

This project provides a hands-on example of video classification using deep learning. It compares two different approaches and shows how transfer learning can improve model performance.

## References

- [TensorFlow Video Classification Tutorial](https://www.tensorflow.org/tutorials/load_data/video)
