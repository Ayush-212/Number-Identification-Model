# MNIST Digit Recognition with CNN

A deep learning project using a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) from the MNIST dataset. This project demonstrates image classification with Keras/TensorFlow.

## 🚀 Overview
This project trains a CNN on the MNIST dataset and provides a script to test custom digit images. The model achieves high accuracy on standard MNIST data and can predict digits from user-provided images.

## 🏗️ Model Architecture
The CNN includes:
- **2 Convolutional Layers** with ReLU activation and max pooling for feature extraction.
- **Flatten Layer** to transition to dense layers.
- **2 Dense Layers** (64 units with ReLU, 10 units with Softmax for classification).
- Trained with Adam optimizer and sparse categorical crossentropy loss.

**Note**: The model is trained on MNIST data with white digits on a black background. Test images are automatically inverted if they have a white background to match this format.

## 📁 Files
- `MNSITcode.py`: Trains the model on MNIST and saves it as `mnist_model.keras`.
- `Test_Model.py`: Loads the model and predicts digits from `image.png`.
- `MNIST_Complete.py`: Combined script for training and testing.
- `mnist_model.keras`: Saved trained model (generated after running `MNSITcode.py`).

## 🛠️ Setup and Usage
1. **Install Dependencies**:
   ```bash
   pip install tensorflow numpy matplotlib pillow
   ```

2. **Train the Model**:
   ```bash
   python MNSITcode.py
   ```
   This trains for 10 epochs with validation and saves the model.

3. **Test on Custom Image**:
   - Place a 28x28 grayscale image named `image.png` in the directory.
   - Run:
     ```bash
     python Test_Model.py
     ```
   - The script handles image inversion if needed and outputs the prediction with confidence.

## 📊 Results
- **Training Accuracy**: ~90-99% on MNIST test set.
- **Custom Prediction**: Works best with clear, centered digits. Includes automatic preprocessing for color inversion.

## 🔧 Notes
- Ensure `image.png` and `image2.png` is grayscale; the script resizes and preprocesses it.
- For low-confidence predictions, check image quality or retrain with more epochs.
- Model uses `.keras` format for compatibility.
