from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('mnist_model.keras')
print("Model loaded successfully.")

# Load and preprocess the test image (assuming it's 28x28 grayscale)
img = image.load_img('image.png', target_size=(
    28, 28), color_mode='grayscale')

# Convert to array and check if inversion is needed (MNIST expects white digits on black background)
img_array = image.img_to_array(img)
if np.mean(img_array) > 127:  # If mostly white, invert (assuming black digits on white)
    img_array = 255 - img_array
    print("Image inverted for better prediction.")

img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)
pred = np.argmax(predictions)
confidence = np.max(predictions)

print(f"Predicted digit: {pred}")
print(f"Confidence: {confidence:.4f}")
# Show probabilities for all digits
print(f"All probabilities: {predictions[0]}")

# Display the image
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {pred} (Confidence: {confidence:.2f})")
plt.show()
