import cv2
import numpy as np

# Specify the path to your image
file_path = 'ball.jpg'

# Read the image
image = cv2.imread(file_path)

# Check if image reading was successful
if image is None:
    print(f"Error: Could not read the image file '{file_path}'")
    exit(1)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to highlight dark regions
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw contours on
image_with_contours = image.copy()

# Draw contours (draw only external contours)
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

# Display the original image with drawn contours
cv2.imshow('Image with Contours around Dark Regions', image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
