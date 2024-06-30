import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image using PIL and convert to a numpy array
image_path = 'ball1.tiff'
image = Image.open(image_path)
image_np = np.array(image)

# Apply Hough Circle Transform to detect large circles
circles = cv2.HoughCircles(image_np, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                           param1=50, param2=30, minRadius=100, maxRadius=500)

# Convert image to RGB for visualization
image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

# Draw detected circles on the image
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(image_rgb, (x, y), r, (255, 0, 0), 4)  # Drawing circles in blue
        cv2.rectangle(image_rgb, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

# Save the image with detected circles
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title("Detected Large Circles")
plt.axis('off')
plt.savefig('detected_circles.png')  # Save the figure
plt.close()

# Isolate the first detected circle for further analysis
if circles is not None:
    x, y, r = circles[0]
    
    # Draw only the detected circle of interest
    isolated_circle = image_rgb.copy()
    cv2.circle(isolated_circle, (x, y), r, (255, 0, 0), 4)
    cv2.rectangle(isolated_circle, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # Extract the region of the circle
    mask = np.zeros_like(image_np)
    cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)
    circle_region = cv2.bitwise_and(image_np, image_np, mask=mask)

    # Save the isolated circle and its region
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(isolated_circle)
    plt.title("Isolated Circle")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(circle_region, cmap='gray')
    plt.title("Circle Region")
    plt.axis('off')

    plt.savefig('isolated_circle_and_region.png')  # Save the figure
    plt.close()
else:
    print("No large circles detected.")
