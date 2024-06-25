import cv2
import numpy as np

# Load the image
image = cv2.imread('ball.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to get a binary image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the largest contour is the ball
ball_contour = max(contours, key=cv2.contourArea)

# Create a mask for the ball
mask = np.zeros_like(gray)
cv2.drawContours(mask, [ball_contour], -1, 255, thickness=cv2.FILLED)

# Apply the mask to the original image to isolate the ball
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Convert the masked image to grayscale
masked_gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to the masked grayscale image
_, masked_binary = cv2.threshold(masked_gray, 127, 255, cv2.THRESH_BINARY)

# Find contours in the masked binary image
masked_contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Ensure that we have exactly two contours (black and white halves)
if len(masked_contours) != 2:
    raise ValueError("The image does not contain exactly two halves.")

# Calculate the moments to find the centers of the contours
M1 = cv2.moments(masked_contours[0])
M2 = cv2.moments(masked_contours[1])

# Calculate the x, y coordinates of the centers
cX1 = int(M1["m10"] / M1["m00"])
cY1 = int(M1["m01"] / M1["m00"])
cX2 = int(M2["m10"] / M2["m00"])
cY2 = int(M2["m01"] / M2["m00"])

# Draw contours and centers on the image
cv2.drawContours(image, masked_contours, -1, (0, 255, 0), 2)
cv2.circle(image, (cX1, cY1), 7, (255, 0, 0), -1)
cv2.circle(image, (cX2, cY2), 7, (0, 0, 255), -1)

# Determine the direction vector
direction_vector = (cX2 - cX1, cY2 - cY1)

# Draw the direction vector on the image
cv2.arrowedLine(image, (cX1, cY1), (cX2, cY2), (255, 255, 0), 2)

# Show the image
cv2.imshow('Direction of North Pole', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
gfortran openexr libatlas-base-dev python3-dev python3-numpy \
libtbb2 libtbb-dev libdc1394-22-dev

