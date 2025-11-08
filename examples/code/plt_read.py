import numpy as np
import matplotlib.pyplot as plt

# Define image dimensions
width = 4096
height = 2168

# Load raw data
with open('../raw_images/Image__2025-05-12__10-47-32.raw', 'rb') as f:
    raw_data = np.fromfile(f, dtype=np.uint16)

# Reshape the data to 2D image
image = raw_data.reshape((height, width))

# Display the image
plt.imshow(image, cmap='gray')
plt.title('16-bit Raw Image')
plt.colorbar()
plt.show()