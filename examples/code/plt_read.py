import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

# Define image dimensions
width: int= 4096
height: int = 2168

# Load raw data
with open('../raw_images/Image__2025-05-12__10-47-32.raw', 'rb') as f:
    raw_data: NDArray[np.uint16] = np.fromfile(f, dtype=np.uint16)

# Reshape the data to 2D image
image: NDArray[np.uint16] = raw_data.reshape((height, width))

# Display the image
def show_image(image: NDArray[np.uint16]) -> None:
    plt.imshow(image, cmap='gray')
    plt.title('16-bit Raw Image')
    plt.colorbar()
    plt.show()

show_image(image)