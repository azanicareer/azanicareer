```
import numpy as np
import cv2
import urllib
import requests
from matplotlib import pyplot as plt

# Load an RGB image from a URL
image_url = "https://static1.cbrimages.com/wordpress/wp-content/uploads/2021/08/Jiji.jpg"  # Replace with the URL of your image
response = requests.get(image_url)
img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
image = cv2.imdecode(img_array, -1)

# Resize the image to 224x224
resized_image = cv2.resize(image, (224, 224))

# Show the grayscale copy
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()
![jiji in grey](https://github.com/azanicareer/azanicareer/blob/main/jiji.png)
---

# Convolve with 10 random filters
num_filters = 10
filter_size = 5  # You can adjust the filter size as needed

for i in range(num_filters):
    random_filter = np.random.randn(filter_size, filter_size)
    filtered_image = cv2.filter2D(resized_image, -1, random_filter)
    
    # Show the filter
    plt.subplot(2, num_filters, i + 1)
    plt.imshow(random_filter, cmap='gray')
    plt.title(f'Filter {i + 1}')
    plt.axis('off')
    
    # Show the feature map
    plt.subplot(2, num_filters, i + num_filters + 1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Ftr Map {i + 1}')
    plt.axis('off')

plt.tight_layout()
plt.show()
```




```markdown
# Convolution and Filter Visualization in Python

This Python code performs the following tasks:
1. Loads an RGB image from a URL.
2. Resizes the image to 224x224 pixels.
3. Converts the image to grayscale.
4. Applies 10 random filters to the image and displays both the filters and their resulting feature maps.

## Getting Started

Before running the code, make sure you have the required libraries installed. You can install them using pip:

```bash
pip install numpy opencv-python-headless matplotlib requests
```

Make sure to replace the `image_url` variable with the URL of the image you want to process.

## Code Explanation

### Load an RGB Image

```python
# Load an RGB image from a URL
image_url = "https://static1.cbrimages.com/wordpress/wp-content/uploads/2021/08/Jiji.jpg"  # Replace with the URL of your image
response = requests.get(image_url)
img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
image = cv2.imdecode(img_array, -1)
```

This section fetches an image from the specified URL and decodes it into an RGB image using OpenCV.

### Resize the Image

```python
# Resize the image to 224x224
resized_image = cv2.resize(image, (224, 224))
```

The code resizes the image to 224x224 pixels.

### Grayscale Conversion

```python
# Show the grayscale copy
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')
```

This part converts the resized image to grayscale and displays it.
![jiji in grayscale](https://github.com/azanicareer/azanicareer/blob/main/jiji.png)
### Convolve with 10 Random Filters

```python
# Convolve with 10 random filters
num_filters = 10
filter_size = 5  # You can adjust the filter size as needed

for i in range(num_filters):
    random_filter = np.random.randn(filter_size, filter_size)
    filtered_image = cv2.filter2D(resized_image, -1, random_filter)
    
    # Show the filter and feature map
    # ...
```

Here, the code applies 10 random filters of a specified size to the resized image and displays both the filters and their corresponding feature maps.

The `random_filter` is a 2D array of random values, and `cv2.filter2D` applies this filter to the image. It then displays the filter and feature map using Matplotlib.

![jiji in filters](https://github.com/azanicareer/azanicareer/blob/main/jiji_in_filters.png)

## Running the Code

After setting up your environment and replacing the `image_url` with your desired image URL, you can run the code in a Python environment like Jupyter Notebook or Google Colab.

Enjoy experimenting with different images and filter sizes to observe the effects of convolution on image processing.
```

Now, the explanation includes the link to your image: `"https://static1.cbrimages.com/wordpress/wp-content/uploads/2021/08/Jiji.jpg"`.
