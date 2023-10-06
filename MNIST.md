```python
#Load MNIST

# Commented out IPython magic to ensure Python compatibility.
 %%capture
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from skimage.util import montage
!pip install wandb
import wandb as wb
from skimage.io import imread

def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))

def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))

def plot(x):
    if type(x) == torch.Tensor :
        x = x.cpu().detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap = 'gray')
    ax.axis('off')
    fig.set_size_inches(7, 7)
    plt.show()

def montage_plot(x):
    x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    plot(montage(x))

# #MNIST
train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

#KMNIST
#train_set = datasets.KMNIST('./data', train=True, download=True)
#test_set = datasets.KMNIST('./data', train=False, download=True)

#Fashion MNIST
# train_set = datasets.FashionMNIST('./data', train=True, download=True)
# test_set = datasets.FashionMNIST('./data', train=False, download=True)

X = train_set.data.numpy()
X_test = test_set.data.numpy()
Y = train_set.targets.numpy()
Y_test = test_set.targets.numpy()

X = X[:,None,:,:]/255
X_test = X_test[:,None,:,:]/255

montage_plot(X[125:150,0,:,:])

#Run random y=mx model on MNIST

X= X.reshape(X.shape[0],784)
X_test = X_test.reshape(X_test.shape[0],784)

X = GPU_data(X)
Y = GPU_data(Y)
X_test = GPU_data(X_test)
Y_test = GPU_data(Y_test)

X.shape
X = X.T
X.shape
x = X[:,0:1]
x.shape
M = GPU(np.random.rand(10,784))
y = M@x
batch_size = 64

x = X[:,0:batch_size]

M = GPU(np.random.rand(10,784))

y = M@x

y = torch.argmax(y,0)

torch.sum((y == Y[0:batch_size]))/batch_size

m_best = 0
acc_best = 0

for i in range(100000):

    step = 0.0000000001

    m_random = GPU_data(np.random.randn(10,784))

    m = m_best  + step*m_random

    y = m@X

    y = torch.argmax(y, axis=0)

    acc = ((y == Y)).sum()/len(Y)


    if acc > acc_best:
        print(acc.item())
        m_best = m
        acc_best = acc
```

---

```python
#Load MNIST

# Commented out IPython magic to ensure Python compatibility.
 %%capture
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from skimage.util import montage
!pip install wandb
import wandb as wb
from skimage.io import imread
```

### Loading Libraries and Data

- The code imports necessary libraries, including NumPy, Matplotlib, Torch, and others.

- It installs the "wandb" library if it's not already installed.

- Functions for GPU-related operations and visualization are defined.

---

```python
def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))

def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))
```

### GPU Helper Functions

- `GPU(data)`: This function converts input data to a PyTorch tensor with GPU support for computation and enables gradient computation.

- `GPU_data(data)`: Similar to the previous function, but without requiring gradients.

---

```python
def plot(x):
    if type(x) == torch.Tensor :
        x = x.cpu().detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap = 'gray')
    ax.axis('off')
    fig.set size_inches(7, 7)
    plt.show()
```

### Visualization

- `plot(x)`: This function is for plotting images. It accepts either a NumPy array or a PyTorch tensor and displays it using Matplotlib.

---

```python
def montage_plot(x):
    x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    plot(montage(x))
```

### Visualization (continued)

- `montage_plot(x)`: Creates a montage (grid) of images and uses the `plot` function for visualization.

---

```python
# #MNIST
train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)
```

### Data Preprocessing

- The code loads the MNIST dataset and extracts training and testing data.

---

```python
X = train_set.data.numpy()
X_test = test_set.data.numpy()
Y = train_set.targets.numpy()
Y_test = test_set.targets.numpy()

X = X[:,None,:,:]/255
X_test = X_test[:,None,:,:]/255
```

### Data Preprocessing (continued)

- The pixel values are normalized to the range [0, 1] by dividing by 255.

---

```python
montage_plot(X[125:150,0,:,:])
```

### Visualization (Montage)

- A sample montage of images from the training dataset is plotted.
![montage plot of numbers](https://github.com/azanicareer/azanicareer/blob/main/mnist_numbers.png)
---

```python
#Run random y=mx model on MNIST

X= X.reshape(X.shape[0],784)
X_test = X_test.reshape(X_test.shape[0],784)

X = GPU_data(X)
Y = GPU_data(Y)
X_test = GPU_data(X_test)
Y_test = GPU_data(Y_test)

X.shape
X = X.T
X.shape
x = X[:,0:1]
x.shape
M = GPU(np.random.rand(10,784))
y = M@x
batch_size = 64
```

### Random Weight Model

- The code reshapes the input data to a flat vector of size 784 (28x28) for each image.

- GPU tensors are created for input data, labels, and testing data.

- A random weight matrix `M` with dimensions (10, 784) is initialized on the GPU.

- The code computes predictions `y` for a batch of input data `x` using random weights.

---

```python
x = X[:,0:batch_size]

M = GPU(np.random.rand(10,784))

y = M@x

y = torch.argmax(y,0)

torch.sum((y == Y[0:batch_size]))/batch_size
```

### Random Weight Model (continued)

- Input data is sliced into a batch of size `batch_size`.

- Another random weight matrix `M` is created.

- Predictions `y` are computed for the batch, and accuracy on this batch is calculated.

---

```python
m_best = 0
acc_best = 0

for i in range(100000):

    step = 0.0000000001

    m_random = GPU_data(np.random.randn(10,784))

    m = m_best  + step*m_random

    y = m@X

    y = torch.argmax(y, axis=0)

    acc = ((y == Y)).sum()/len(Y)


    if acc > acc_best:
        print(acc.item())
        m_best = m
        acc_best = acc
```

### Training Loop

- The code enters a training loop with 100,000 iterations.

- In each iteration, a small random change is made to the weight matrix `M`.

- Predictions `y` are calculated for the entire training dataset using the updated `M`.

- The accuracy of the model on the entire training dataset is computed.

- If the new accuracy is better than the previous best accuracy, the model and accuracy are updated.

---

The purpose of this code is to demonstrate a simple classification task using random weights and a training loop. It continuously updates the model's weights to achieve better accuracy on the MNIST dataset.
