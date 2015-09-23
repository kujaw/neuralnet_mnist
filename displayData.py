def display():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    # Load data from matlab file or ...
    data = loadmat('ex4data1.mat')
    X, y = data['X'], data['y']
    # ... load data from .csv file
    #data = np.loadtxt('ex4data1.csv', delimiter=',')
    #X, y = data[:,:-1], data[:,-1]

    # Randomly choose 100 images (rows) for display
    XShuffled = X
    np.random.shuffle(XShuffled)
    XDisplay = XShuffled[0:100]

    # Initialize variables needed for creating the image
    m, n = np.shape(XDisplay)
    exampleWidth = int(np.ceil(np.sqrt(n)))
    exampleHeight = exampleWidth
    imageRows = int(np.ceil(np.sqrt(m)))
    imageCols = imageRows
    imageNum = 0

    imgMatrix = np.zeros((exampleWidth*imageRows, exampleHeight*imageCols))

    # Function that reshapes vectors, merges them to create bigger array of 100 digits
    for i in range(imageCols):
        for j in range(imageRows):
            if imageNum >= m:
                break
            example = XDisplay[imageNum,:].reshape(exampleWidth, exampleWidth)

            imgMatrix[j*exampleWidth:j*exampleWidth+exampleWidth,
            i*exampleHeight:i*exampleHeight+exampleHeight] = example[:,:]

            imageNum += 1

    # Rotate & display digits array
    plt.imshow(np.flipud(np.rot90(imgMatrix)), cmap="gray")
    plt.title('MNIST digits sample data visualization')
    plt.show()