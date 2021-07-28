Week 4

## Deep Neural Networks

> Understand the key computations underlying deep learning, use them to build and train deep neural networks, and apply it to computer vision.

### [](https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/1-%20Neural%20Networks%20and%20Deep%20Learning#deep-l-layer-neural-network)Deep L-layer neural network

- Shallow NN is a NN with one or two layers.
- Deep NN is a NN with three or more layers.
- We will use the notation`L`to denote the number of layers in a NN.
- `n[l]`is the number of neurons in a specific layer`l`.
- `n[0]`denotes the number of neurons input layer.`n[L]`denotes the number of neurons in output layer.
- `g[l]`is the activation function.
- `a[l] = g[l](z[l])`
- `w[l]`weights is used for`z[l]`
- `x = a[0]`,`a[l] = y'`

### Forward Propagation in a Deep Network

- Forward propagation general rule for one input:
    
    ```python
    z[l] = W[l]a[l-1] + b[l]
    a[l] = g[l](a[l])
    ```
    
- Forward propagation general rule for`m`inputs:
    
    ```python
    Z[l] = W[l]A[l-1] + B[l]
    A[l] = g[l](A[l]) 
    ```
    
- We need a for loop to go over all the layers ( l goes from 1 - L)
    
- We can't compute the whole layers forward propagation without a for loop so its OK to have a for loop here.
    

### Getting your matrix dimensions right

- Dimension of`W`is`(n[l],n[l-1])`. Can be thought by right to left.
- Dimension of`b`is`(n[l],1)`
- `dw`has the same shape as`W`, while`db`is the same shape as`b`
- Dimension of`Z[l],``A[l]`,`dZ[l]`, and`dA[l]`is`(n[l],m)`

### Why deep representations?

- Deep NN makes relations with data from simpler to complex. In each layer it tries to make a relation with the previous layer. E.g.
- 1.  Face recognition application:
        
        - Image ==> Edges ==> Face parts ==> Faces ==> desired face
- 2.  Audio recognition application:
        
        - Audio ==> Low level sound features like (sss,bb) ==> Phonemes ==> Words ==> Sentences
- Neural Researchers think that deep neural networks "think" like brains (simple ==> complex)
- Circuit theory and deep learning:
    - [![](../../_resources/7d2929c1bc2a434baea76d1d934b5302.png)](https://github.com/mbadry1/DeepLearning.ai-Summary/blob/master/1-%20Neural%20Networks%20and%20Deep%20Learning/Images/07.png)
- When starting on an application don't start directly by dozens of hidden layers. Try the simplest solutions (e.g. Logistic Regression), then try the shallow neural network and so on.

### Building blocks of deep neural networks

- Forward and back propagation for a layer l:
    - [<img src="../../_resources/af5ff5325ef947b69abaf714bde00a41.png" alt="Untitled" width="765" height="264" class="jop-noMdConv">](https://github.com/mbadry1/DeepLearning.ai-Summary/blob/master/1-%20Neural%20Networks%20and%20Deep%20Learning/Images/10.png)
- Deep NN blocks:
    - [![](../../_resources/719727126a304eeab18975db0d8443e0.png)](https://github.com/mbadry1/DeepLearning.ai-Summary/blob/master/1-%20Neural%20Networks%20and%20Deep%20Learning/Images/08.png)

### Forward and Backward Propagation

- Pseudo code for forward propagation for layer l:
    
    ```
    Input  A[l-1]
    Z[l] = W[l]A[l-1] + b[l]
    A[l] = g[l](Z[l])
    Output A[l], cache(Z[l]) 
    ```
    
- Pseudo code for back propagation for layer l:
    
    ```python
    Input da[l], Caches
    dZ[l] = dA[l] * g'[l](Z[l])  or  A[l] - Y
    dW[l] = np.dot(dZ[l],A[l-1].T) / m
    db[l] = np.sum(dZ[l],axis=1, keepdims=True)/m                # Dont forget axis=1, keepdims=True
    dA[l-1] = np.dot(w[l].T , dZ[l])           
    Output dA[l-1], dW[l], db[l]
    ```
    

<img src="../../_resources/cd2dac15b63e4c83a4fddcaa489f6f3c.png" alt="5b6d1d34ad57b160c956185b3f27f038.png" width="730" height="408" class="jop-noMdConv">

### Parameters vs Hyperparameters

- Main parameters of the NN is `W` and `b`
- Hyper parameters (parameters that control the algorithm) are like:
    - Learning rate.
    - Number of iteration.
    - Number of hidden layers `L`.
    - Number of hidden units `n`.
    - Choice of activation functions.