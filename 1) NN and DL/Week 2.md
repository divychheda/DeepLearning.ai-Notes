Week 2

## Neural Networks Basics

> Learn to set up a machine learning problem with a neural network mindset. Learn to use vectorization to speed up your models.

### Binary classification

- Mainly he is talking about how to do a logistic regression to make a binary classifier.(Binary classification where ans is a '1 or 0', 'Yes or No' etc)
- He talked about an example of stating if the current image contains a cat or not.
    <img src="../../_resources/6ffb200e44244a6c8efba2fd6bc68919.png" alt="e285aab7e03251da7c62b0517de1bc97.png" width="731" height="408" class="jop-noMdConv">
- Here are some notations:
    - `M is the number of training vectors`
    - `Nx is the size of the input vector`
    - `Ny is the size of the output vector`
    - `X(1) is the first input vector`
    - `Y(1) is the first output vector`
    - `X = [x(1) x(2).. x(M)]`
    - `Y = [(y(1) y(2).. y(M))]`
        <img src="../../_resources/90222a229b8e4f7bb199dd1aa54801d7.png" alt="14191cda77334a5e78adad3ed88e835f.png" width="744" height="422" class="jop-noMdConv">

### Logistic regression

- Algorithm is used for classification or probbility.
- Equations:
    - In binary classification `Y` has to be between `0` and `1`.
    - If x is a vector: `y = w(transpose)x + b`
    - If we need y to be in between 0 and 1 , let `y-hat` = probability that y = 1 for input x
    - `y-hat = sigmoid(w(transpose)x + b)`
    - where `sigmoid(z) = 1/(1+e^-z)`
- In the equation `w` is a `nx` dimensional vector and `b` is a real number.<img src="../../_resources/4c93753972c143859e7d75df861ef250.png" alt="d3348ce935a55d8b4ef974026b518614.png" width="845" height="462" class="jop-noMdConv">

### Logistic regression cost function

- First loss function would be the square root error: `L(y',y) = 1/2 (y' - y)^2`
    - But we won't use this notation because it leads us to optimization problem which is non convex, means it contains local optimum points.
- This is the function that we will use: `L(y',y) = - (y*log(y') + (1-y)*log(1-y'))`
- To explain the last function lets see:
    - if `y = 1` ==\> `L(y',1) = -log(y')` ==\> we want `y'` to be the largest ==> `y`' biggest value is 1
    - if `y = 0` ==\> `L(y',0) = -log(1-y')` ==\> we want `1-y'` to be the largest ==> `y'` to be smaller as possible(close to 0) because it can only has 1 value.
- Then the Cost function will be: `J(w,b) = (1/m) * Sum(L(y'[i],y[i]))`
- The loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set.

<img src="../../_resources/546cfbcea70b4f1db3c53a1339e63d57.png" alt="ae38dea27c4b5ff7f15ed80f18a553e1.png" width="876" height="494" class="jop-noMdConv">

### Gradient Descent

- We want to predict `w` and `b` that minimize the cost function.
    
- Our cost function is convex.
    
- First we initialize `w` and `b` to 0,0 or initialize them to a random value in the convex function and then try to improve the values the reach minimum value.
    
- In Logistic regression people always use 0,0 instead of random.
    
- <img src="../../_resources/f7f39ea60797471d84cfe8e6a8c62176.png" alt="09c1f77bc7c24f6876994a82adb89518.png" width="751" height="254" class="jop-noMdConv">
- The gradient decent algorithm repeats: `w = w - alpha * dw` where `alpha is the learning rate` and `dw` is the derivative of `w`.
    
- The actual equations we will implement:
    
    - `w = w - alpha * d(J(w,b) / dw)` (how much the function slopes in the w direction/ on w axis)
    - `b = b - alpha * d(J(w,b) / db)` (how much the function slopes in the b direction/ on b axis)

<img src="../../_resources/d3a28a8e971c43fdb9c9bc4e9d31defc.png" alt="30bab276a7eedff37b2395cf9c81366a.png" width="731" height="408" class="jop-noMdConv">

### Computation graph

- Its a graph that organizes the computation from left to right.[![](../../_resources/a7b52d78ff0143f6ad1616c49a694552.png)](https://github.com/mbadry1/DeepLearning.ai-Summary/blob/master/1-%20Neural%20Networks%20and%20Deep%20Learning/Images/02.png)
- To calculate function we go from left to right, but to calculate derivatives for optimizing our cost function we will go from right to left (backpropagation), opposite to blue arrows.

### Derivatives with a Computation Graph

- Calculus chain rule says: If `x -> y -> z` (x effect y and y effects z) Then `d(z)/d(x) = d(z)/d(y) * d(y)/d(x)`<img src="../../_resources/c94b0e1ccf1f478d842510a9cae7ebae.png" alt="e1ebf5c083fc6a8e315c279596a4b57c.png" width="731" height="403" class="jop-noMdConv">
- `dvar` is notation used to represent `dFinal-Output-var/dvar` in python codes, in our case we can write variable name of `dJ/dv` as `dv`.
    <img src="../../_resources/eeb026a8d99f4318a2024810d3dbaea7.png" alt="87de2bfd33c2adc9971da1fc611dacce.png" width="737" height="408" class="jop-noMdConv">

### Logistic Regression Gradient Descent

- In the video he discussed the derivatives of gradient decent example for one sample with two features `x1` and `x2`.[![](../../_resources/2b51384f159a42e1b6fcf1bb6fe85e55.png)](https://github.com/mbadry1/DeepLearning.ai-Summary/blob/master/1-%20Neural%20Networks%20and%20Deep%20Learning/Images/04.png)

### Gradient Descent on m Examples

- Lets say we have these variables:
    
    ```
        X1                  Feature
        X2                  Feature
        W1                  Weight of the first feature.
        W2                  Weight of the second feature.
        B                   Bias - Logistic Regression parameter.
        M                   Number of training examples
        Y(i)                Expected output of i
    ```
    
- So we have:
    
- Then from right to left we will calculate derivations compared to the result:
    
    ```python3
        d(a)  = d(l)/d(a) = -(y/a) + ((1-y)/(1-a))
        d(z)  = d(l)/d(z) = a - y
        d(W1) = X1 * d(z)
        d(W2) = X2 * d(z)
        d(B)  = d(z) 
    ```
    
- From the above we can conclude the logistic regression pseudo code:
    
    ```python3
     J = 0; dw1 = 0; dw2 = 0; db = 0;     
        w1 = 0; w2 = 0; b=0;							# Weights
        for i = 1 to m
            # Forward pass
            z(i) = W1*x1(i) + W2*x2(i) + b
            a(i) = Sigmoid(z(i))
            J += (Y(i)*log(a(i)) + (1-Y(i))*log(1-a(i)))
    
            # Backward pass
            dz(i) = a(i) - Y(i)
            dw1 += dz(i) * x1(i)
            dw2 += dz(i) * x2(i)
            db  += dz(i)
        J /= m
        dw1/= m
        dw2/= m
        db/= m
    
        # Gradient descent
        w1 = w1 - alpha * dw1
        w2 = w2 - alpha * dw2
        b = b - alpha * db 
    ```
    
- The gradient descent steps should run for some iterations to minimize error.
    
- So there will be two inner loops to implement the logistic regression.
    
- Vectorization is so important on deep learning to reduce loops. In the last code we can make the whole loop in one step using vectorization!
    

### Vectorization

- Deep learning shines when the dataset are big. However for loops will make you wait a lot for a result. Thats why we need vectorization to get rid of some of our for loops.
- NumPy library (dot) function is using vectorization by default.
- The vectorization can be done on CPU or GPU thought the SIMD operation (single instruction multiple data). But its faster on GPU.
- Whenever possible avoid for loops. Loops take upto 200-300 times more time than vectorization.
- Most of the NumPy library methods are vectorized.

### Vectorization of Logistic regression

- We will implement Logistic Regression using one for loop then without any for loop.<img src="../../_resources/27074f5513c14c7785ef74abb76c75b2.png" alt="c8235a29ba04676c6b1d094c8143442c.png" width="715" height="403" class="jop-noMdConv">
    
- If there were more features than 2, we would need another loop to go over all the dwj.
    
- This can be eliminated by taking dw as a numpy array and doing calcs as shown above.
    

<img src="../../_resources/f9fd3ea2fcdd4285bc499a898dde0d23.png" alt="03c9180740631e619ed4bb67c21c8c04.png" width="744" height="432" class="jop-noMdConv">

- As an input we have a matrix `X` and its `[Nx, m]` and a matrix `Y` and its `[Ny, m]`.
    
- We will then compute at instance `[z1,z2...zm] = W' * X + [b,b,...b]`. This can be written in python as:
    
    ```
     	Z = np.dot(W.T,X) + b    # Vectorization, then broadcasting, Z shape is (1, m)
      	A = 1 / 1 + np.exp(-Z)   # Vectorization, A shape is (1, m) 
    ```
    

<img src="../../_resources/6d57f23e15284c5399591209a9dce948.png" alt="51e763ef02fb280ccd0c9a723fd095e5.png" width="733" height="415" class="jop-noMdConv">

Vectorizing Logistic Regression's Gradient Output:

```
 	dz = A - Y                  # Vectorization, dz shape is (1, m)
  	dw = np.dot(X, dz.T) / m    # Vectorization, dw shape is (Nx, 1)
  	db = dz.sum() / m           # Vectorization, dz shape is (1, 1) 
```

### Broadcasting

- In NumPy, `obj.sum(axis = 0)` sums the columns while `obj.sum(axis = 1)` sums the rows.
    
- In NumPy, `obj.reshape(1,4)` changes the shape of the matrix by broadcasting the values.
    
- Reshape is cheap in calculations so put it everywhere you're not sure about the calculations.
    
- Broadcasting works when you do a matrix operation with matrices that doesn't match for the operation, in this case NumPy automatically makes the shapes ready for the operation by broadcasting the values.
    
- In general principle of broadcasting. If you have an (m,n) matrix and you add(+) or subtract(-) or multiply(*) or divide(/) with a (1,n) matrix, then this will copy it m times into an (m,n) matrix. The same with if you use those operations with a (m , 1) matrix, then this will copy it n times into (m, n) matrix. And then apply the addition, subtraction, and multiplication of division element wise.
    

<img src="../../_resources/9605c7d0922749c09cf5703dd9d18673.png" alt="8d320bd71f8b3a59acc8abb1553356c4.png" width="675" height="376" class="jop-noMdConv">

<img src="../../_resources/5c731c4badc54588879808aea0953e92.png" alt="f918e1f5c1c3584b448a2b4b0113f336.png" width="387" height="212" class="jop-noMdConv"><img src="../../_resources/bd5fab8d746e44789e9217708340e041.png" alt="57cb026c64552cee2b3bb9c50bd7c571.png" width="423" height="215" class="jop-noMdConv">

### Python Stuff

- Some tricks to eliminate all the strange bugs in the code:
    
    - If you didn't specify the shape of a vector, it will take a shape of `(m,)` and the transpose operation won't work. You have to reshape it to `(m, 1)`
    - Try to not use the rank one matrix in ANN
    - Don't hesitate to use `assert(a.shape == (5,1))` to check if your matrix shape is the required one.
    - If you've found a rank one matrix try to run reshape on it.