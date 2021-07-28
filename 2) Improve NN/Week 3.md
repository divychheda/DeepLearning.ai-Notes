Week 3

### Tuning process

- We need to tune our hyperparameters to get the best out of them.
- Hyperparameters importance are (as for Andrew Ng):
    1.  Learning rate.
    2.  Momentum beta.
    3.  Mini-batch size.
    4.  No. of hidden units.
    5.  No. of layers.
    6.  Learning rate decay.
    7.  Regularization lambda.
    8.  Activation functions.
    9.  Adam `beta1`, `beta2` & `epsilon`.

<img src="../../_resources/1ce4edd0b51d4328b623cb8faf5dc1cf.png" alt="ac04e2a8dc4d6826266c99a310eccc92.png" width="754" height="545">

**Order of importance** -\> red > yellow > purple (adam's beta1, beta2 is usually kept fixed to the values mentioned)

- One of the older ways to tune was to sample a grid with `N` hyperparameter settings and then try all settings combinations on your problem.
- **Try random values: don't use a grid.**
- You can use `Coarse to fine sampling scheme`:
    - When you find some hyperparameters values that give you a better performance - zoom into a smaller region around these values and sample more densely within this space.
- These methods can be automated.

### Using an appropriate scale to pick hyperparameters

- Let's say you have a specific range for a hyperparameter from "a" to "b". It's better to search for the right ones using the logarithmic scale rather then in linear scale:
    - Calculate: `a_log = log(a) # e.g. a = 0.0001 then a_log = -4`
    - Calculate: `b_log = log(b) # e.g. b = 1 then b_log = 0`
    - Then:
        
        ```
        r = (a_log - b_log) * np.random.rand() + b_log
        # In the example the range would be from [-4, 0] because rand range [0,1)
        result = 10^r 
        ```
        
        It uniformly samples values in log scale from \[a,b\].
- If we want to use the last method on exploring on the "momentum beta":
    - Beta best range is from 0.9 to 0.999.
    - You should search for `1 - beta in range 0.001 to 0.1 (1 - 0.9 and 1 - 0.999)` and the use `a = 0.001` and `b = 0.1`. Then:
        
        ```
        a_log = -3
        b_log = -1
        r = (a_log - b_log) * np.random.rand() + b_log
        beta = 1 - 10^r   # because 1 - beta = 10^r 
        ```
- 