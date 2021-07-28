Week 1

# Week 1

## What is NN?

<img src="../../_resources/87adc7e05914443d8358ca2e4bd8cda8.png" alt="57c9608207fcbcdaa7d3a26c3aee21d0.png" width="763" height="409">

1.  That function (pointed by arrow) is a single Neuron Neural Network, more complex ones have several neurons/nodes stacked together.
2.  The graph/function is a ReLU, blue line denoting the equation to predict price of house.

<img src="../../_resources/037d3eca31e74720a6b9cd45fcfa212f.png" alt="f776aa4547719586daed74e3c95929e6.png" width="765" height="384">
A larger NN formed by stacking more of the functions/nodes. Each of these is a ReLU.
Its basically a mapping from x (input) to y (output).

<img src="../../_resources/ff6a7220f73a4cfa86ee4e24c1a0bc5d.png" alt="1b49b9f2834b641fb4b609776e06a697.png" width="902" height="453">
Here every input feature is connected to every node of inner layer, hence they are densely connected.

## Supervised learning with neural networks

Different types of neural networks for supervised learning which includes:

- CNN or convolutional neural networks (Useful in computer vision)
    
- RNN or Recurrent neural networks (Useful in Speech recognition or NLP as data is a sequence)
    
- Standard NN (Useful for Structured data)
    
- Hybrid/custom NN or a Collection of NNs types (eg self-driving car)
    
- Structured data is like the databases and tables.
    
- Unstructured data is like images, video, audio, and text.
    
- Structured data gives more money because companies relies on prediction on its big data.
    
- slightly more difficult to perform deep learning on unstructured data.
    

## Why is deep learning taking off?

- Deep learning is taking off for 3 reasons:
    
    1.  Data:
        - Using this image we can conclude:
            [<img src="../../_resources/6f3ee601e76a4a508b4e3dffd084f8f7.png" alt="" width="889" height="427">](https://github.com/mbadry1/DeepLearning.ai-Summary/blob/master/1-%20Neural%20Networks%20and%20Deep%20Learning/Images/11.png)
            `Note: amount of data is denoted by 'm'.`
        - For small data NN can perform as well as Linear regression or SVM (Support vector machine)
        - For big data a small NN is better that SVM
        - For big data a big NN is better than small NN.
        - We will have a lot of data because the world is using the computers and softwares increasingly, hence deep learning will grow.
            - Mobiles
            - IOT (Internet of things)
    2.  Computation:
        - GPUs.
        - Powerful CPUs.
        - Distributed computing.
        - ASICs
    3.  Algorithm:
        1.  Creative algorithms has appeared that changed the way NN works.
            - For example using RELU function is so much better than using SIGMOID function in training a NN because it helps with the vanishing gradient problem. (for greater x, gradient becomes almost 0 hence learning very slow)