# My_neural_network
Neural Network from scratch in Python

If we call:

- Output: $Y$
- Input: $X$
- Weight: $W$
- Bias: $B$
- Activation function: $f(z = W\cdot X + B)$
- Cost function: $C$

## Forward Propagation
In forward propagation, for each layer:

$X$ =  $\rightarrow$ $\boxed{\text{layer}}$ $\rightarrow$ $Y$

$Y$ = f($W$ $\cdot$ $X$ + $B$)



The output for one layer is the input for the next layer.

**At the end of the forward propagation, we can calculate the error as the derivative of the cost function with respect to the output for the neural network model output. We will call it the output gradient $\frac{\partial C}{\partial Y}$ .** 

## Backward Propagation
In backward propagation, for each layer:

\begin{aligned}
\displaystyle\frac{\partial C}{\partial X} &\leftarrow \boxed{\pmb{layer}} \leftarrow \frac{\partial C}{\partial Y} \\[10pt]
\frac{\partial C}{\partial z} &= f'(z) \odot \frac{\partial C}{\partial Y} \\[10pt]
\frac{\partial C}{\partial w} &= \frac{\partial C}{\partial z} \cdot X^T \\[10pt]
\frac{\partial C}{\partial B} &= \frac{\partial C}{\partial Y} \\[10pt]
\frac{\partial C}{\partial X} &= w^T \cdot \frac{\partial C}{\partial z} \quad (\text{the new output gradient for the previous layer})
\end{aligned} 

## For example:

![%D9%86%D8%B5%20%D9%81%D9%82%D8%B1%D8%AA%D9%83.png](attachment:%D9%86%D8%B5%20%D9%81%D9%82%D8%B1%D8%AA%D9%83.png)


If we have 2x1 input and 3 hidden layers, here is how to do the forward and backward propagation for the output layer and the rest layers will be the same:


1. Forward pass:
\begin{aligned}
z^{(4)} &= w^{(4)} \cdot a^{(3)} + b^{(4)} \\[5pt]
a^{(4)} &= f(z^{(4)}) \
\end{aligned}

2. The derivative of the cost function (output gradient) $\frac{\partial C}{\partial a^{(4)}}$.


3. Backward pass:
\begin{aligned}
\frac{\partial C}{\partial z^{(4)}} &= \frac{\partial a^{(4)}}{\partial z^{(4)}} \cdot \frac{\partial C}{\partial a^{(4)}} \\
&= f'(z^{(4)}) \odot \frac{\partial C}{\partial a^{(4)}} \\[10pt]
\frac{\partial C}{\partial w^{(4)}} &= \frac{\partial z^{(4)}}{\partial w^{(4)}} \cdot \frac{\partial C}{\partial z^{(4)}} \\
&= \frac{\partial C}{\partial z^{(4)}} \cdot (a^{(3)})^{T} \\[10pt]
\frac{\partial C}{\partial a^{(3)}} &= \frac{\partial z^{(4)}}{\partial a^{(3)}} \cdot \frac{\partial C}{\partial z^{(4)}} \\
&= (w^{(4)})^{T} \cdot \frac{\partial C}{\partial z^{(4)}} \\
\end{aligned}

For the 3rd hidden layer we will take the $ \frac{\partial C}{\partial a^{(3)}} $ as the output gradient.
