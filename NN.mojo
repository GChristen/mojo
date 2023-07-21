alias type: DType = DType.float32    
alias netls: Int = simdwidthof[type]()

def linear_forward(A: Tensor[type], W: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    let Z = W.dot(A) + b
    #assert(Z.shape == (W.shape[0], A.shape[1]))
    #cache = (A, W, b)    
    return Z

#Test writing a W.X+b op, input size 2, hum_hidden = 32
let feature_size = 2
let num_samples = 10

let W1_size = 32
let W2_size = 32

let input_data = Tensor[type].arange(__idx(feature_size, num_samples))

let W1 = Tensor[type].arange(__idx(W1_size, feature_size))*0.01
let W2 = Tensor[type].arange(__idx(W2_size, W1_size))*0.01
let b1 = Tensor[type](W1_size,1)
let b2 = Tensor[type](W2_size,1)

#first output should be 32,10 (32 features for the 10 samples
let z = linear_forward(input_data, W1, b1)