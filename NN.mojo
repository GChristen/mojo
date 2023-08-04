alias type: DType = DType.float32    
alias netls: Int = simdwidthof[type]()
    
#Need a was to cache / keep track of the A, W, and b tensors
def linear_forward(A: Tensor[type], W: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let Z = W.dot(A) + b
    return Z

def relu_backward(dA: Tensor[type], Z: Tensor[type]) -> Tensor[type]:
    let dZ = dA
    #dZ[Z <= 0] = 0
    return dZ

#Test writing a W.X+b op, input size 2, hum_hidden = 32
let feature_size = 2
let num_samples = 10

# Learning rate
let alpha = 0.01

let W1_size = 32
let W2_size = 32
let final_size = 1

let input_data = Tensor[type].arange(__idx(feature_size, num_samples))
let Y = Tensor[type](1, num_samples)

#create the elements for 2 layer network
var W1 = Tensor[type].random(W1_size, feature_size)*0.01
var W2 = Tensor[type].random(W2_size, W1_size)*0.01
var FC = Tensor[type].random(final_size, W2_size)*0.01

var b1 = Tensor[type](W1_size,1)
var b2 = Tensor[type](W2_size,1)
var b3 = Tensor[type](final_size,1)

#run forward
var Z1 = linear_forward(input_data, W1, b1)
var A1 = Z1.relu()

var Z2 = linear_forward(A1, W2, b2) 
var A2 = Z2.relu()
var F = linear_forward(A2, FC, b3)

#calculate cost
var cost = (-1.0/num_samples) *  Y.dot(F.log().T()) + (1-Y).dot( (1-F).log().T() )  

#run backward, calculating and setting gradients
var dF = (-Y / F) + (1 - Y) / (1-F)

#update the parameter based on the gradients

# Backpropagation through the final linear layer
var m = dF.shape[1]
let dFC = (1.0 / m) * dF.dot(A2.T())
let db3 = (1.0 / m) * dF.sum_rows()
let dA2 = FC.T().dot(dF)

# Backpropagation through the second ReLU activation
dA2_relu = relu_backward(dA2, A2)

# Backpropagation through the second linear layer
m = dA2_relu.shape[1]
let dW2 = (1.0 / m) * dA2_relu.dot(A1.T())
let db2 = (1.0 / m) * dA2_relu.sum_rows()
let dA1 = W2.T().dot(dA2_relu)


# Backpropagation through the first ReLU activation
dA1_relu = relu_backward(dA1, A1)

# Backpropagation through the first linear layer
m = dA1_relu.shape[1]
let dW1 = (1.0 / m) * dA1_relu.dot(input_data.T())
let db1 = (1.0 / m) * dA1_relu.sum_rows()

# Updating the parameters using gradient descent
W1 = W1 - alpha * dW1
b1 = b1 - alpha * db1

W2 = W2 - alpha * dW2
b2 = b2 - alpha * db2

FC = FC - alpha * dFC
b3 = b3 - alpha * db3
