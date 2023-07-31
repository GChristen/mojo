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

@value
@register_passable
struct TensorRef:
    var id: Int 
    var op_ref: Pointer[Int]    
    var data_ref: DTypePointer[type]
    var grad_ref: DTypePointer[type]
    #a reference to the operation that created this tensor
    
    fn __init__(inout tensor: Tensor, id: Int) -> Self:
        #get the opref from the tensor
        let p = tensor.op_ref.bitcast[Int]()
        return Self(id, p, tensor.data, tensor.grads)

@value
@register_passable
struct OpRef:
    var op: Int
    var left: TensorRef
    var right: TensorRef
    
fn create_null_opref()-> Pointer[OpRef]:    
    let null_tensor = TensorRef(-1, Pointer[Int].get_null(), DTypePointer[type].get_null(),  DTypePointer[type].get_null())
    var op_ref = OpRef(-1, null_tensor, null_tensor)
    return Pointer[OpRef].address_of(op_ref)

@value
struct BackPropHarness:
    var cur_id: Int
    var ops : DynamicVector[OpRef]
    
    fn __init__(inout self: Self):
        self.cur_id = 0
        self.ops = DynamicVector[OpRef]()
        
    fn __get_id(inout self: Self)-> Int:
        self.cur_id+=1
        return self.cur_id
    
    fn add_op(self: Self,  lhs: TensorRef,  rhs: TensorRef, op: Int) -> Pointer[OpRef]:
        #let op_ref = OpRef(op, lhs, rhs)
        #self.ops.push_back(op_ref)
        return create_null_opref()
    
    fn backward(self: Self):
        #get the last OpRef
        #make sure it's a scalar
        #build topo
          #for every child reference
        pass
    

#how to inject harness
var harness = BackPropHarness()