from Vector import DynamicVector
from Math import sqrt, exp
from String import String
from List import VariadicList, DimList, Dim
from Buffer import Buffer
from DType import DType
from Pointer import DTypePointer
from Random import rand
from Memory import memset_zero
from Reductions import variance, mean
from TargetInfo import simdwidthof


#Helper functions. These functions smell, it would be good to remove these
#What to do with these functions - they smell

#consider this https://docs.modular.com/mojo/MojoStdlib/Index.html#product
fn __get_products(list: DynamicVector[Dim]) -> DynamicVector[Int]:
    var products = DynamicVector[Int](len(list)+1)#fix
    products[0] = 1
    #calculate products
    for i in range(len(list)):
        products[i+1]= products[i]*list[i].get()
    return products

fn __get_dim_product(rank: Int, dims: DynamicVector[Dim]) -> Int:
    var size = dims[0].get()
    for i in range(1, rank):
        size *= dims[i].get()
    return size    

#this should be much simpler
fn __get_dim_product(rank: Int, dims: DynamicVector[Int]) -> Int:
    var size = dims[0]
    for i in range(1, rank):
        size *= dims[i]
    return size    

#convert Ints representing location to DynamicVector used as index
fn __idx(*idx: Int) -> DynamicVector[Int]:
    let list = VariadicList(idx)
    var ret = DynamicVector[Int](len(list))
    for i in range(len(list)):
        ret.push_back(list[i])
    return ret

fn __vector_from_list[type: AnyType](list: VariadicList[type]) -> DynamicVector[type]:
    var ret = DynamicVector[type]()
    for i in range(len(list)):
        ret.push_back(list[i])
    return ret
    
fn __vector_to_string(vec: DynamicVector[Int]) -> String:
    var out = String("[")
    for i in range(len(vec)):
        out += String(vec[i])
        out+= ", "
    return out[0:len(out)-2]+"]"
    
#Tensor implementation backed by Pointed, uses Buffers where needed
struct Tensor[type: DType]:
    #hacking around the gaps in DimList right now(e.g. no len)
    var rank: Int
    var size: Int
    var shape: DynamicVector[Dim]
    
    var data: DTypePointer[type]
    var grads: DTypePointer[type] #TODO: gradients don't need the same type   
        
    fn __init__(inout self, shape: DynamicVector[Dim]):
        self.rank = len(shape)
        self.shape = shape
        self.size = __get_dim_product(self.rank, shape)
        self.data = DTypePointer[type].alloc(self.size)
        self.grads = DTypePointer[type].alloc(self.size)
        memset_zero(self.data, self.size)
        memset_zero(self.grads, self.size)

    fn __init__(inout self, owned data: DTypePointer[type], owned shape: DynamicVector[Dim]):
        self.rank = len(shape)
        self.shape = shape
        self.size = __get_dim_product(self.rank, shape)
        self.data = data                
        self.grads = DTypePointer[type].alloc(self.size)        
        memset_zero(self.grads, self.size)
    
    fn __del__(owned self):
        self.data.free()
    
    #fill ops, remove rank once it is infered
    @staticmethod
    fn ones[rank: Int, shape:DynamicVector[Dim]]() -> Self:
        alias size = __get_dim_product(rank, shape)
        let p = DTypePointer[type].alloc(size)
        let x = Buffer[size, type](p).fill(1)
        return Self(p^, shape)

    @staticmethod
    fn zeros[rank: Int, shape:DynamicVector[Dim]]() -> Self:
        alias size = __get_dim_product(rank, shape)        
        let p = DTypePointer[type].alloc(size)
        let x = Buffer[size, type](p).zero()
        return Self(p^, shape)    
    
    @staticmethod
    fn random[rank: Int, shape:DynamicVector[Dim]]() -> Self:
        let size = __get_dim_product(rank, shape)
        let p = DTypePointer[type].alloc(size)
        rand[type](p, size)        
        return Self(p^, shape)    

    @staticmethod
    #fill with range until full
    fn arange[rank: Int, shape:DynamicVector[Dim]](start: Int =0, step: Int =1) -> Self:
        alias size = __get_dim_product(rank, shape)
        let p = DTypePointer[type].alloc(size)
        let y = Buffer[size, type](p)
        var val = start
        for i in range(size):
            y[i]=val
            val+=step
        return Self(p^, shape) 
    
    ### Access ops
    
    #TODO: 
    # - Implement bounds checking for sets and gets
    # - Remove the hack (using DynamicVector for indices) because setItem can unpack Int args before value
    @always_inline
    fn __getitem__(self, index: DynamicVector[Int]) -> SIMD[type, 1]:
        return self.load[1](index)    
    
    @always_inline
    fn __getitem__(self, *loc: Int) -> SIMD[type, 1]:        
        let list = VariadicList[Int](loc)
        var index = DynamicVector[Int]()
        for i in range(len(list)):
            index.push_back(list[i])        
        return self.load[1](index)    

    @always_inline
    fn load[nelts:Int](self, index: DynamicVector[Int]) -> SIMD[type, nelts]:
        var offset = 0
        #the offset is calculated as prod(shape:[:-1])*loc[-1] + ... + loc[0]        
        
        #calculate products 
        let products = __get_products(self.shape)
        
        for i in range(len(index)):
            offset += products[i]*index[i]

        return self.data.simd_load[nelts](offset)

    @always_inline
    #Set item, should have args be index, val. Right now it's val, index to get around named args after *p args
    fn __setitem__(self, index: DynamicVector[Int], val:SIMD[type,1]):
        self.store[1](index, val)
    
    @always_inline
    #Implement this when name keywords are supported
    #fn __setitem__(self, *loc: Int, val:SIMD[type,1]):
    #    let list = VariadicList[Int](loc)
    #    var index = DynamicVector[Int]()
    #    for i in range(len(list)):
    #        index.push_back(list[i])        
    #    self.store[1](index, val)

    @always_inline
    fn store[nelts:Int](self, index: DynamicVector[Int], val:SIMD[type,nelts]):
        var offset = 0 
        
        #calculate products 
        let products = __get_products(self.shape)
        
        for i in range(len(index)):
            offset += products[i]*index[i]
        
        self.data.simd_store[nelts](offset, val)     
    
    ### Display Ops
    fn print_tensor_recursive(self, dim: Int, inout indices: DynamicVector[Int], inout out: String):
        #Tons of potential issues here: recursion, print one at a time, etc
        if dim == len(self.shape):  # Base case: reached the innermost dimension
            out = out + " " + self[indices]
            return

        for i in range(self.shape[dim].get()):
            indices[dim] = i
            self.print_tensor_recursive(dim + 1, indices, out)

            if i == self.shape[dim].get() - 1:
                out = out + "\n"  # Move to the next line after printing the last element of a dimension
    
    fn print_tensor(self):
        var indices = DynamicVector[Int]()
        for i in range(len(self.shape)):
            indices.push_back(0)
            
        var out = String("")
        self.print_tensor_recursive(0, indices, out) 
        print(out)
    
    fn show(self, summary: Bool = True):
        #shape
        var ret = String("")
        if summary:
            #shape            
            ret = ret + "Shape: ["
            for i in range(len(self.shape)):
                ret += String(self.shape[i].get()) + ", "
            ret = ret[0:len(ret)-2] + "]\n"            
            #rank
            ret = ret + "Rank: " + String(self.rank)
            #size
            ret = ret + "\nSize: " + String(self.size) + "\n"

        ret = ret + "Data: \n"
        print(ret)
        self.print_tensor()

    ### math ops
#    fn __mul__(self: Self, rhs: Int) -> Self:
#        var ret = Tensor[type](self.shape)
#        return Tensor[type].ones[2, VariadicList[Dim](9, 3)]()
    
    fn __mul__(self: Self, rhs: Self) -> Self:
        return Tensor[type].ones[2, VariadicList[Dim](9, 3)]()
    
    fn __sub__(self: Self, rhs: Self) -> Self:
        return Tensor[type].ones[2, VariadicList[Dim](9, 3)]()

    #add element wise with broadcasting if needed
    fn __add__(self: Self, rhs: Int) -> Self:
        return Tensor[type].ones[2, VariadicList[Dim](9, 3)]()
    
    fn __add__(self: Self, rhs: Self) -> Self:
        return Tensor[type].ones[2, VariadicList[Dim](9, 3)]()
    
    fn __pow__(self: Self, rhs: Int) -> Self:
        return Tensor[type].ones[2, VariadicList[Dim](9, 3)]()
    
    fn __exp__(self: Self) -> Self:
        return Tensor[type].ones[2, VariadicList[Dim](9, 3)]()
    
    #should we do __i methods for when the Tensor is a scalar?
    fn __imul__(inout self: Self, rhs: Int):
        alias nelts: Int = simdwidthof[type]()
        let num_iter = self.size // nelts
        
        #fill by nelts steps
        for i in range(num_iter):
            self.store[nelts](nelts*i, self.load[nelts](nelts*i) * rhs)        
            
        #fill remainder
        for i in range(nelts*(self.size//nelts), self.size):
            #can't yet use *int in setItem
            self[__idx(i)]= self[i] * rhs            
    
    
    ### Reduce ops
    
    ### NN Ops