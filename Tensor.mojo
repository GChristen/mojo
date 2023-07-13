#cant do import, need to do from import
from Vector import DynamicVector
from Math import sqrt, exp
from String import String
from List import VariadicList
from Buffer import Buffer, NDBuffer
from DType import DType
from Pointer import DTypePointer
from Random import rand
from Memory import memset_zero, memcpy
from Reductions import variance, mean
from TargetInfo import simdwidthof
from IO import print_no_newline
from Assert import debug_assert
from Functional import vectorize

#Helper functions. These functions smell, it would be good to remove these
#What to do with these functions - they smell


#consider this https://docs.modular.com/mojo/MojoStdlib/Index.html#product
fn __get_products(list: DynamicVector[Int]) -> DynamicVector[Int]:
    var products = DynamicVector[Int](len(list)+1)#fix
    products[0] = 1
    #calculate products
    for i in range(len(list)):
        products[i+1]= products[i]*list[i]
    return products

#this should be much simpler
fn __get_dim_product(dims: VariadicList[Int]) -> Int:
    let rank = len(dims)    
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
    
#Tensor implementation backed by Pointer
#Consider removing the need for Buffers, using only pointers to do things like fills, 
# - since Buffers needs Compile time size - which means we can OnlyUse Vairadic list (register passable), which is not good
# - first make changes to remove need for Buffer
# - [done] use print no line
# - [done] remove Dim (not getting any benefit from it, make Int)
struct Tensor[type: DType]:
    #hacking around the gaps in DimList right now(e.g. no len)
    var rank: Int
    var size: Int
    var shape: DynamicVector[Int]
    
    var data: DTypePointer[type]
    var grads: DTypePointer[type] #TODO: gradients don't need the same type   
        
    fn __init__(inout self, shape: VariadicList[Int]):
        self.rank = len(shape)
        self.shape = __vector_from_list(shape)
        self.size = __get_dim_product(shape)
        self.data = DTypePointer[type].alloc(self.size)
        self.grads = DTypePointer[type].alloc(self.size)
        memset_zero(self.data, self.size)
        memset_zero(self.grads, self.size)

    fn __init__(inout self, owned data: DTypePointer[type], owned shape: VariadicList[Int]):
        self.rank = len(shape)
        self.shape = __vector_from_list(shape)
        self.size = __get_dim_product(shape)
        self.data = data                
        self.grads = DTypePointer[type].alloc(self.size)        
        memset_zero(self.grads, self.size)
        
    fn __copyinit__(inout self, existing: Self):
        self.rank = existing.rank
        self.size = existing.size
        self.shape = existing.shape.deepcopy() 
        self.data = DTypePointer[type].alloc(existing.size)
        memcpy(self.data, existing.data, existing.size)
        self.grads = DTypePointer[type].alloc(existing.size)
        memcpy(self.grads, existing.grads, existing.size)        

    fn __moveinit__(inout self, owned existing: Self):
        self.rank = existing.rank
        self.size = existing.size
        self.shape = existing.shape.deepcopy() 
        self.data = DTypePointer[type].alloc(existing.size)
        memcpy(self.data, existing.data, existing.size)
        self.grads = DTypePointer[type].alloc(existing.size)
        memcpy(self.grads, existing.grads, existing.size)        
        
    fn __del__(owned self):
        self.data.free()
        self.grads.free()

    ###Fill ops
    
    #Can this whole function be done with memset?
    fn fill(inout self, val: Int):
        alias nelts: Int = simdwidthof[type]()
        let num_iter = self.size // nelts
        
        #fill by nelts steps
        for i in range(num_iter):
            self.store[nelts](__idx(nelts*i), val)        
            
        #fill remainder
        for i in range(nelts*(self.size//nelts), self.size):
            #can't yet use *int in setItem
            self[__idx(i)]= val        
    
    @staticmethod
    fn ones(shape:VariadicList[Int]) -> Self:
        var x = Self(shape)
        x.fill(1) #this triggers a moveinit
        return x
            
    @staticmethod
    fn zeros(*dims: Int) -> Self:
        return Self(VariadicList[Int](dims)) #by default the values are set to 0

    @staticmethod
    fn zeros(shape:VariadicList[Int]) -> Self:
        return Self(shape) #by default the values are set to 0

    @staticmethod
    fn random(shape:VariadicList[Int]) -> Self:
        let size = __get_dim_product(shape)
        let p = DTypePointer[type].alloc(size)
        rand[type](p, size)        
        return Self(p^, shape)    

    @staticmethod
    #fill with range until full
    fn arange(shape:VariadicList[Int], start: Int =0, step: Int =1) -> Self:
        var x = Tensor[type](shape)        
        var val = start
        for i in range(x.size):
            x[__idx(i)]=val
            val+=step  
        return x
    
    ### Access ops
    
    #TODO: 
    # - Implement bounds checking for sets and gets
    # - Remove the hack (using DynamicVector for indices) because setItem can unpack Int args before value
    # - implement fn __setitem__(self, *loc: Int, val:SIMD[type,1]):
    # - change bounds checking, add raises
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
            
        if offset > self.size:
            print("Warning. Index ", __vector_to_string(index) , "outside of bounds")
        return self.data.simd_load[nelts](offset)

    @always_inline
    fn __setitem__(self, index: DynamicVector[Int], val:SIMD[type,1]) :
        self.store[1](index, val)

    @always_inline
    fn store[nelts:Int](self, index: DynamicVector[Int], val:SIMD[type,nelts]) :
        var offset = 0 

        #calculate products 
        let products = __get_products(self.shape)
                
        for i in range(len(index)):
            offset += products[i]*index[i]
           
        #bounds check
        if offset > self.size:
            print("Warning. Index ", __vector_to_string(index) , "outside of bounds")

        self.data.simd_store[nelts](offset, val)     
    
    ### Display Ops
    fn print_tensor_recursive(self, dim: Int, inout indices: DynamicVector[Int], inout out: String):
        #Tons of potential issues here: recursion, print one at a time, etc
        if dim == len(self.shape):  # Base case: reached the innermost dimension
            out = out + " " + self[indices]
            return

        for i in range(self.shape[dim]):
            indices[dim] = i
            self.print_tensor_recursive(dim + 1, indices, out)

            if i == self.shape[dim] - 1:
                out = out + "\n"  # Move to the next line after printing the last element of a dimension
    
    
    fn show(self, summary: Bool = False):
        #print summary
        if summary:
            #shape            
            print("Shape: ", __vector_to_string(self.shape))
            #rank
            print("Rank: ", self.rank)
            #size
            print("Size: ", self.size)

        #print data - call recursive print
        print("Data:")
        var indices = DynamicVector[Int]()
        for i in range(len(self.shape)):
            indices.push_back(0)
            
        var out = String("")
        self.print_tensor_recursive(0, indices, out) 
        print(out)        
    
    ### Simple operators (add, sub, mul, exp, pow)                    
    fn __imul__(inout self: Self, rhs: Int):
        alias nelts: Int = simdwidthof[type]()
        @parameter
        fn op[opsize: Int](n : Int):
            self.store[opsize](__idx(n), self.load[opsize](__idx(n)) * rhs)
        vectorize[nelts, op](self.size)
        
    fn __mul__(self: Self, rhs: Int) -> Self:
        var x = self
        x*=rhs
        return x
   
    #layout is N, C, H, W
    fn __mul__(self: Self, rhs: Self) -> Self:
        alias nelts: Int = simdwidthof[type]()
        
        #consider changing rank to len (safer)
        if rhs.rank == 1:            
            if self.rank == 1: #what about colum vectors? rank 2
                if self.size != rhs.size: print("Shapes not aligned")
                var result = Tensor[type].zeros(self.shape[0])
                @parameter
                fn op[opsize: Int](n : Int):
                    let product = self.load[opsize](__idx(n)) * rhs.load[opsize](__idx(n))
                    result.store[opsize](__idx(n), product)
                vectorize[nelts, op](self.size)
                return result
        
        #temp catch all
        return Tensor[type].zeros(1)
    
    #layout is N, C, H, W
    fn dot(self: Self, rhs: Self) -> Self:
        alias nelts: Int = simdwidthof[type]()
        #add shape checks here        
        
        #consider changing rank to len (safer)
        if rhs.rank == 1:            
            if self.rank == 1: #what about colum vectors? rank 2
                var result = Tensor[type].zeros(1)
                @parameter
                fn op[opsize: Int](n : Int):
                    let product = self.load[opsize](__idx(n)) * rhs.load[opsize](__idx(n))
                    result.store[1](__idx(0), result.load[1](0) + (product.reduce_add()))
                vectorize[nelts, op](self.size)
                return result
            
        return Tensor[type].zeros(1)
    
    fn __iadd__(inout self: Self, rhs: Int):
        alias nelts: Int = simdwidthof[type]()
        @parameter
        fn op[opsize: Int](n : Int):
            self.store[opsize](__idx(n), self.load[opsize](__idx(n)) + rhs)
        vectorize[nelts, op](self.size)
    
    fn __add__(self: Self, rhs: Int) -> Self:
        var x = self
        x+=rhs
        return x

    fn __isub__(inout self: Self, rhs: Int):
        alias nelts: Int = simdwidthof[type]()
        @parameter
        fn op[opsize: Int](n : Int):
            self.store[opsize](__idx(n), self.load[opsize](__idx(n)) - rhs)
        vectorize[nelts, op](self.size)
    
    fn __sub__(self: Self, rhs: Int) -> Self:
        var x = self
        x-=rhs
        return x

    fn __itruediv__(inout self: Self, rhs: Int):
        alias nelts: Int = simdwidthof[type]()
        @parameter
        fn op[opsize: Int](n : Int):
            self.store[opsize](__idx(n), self.load[opsize](__idx(n)) / rhs)
        vectorize[nelts, op](self.size)
    
    fn __truediv__(self: Self, rhs: Int) -> Self:
        var x = self
        x/=rhs
        return x    
    ### Reduce ops
    
    ### NN Ops