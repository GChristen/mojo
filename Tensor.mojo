#cant do import, need to do from import
from Vector import DynamicVector
from Math import sqrt, exp, max, mul, add, sub
from String import String
from List import VariadicList
from Buffer import Buffer, NDBuffer
from DType import DType
from Pointer import DTypePointer
from Random import rand, randint
from Memory import memset_zero, memcpy
from Reductions import variance, mean
from TargetInfo import simdwidthof
from IO import print_no_newline
from Assert import debug_assert
from Functional import vectorize
from Intrinsics import strided_load
from TypeUtilities import rebind


#Helper functions. These functions smell, it would be good to remove these

#consider this https://docs.modular.com/mojo/MojoStdlib/Index.html#product
fn __get_products(list: DynamicVector[Int]) -> DynamicVector[Int]:
    var products = DynamicVector[Int](len(list)+1)#fix
    products[0] = 1
    #calculate products
    for i in range(len(list)):
        products[i+1]= products[i]*list[i]
    return products

fn __get_dim_product(dims: DynamicVector[Int], rank: Int) -> Int:
    var size = dims[0]
    for i in range(1, rank):
        size *= dims[i]
    return size    

#convert int list representing index/dimensions to DynamicVector
fn __idx(*idx: Int) -> DynamicVector[Int]:
    let list = VariadicList(idx)
    return __idx(list)

fn __idx(list: VariadicList[Int]) -> DynamicVector[Int]:
    var ret = DynamicVector[Int]()
    for i in range(len(list)):
        ret.push_back(list[i])
    return ret

fn __vector_compare(a: DynamicVector[Int], b:DynamicVector[Int]) -> Bool:
    if len(a) != len(b): return False
    
    for i in range(len(a)):
        if a[i]!=b[i]: return False
    
    return True

fn __vector_to_string(vec: DynamicVector[Int]) -> String:
    var out = String("[")
    for i in range(len(vec)):
        out += String(vec[i])
        out+= ", "
    return out[0:len(out)-2]+"]"


    
#Tensor implementation backed by Pointer
# - [done] make changes to remove need for Buffer within the struct
# - [done] don't built long return strings for print, use print_no_line where needed
# - [done] remove Dim (not getting any benefit from it, make Int)
# - Bounds checking everywhere!
# - use let for rank and size in struct, once supported
# - implement basic ops with scalars other than SIMD[type,1] (like int)
# - add a from_numpy static method
struct Tensor[type: DType]:
    #hacking around the gaps in DimList right now(e.g. no len)
    var rank: Int
    var size: Int
    var shape: DynamicVector[Int]
    
    var data: DTypePointer[type]
    var grads: DTypePointer[type] #TODO: gradients don't need the same type   

    fn __init__(inout self, *dims: Int):
        let shape = __idx(dims)
        self.rank = len(shape)
        self.shape = shape
        self.size = __get_dim_product(shape, self.rank)
        
        self.data = DTypePointer[type].alloc(self.size)
        self.grads = DTypePointer[type].alloc(self.size)
        memset_zero(self.data, self.size)
        memset_zero(self.grads, self.size)
    
    fn __init__(inout self, shape: DynamicVector[Int]):
        self.rank = len(shape)
        self.shape = shape
        self.size = __get_dim_product(shape, self.rank)        
        self.data = DTypePointer[type].alloc(self.size)
        self.grads = DTypePointer[type].alloc(self.size)
        memset_zero(self.data, self.size)
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
        
        @parameter
        fn op[nelts: Int](n : Int):
            self.data.simd_store[nelts](n, val)                 
        vectorize[nelts, op](self.size)
    
    @staticmethod
    fn ones(*dims: Int) -> Self:
        let shape = __idx(dims)
        var x = Tensor[type](shape)
        x.fill(1) #this triggers a moveinit
        return x
            
    @staticmethod
    fn zeros(*dims: Int) -> Self:
        let shape = __idx(dims)
        return Self(shape) #by default the values are set to 0
    
    @staticmethod
    fn random_int(shape: DynamicVector[Int], int_low:Int =0, int_high:Int =10) -> Self:
        let x = Tensor[type](shape)
        randint[type](x.data, x.size, int_low, int_high)
        return x        
        
    @staticmethod
    fn random(*dims: Int) -> Self:
        let shape = __idx(dims)
        let x = Tensor[type](shape)
        rand[type](x.data, x.size)        
        return x

    #TODO: 
    # - change the output to match np.arange.reshape output, right now it's in 'F'
    # - turn dims into *Int, once name arguments are supported
    @staticmethod
    fn arange(shape:DynamicVector[Int], start: Int =0, step: Int =1) -> Self:
        var x = Tensor[type](shape)        
        var val = start
        for i in range(x.size):
            x[__idx(i)]=val
            val+=step  
        return x
    
    ### Access ops
    
    #TODO: 
    # - Remove the hack (using DynamicVector for indices) because setItem can unpack Int args before value
    # - implement fn __setitem__(self, *loc: Int, val:SIMD[type,1]):
    # - change bounds checking, add raises
    # - implement slices
    #FEATURE_REQUEST: 
    # - strided_ops could support offset argument
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
    fn load_stride[nelts:Int](self, offset:Int, stride: Int) -> SIMD[type, nelts]:
        return self.data.offset(offset).simd_strided_load[nelts](stride)
    
    @always_inline
    fn load[nelts:Int](self, index: DynamicVector[Int]) -> SIMD[type, nelts]:
        #the offset is calculated as prod(shape:[:-1])*loc[-1] + ... + loc[0]        
        let offset = self.index_to_offset(index)
        return self.data.simd_load[nelts](offset)
    
    @always_inline
    fn __setitem__(self, index: DynamicVector[Int], val:SIMD[type,1]) :
        self.store[1](index, val)

    @always_inline
    fn store_stride[nelts:Int](self, offset:Int, stride:Int, val:SIMD[type,nelts]) :
        self.data.offset(offset).simd_strided_store[nelts](val, stride)    

    @always_inline
    fn store[nelts:Int](self, index: DynamicVector[Int], val:SIMD[type,nelts]) :
        let offset = self.index_to_offset(index) 
        self.data.simd_store[nelts](offset, val)                     
    
    #return an offset into data pointer, represented by the index
    fn index_to_offset(self, index: DynamicVector[Int]) -> Int:
        var offset = 0 

        #calculate products 
        let products = __get_products(self.shape)
                
        for i in range(len(index)):
            offset += products[i]*index[i]
           
        #bounds check
        if offset > self.size:
            print("Warning. Index ", __vector_to_string(index) , "outside of bounds")
        
        return offset

    fn index_to_offset(self, *index: Int) -> Int:
        let index_vector = __idx(VariadicList[Int](index))
        return self.index_to_offset(index_vector)
    
    ### Display Ops
    ### Refactor to use op_over_dimension
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
    
    ### Operators (add, sub, mul, exp, pow)                    
    
    #layout is N, C, H, W
    # Computes Self ☉ rhs (https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
    #TODO
    # - Important: shape cheks to raise errors
    # - Add func name to error messages
    # - [done] matrix . matrix mul (rhs.rank==2)
    # - later: implement for column vectors (2,1) or (1,1,1,X) shapes 
    fn __mul__(self: Self, rhs: Self) -> Self:
        alias nelts: Int = simdwidthof[type]()
        let lhs_last_shape = self.shape[len(self.shape)-1]
        let rhs_last_shape = rhs.shape[len(rhs.shape)-1]        
                
        #self ☉ vec
        if rhs.rank == 1:
            if not(lhs_last_shape == rhs_last_shape): print("Shapes not aligned:", lhs_last_shape, rhs_last_shape)
            #vec * vec
            if self.rank == 1: 
                var result = Tensor[type].zeros(self.shape[0])
                @parameter
                fn rowmul_11[opsize: Int](n : Int):
                    let product = self.load[opsize](__idx(n)) * rhs.load[opsize](__idx(n))
                    result.store[opsize](__idx(n), product)
                vectorize[nelts, rowmul_11](self.size)
                return result
            #tensor(>= 2R) ☉ vec
            else:
                var result = Tensor[type](self.shape)
                                
                fn vec_mul(indices: DynamicVector[Int]):
                    #get the stride and offset for this row
                    let stride = __get_dim_product(self.shape, self.rank-1)
                    let offset = self.index_to_offset(indices)
                    
                    #vectorize a strided load (row), multiply w/ rhs(vector), and do strided_store
                    @parameter
                    fn rowmul_1X[opsize: Int](n : Int):
                        let cur_offset = offset+(n*stride)                        
                        let row = self.load_stride[opsize](cur_offset, stride)
                        let product = mul[type, opsize](row, rhs.load[opsize](__idx(n)) )
                        result.store_stride[opsize](cur_offset, stride, product)
                    vectorize[nelts, rowmul_1X](lhs_last_shape)
                
                #run this for all rows in tensor    
                self.op_over_dimension(self.rank - 2, vec_mul)                    
                
                return result
        #tensor ☉ matrix
        elif rhs.rank == 2:
            var result = Tensor[type](self.shape)
            
            @parameter
            fn tensor_mul(indices: DynamicVector[Int]):
                #get the strides (lhs stride may be bigger as lhs may be > R2)
                let stride = __get_dim_product(self.shape, self.rank-1)
                let rhs_stride = rhs.shape[0] #only need __get_dim_product(rhs.shape, rhs.rank-1) for rank > 2

                #get the offset - the location of the start of lhs and rhs rows
                let lhs_offset = self.index_to_offset(indices)
                let rhs_offset = indices[len(indices)-2] #only need rhs.index_to_offset(indices[len(indices)-2]) for rank>2
                      
                #load lhs row, load rhs row, mul, and store in result
                @parameter
                fn rowmul_X2[opsize: Int](n : Int): 
                    let cur_lhs_offset = lhs_offset+(n*stride)
                    let lhs_row = self.load_stride[opsize](cur_lhs_offset, stride)
                    let rhs_row = rhs.load_stride[opsize](rhs_offset+(n*rhs_stride), rhs_stride)
                    let product = mul[type, opsize](lhs_row, rhs_row)
                    result.store_stride[opsize](cur_lhs_offset, stride, product)
                vectorize[nelts, rowmul_X2](lhs_last_shape)            
                
            #run this for all matrices(last two dims) in the tensor
            self.op_over_dimension(self.rank - 2, tensor_mul)

            return result
         

        #temp catch all
        return Tensor[type].zeros(1)   
            
    #layout is N, C, H, W
    #TODO:
    # - [done] recreate numpy dot for vec.vec, matrix.vec, matrix.matrx
    # - implement tensor(>R2) . matrix or vec (extra dimension is the batch)
    fn dot(self: Self, rhs: Self) -> Self:
        alias nelts: Int = simdwidthof[type]()
        #vec . vec [1]
        if rhs.rank == 1 and self.rank == 1: #what about colum vectors (with a higher rank but empty rows(1,...,X)?
            var result = Tensor[type].zeros(1)
            @parameter
            fn dot_11[opsize: Int](n : Int):
                let product = self.load[opsize](__idx(n)) * rhs.load[opsize](__idx(n))
                result[__idx(0)]+= product.reduce_add()
            vectorize[nelts, dot_11](self.size)
            return result 
        #matrix.matrix [MxN]
        elif self.rank == 2 and rhs.rank == 2:         
            if not(self.shape[1] == rhs.shape[1]): print("In dot shapes not aligned:", self.shape[1], rhs.shape[0])
            var result = Tensor[type].zeros(self.shape[0], rhs.shape[1])  
            for m in range(result.shape[0]):                    
                for k in range(self.shape[1]):
                    @parameter
                    fn dot_22[nelts : Int](n : Int):
                        result.store[nelts](__idx(m,n), result.load[nelts]( __idx(m,n) ) + self[ __idx(m,k) ] * rhs.load[nelts]( __idx(k,n) ))
                    vectorize[nelts, dot_22](result.shape[1])
            return result            
        #matrix . vec [result is 1xM]
        elif self.rank == 2 and rhs.rank == 1:             
            if not(self.shape[1] == rhs.shape[0]): print("In dot, shapes not aligned:", self.shape[1], rhs.shape[0])
            var result = Tensor[type].zeros(self.shape[0])  
            #for each row of self (or col of result), do a vec.vec - do: (strided load of row) * rhs, store in n
            for m in range(result.shape[0]):
                let stride = __get_dim_product(self.shape, self.rank-1)
                let offset = self.index_to_offset(__idx(m))                
                @parameter
                fn rowdot[opsize: Int](n : Int):
                    let cur_offset = offset+(n*stride)
                    let row = self.data.offset(cur_offset).simd_strided_load[opsize](stride)
                    let product = mul[type, opsize](row, rhs.load[opsize](__idx(n)) )
                    result[__idx(m)] += product.reduce_add()
                vectorize[nelts, rowdot](rhs.shape[0])
            return result
        #tensor . vec
        elif self.rank == 3 and rhs.rank == 1:
            #for each matrix in first dim, do matrix.vec
            pass
        #tensor . matrix
        elif self.rank == 3 and rhs.rank == 2:            
            pass
        
        return Tensor[type].zeros(1)    
    
    fn __imul__(inout self: Self, rhs: SIMD[type,1]):
        alias nelts: Int = simdwidthof[type]()
        @parameter
        fn op[opsize: Int](n : Int):
            self.store[opsize](__idx(n), self.load[opsize](__idx(n)) * rhs)
        vectorize[nelts, op](self.size)
        
    fn __mul__(self: Self, rhs: SIMD[type,1]) -> Self:
        var x = self
        x*=rhs
        return x
    
    fn __iadd__(inout self: Self, rhs: SIMD[type,1]):
        alias nelts: Int = simdwidthof[type]()
        @parameter
        fn op[opsize: Int](n : Int):
            self.store[opsize](__idx(n), self.load[opsize](__idx(n)) + rhs)
        vectorize[nelts, op](self.size)
    
    fn __add__(self: Self, rhs: SIMD[type,1]) -> Self:
        var x = self
        x+=rhs
        return x
        
    fn __isub__(inout self: Self, rhs: SIMD[type,1]):
        alias nelts: Int = simdwidthof[type]()
        @parameter
        fn op[opsize: Int](n : Int):
            self.store[opsize](__idx(n), self.load[opsize](__idx(n)) - rhs)
        vectorize[nelts, op](self.size)
    
    fn __sub__(self: Self, rhs: SIMD[type,1]) -> Self:
        var x = self
        x-=rhs
        return x

    fn __itruediv__(inout self: Self, rhs: SIMD[type,1]):
        alias nelts: Int = simdwidthof[type]()
        @parameter
        fn op[opsize: Int](n : Int):
            self.store[opsize](__idx(n), self.load[opsize](__idx(n)) / rhs)
        vectorize[nelts, op](self.size)
    
    fn __truediv__(self: Self, rhs: SIMD[type,1]) -> Self:
        var x = self
        x/=rhs
        return x    
    
    
    fn __iadd__(inout self: Self, rhs: Self):
        pass
    
    fn __add__(self: Self, rhs: Self) -> Self:
        alias nelts: Int = simdwidthof[type]()
        var result = Tensor[type](self.shape)        
        #tensor+tensor or vec+vec with same shape - treat as flat buffers, add all
        if __vector_compare(self.shape, rhs.shape):
            @parameter
            fn same_add[opsize: Int](n : Int):
                let product = add( self.load[opsize](__idx(n)) , rhs.load[opsize](__idx(n)))
                result.store[opsize](__idx(n), product)
            vectorize[nelts, same_add](self.size)
            return result            

        #tensor + vec row add
        if rhs.rank==1 and rhs.shape[0] == self.shape[len(self.shape)-2]:
            print("in rowadd")
            fn rowvec_add(indices: DynamicVector[Int]):
                #get the stride and offset for this row
                let stride = __get_dim_product(self.shape, self.rank-1)
                let offset = self.index_to_offset(indices)

                #vectorize a strided load (row), multiply w/ rhs(vector), and do strided_store
                @parameter
                fn rowadd[opsize: Int](n : Int):
                    let cur_offset = offset+(n*stride)                        
                    let row = self.load_stride[opsize](cur_offset, stride)
                    let product = add[type, opsize](row, rhs.load[opsize](__idx(n)) )
                    result.store_stride[opsize](cur_offset, stride, product)
                vectorize[nelts, rowadd](rhs.shape[0])

            #run this for all rows in tensor    
            self.op_over_dimension(self.rank - 2, rowvec_add)                    
            return result
    
        #tensor + vec(X,1) col add (add col to the second to last dim of tensor
        if rhs.rank==2 and rhs.shape[1]==1 and rhs.shape[0] == self.shape[len(self.shape)-2]:
            fn colvec_add(indices: DynamicVector[Int]):
                #vectorize a strided load (row), multiply w/ rhs(vector), and do strided_store
                @parameter
                fn coladd[opsize: Int](n : Int):
                    var cur_idx = indices.deepcopy()
                    cur_idx[len(indices)-1]=n
                    result.store[opsize](cur_idx, self.load[opsize](cur_idx) + rhs.load[opsize](__idx(n)))
                vectorize[nelts, coladd](rhs.shape[0])

            #run this for all rows in tensor    
            self.op_over_dimension(self.rank - 2, colvec_add)                    
            return result
        
        #change to failure
        return Tensor[type].zeros(1)
    
    ### Reduce ops
        
    #generic function to loop over dimensions of a tensor
    #and execute an operation over last_dim dimension (like scalar, row, colum, etc)
    fn op_over_dimension(self, last_dim_index: Int, op: fn(DynamicVector[Int]) capturing-> None ):            
            #init running index to 0,...,0 for shape
            var indices = self.shape.deepcopy()
            for i in range(len(indices)): indices[i] = 0            

            while True: 
                #reached the end of the first dimension, done looping through all dims
                if (self.shape[0]-1 <  indices[0]):
                    break    

                #execute op
                op(indices)
                            
                
                # Move to the next element in the last dimension
                indices[last_dim_index] += 1

                #For all dimensions
                for i in range(len(self.shape)-2, -1, -1):
                    #if not the first dim and reached end, reset current and add one to previous
                    if indices[i] == self.shape[i] and i!=0: 
                        indices[i]=0
                        indices[i-1]+=1
    
