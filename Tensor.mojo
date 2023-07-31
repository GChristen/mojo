#TODO: make this it's own module once imports work, Tensor not subscritable
from Vector import DynamicVector
from Math import sqrt, exp, max, mul, add, sub, div
from String import String
from List import VariadicList
from Buffer import Buffer, NDBuffer
from DType import DType
from Pointer import DTypePointer
from Random import rand, randint
from Memory import memset_zero, memcpy
from Reductions import variance, mean
from TargetInfo import simdwidthof
from IO import print, print_no_newline
from Assert import debug_assert
from Functional import vectorize
from Intrinsics import strided_load


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
# - [done-ish] add docstrings
# - consider going back to DimList (in order to make this type 
@value
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
        """ Fill this instance of tensor with val. Note, this fn doesn't cast."""
        alias nelts: Int = simdwidthof[type]()
        
        @parameter
        fn op[nelts: Int](n : Int):
            self.data.simd_store[nelts](n, val)                 
        vectorize[nelts, op](self.size)
    
    @staticmethod
    fn ones(*dims: Int) -> Self:
        """Static fn to create a new Tensor filled with ones of shape _dims_."""
        let shape = __idx(dims)
        var x = Tensor[type](shape)
        x.fill(1) #this triggers a moveinit
        return x
            
    @staticmethod
    fn zeros(*dims: Int) -> Self:
        """An alias to Tensor(*dims), creates a new tensor filled with 0 of shape _dims_."""
        let shape = __idx(dims)
        return Self(shape) #by default the values are set to 0
    
    @staticmethod
    fn random_int(shape: DynamicVector[Int], int_low:Int =0, int_high:Int =10) -> Self:
        """Static fn to create a new Tensor of shape _shape, filled with random ints. 
           Note, this fn takes DynamicVector for shape (as opposed to *dims), bcs Mojo doesn't support unpacking yet"""
        let x = Tensor[type](shape)
        randint[type](x.data, x.size, int_low, int_high)
        return x        
        
    @staticmethod
    fn random(*dims: Int) -> Self:
        """Static fn to create a new Tensor filled with random values of Tensor param type"""
        let shape = __idx(dims)
        let x = Tensor[type](shape)
        rand[type](x.data, x.size)        
        return x

    #TODO: 
    # - change the output to match np.arange.reshape output, right now it's in 'F'
    # - turn dims into *Int, once name arguments are supported
    @staticmethod
    fn arange(shape:DynamicVector[Int], start: Int =0, step: Int =1) -> Self:
        """Static fn to create a new tendor filled with a list of increasing numbers, 
           starting at _start_, increasing by _step_ size. Will continue until shape is full.
           Note, this fn takes DynamicVector for shape (as opposed to *dims), bcs Mojo doesn't support unpacking yet"""
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
        """Utility fn to load data from memory with a stride from this Tensor, 
        starting at _offset_ and with stride of _stride_
        
        If data at pointer _self.data_ was [0,1,2,3,4,5,...,N]
        load_stride(0, 2) would return 0,2,4,...,N
        """
        return self.data.offset(offset).simd_strided_load[nelts](stride)
    
    @always_inline
    fn load[nelts:Int](self, index: DynamicVector[Int]) -> SIMD[type, nelts]:
        """Utility fn to load _nelts_ sized data from memory for this Tensor 
        at the provided index.
        
        Parameters
        ----------
        nelts : int
            Size of data to load. (This often should match the SIMD size of the type)

        Arguments
        ----------
        index : DynamicVector[Int]
            A list of Ints specifying where to start loading data from. Has the same layout as shape.
            The index (in 1 or more dimensions) is used to comupte an offset into memory. 
            The offset is calculated as the product of: data.shape[:-1]*index[-1] + ... + loc[0]        
            
        Returns
        -------
        SIMD[type, nelts]
        """
        #
        let offset = self.index_to_offset(index)
        return self.data.simd_load[nelts](offset)
    
    @always_inline
    fn __setitem__(self, index: DynamicVector[Int], val:SIMD[type,1]) :
        self.store[1](index, val)

    @always_inline
    fn store_stride[nelts:Int](self, offset:Int, stride:Int, val:SIMD[type,nelts]) :
        """Utility fn to store data in a Tensor, with a stride. See load_stride"""
        self.data.offset(offset).simd_strided_store[nelts](val, stride)    

    @always_inline
    fn store[nelts:Int](self, index: DynamicVector[Int], val:SIMD[type,nelts]) :
        """Utility fn to store data in a Tensor. See load."""
        let offset = self.index_to_offset(index) 
        self.data.simd_store[nelts](offset, val)                     
    
    #return an offset into data pointer, represented by the index
    fn index_to_offset(self, index: DynamicVector[Int]) -> Int:
        """Given an index (DynamicVector[Int]) in R>=R1, calculate an offset in R1 (flat)"""        
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
        """Overloaded index_to_offset that accepts a list of Int as the index"""
        let index_vector = __idx(VariadicList[Int](index))
        return self.index_to_offset(index_vector)
    
    ### Display Ops
    ### TODO: Refactor to use op_over_dimension
    fn __print_tensor_recursive(self, dim: Int, inout indices: DynamicVector[Int], inout out: String):
        """Internal utility fn to print a tensor. Recursively iterates through each index and prints value"""
        if dim == len(self.shape):  # Base case: reached the innermost dimension
            out = out + " " + self[indices]
            return

        for i in range(self.shape[dim]):
            indices[dim] = i
            self.__print_tensor_recursive(dim + 1, indices, out)

            if i == self.shape[dim] - 1:
                out = out + "\n"  # Move to the next line after printing the last element of a dimension
    
    
    fn show(self, summary: Bool = False, data: Bool=True):
        """Utility fn to print a Tensor. Can print summary, data or both"""
        #print summary
        if summary:
            #shape            
            print("Shape: ", __vector_to_string(self.shape))
            #rank
            print("Rank: ", self.rank)
            #size
            print("Size: ", self.size)

        #print data - call recursive print
        if data:
            print("Data:")
            var indices = DynamicVector[Int]()
            for i in range(len(self.shape)):
                indices.push_back(0)
            
            var out = String("")
            self.__print_tensor_recursive(0, indices, out) 
            print(out)        
    
    ### Operators (add, sub, mul, exp, pow)
    #Note, ignore ips in autograd - iops (e.g. imul), are trickier for autograd                
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
    
    # Generic brodacast op method.  Supports following:    
    # TODO:
    # - Clean! Very messy. Didn't find a way to pass an op as param or argument. Went for 
    # messy ifs and a fake enum. Relevant discussion: https://github.com/modularml/mojo/issues/271
    # - Important: change shape checks to raise errors
    # - Add op name to error messages
    fn broadcast_op[op: Int](self: Self, rhs: Self) -> Self:
        """Utility fn to execute an operation _op_ with broadcast semantics. For reference 
        the layout (indexing regime) for data in a Tensor N, C, H, W.

        Supports opertations for the following Tensor shape combinations:
          - vec * vec or tensor * tensor where the shapes are equal: will do an element 
          wise operation (self _op_ rhs) for each value in both tensors.
          - tensor (>=R2) * row vector (R1): if row shapes match, will do an element wise
          operation for each element in every row of self with each element in rhs (single row)
          - tensor (>=R2) * col vector (R2; (rows, 1)): if column shapes match will do an element 
          wise operation for each element in every column of self with each single element in each
          row of rhs.
          - tensor (R>=2) * matrix (R2): if matrix shapes match (rhw shape matches the last two 
          dimensions of tensor), will do an element wise _op_ for each matrix in tensor with rhs (matrix)
          
        
        Parameters
        ----------
        op : int
            The 'enum' value of the operations to execute. 
              0 = add
              1 = sub
              2 = div
              3 = mul (element wise multiplication as opposed to dot product)

        Arguments
        ----------
        self : Self 
            The current Tensor, instantiated with parameter type. 
            
        rhs : Self 
            The right hand side term of the operation. 
            
        Returns
        -------
        Self                
        """
        alias nelts: Int = simdwidthof[type]()
        var result = Tensor[type](self.shape)
        
        let lhs_last_shape = self.shape[len(self.shape)-1]
        let rhs_last_shape = rhs.shape[len(rhs.shape)-1]
        
        #tensor+tensor or vec+vec with same shape - treat as flat buffers, add all
        if __vector_compare(self.shape, rhs.shape):
            @parameter
            fn same_op[opsize: Int](n : Int):
                if op == 0:
                    result.store[opsize](__idx(n), add( self.load[opsize](__idx(n)) , rhs.load[opsize](__idx(n))) )
                elif op ==1:
                    result.store[opsize](__idx(n), sub( self.load[opsize](__idx(n)) , rhs.load[opsize](__idx(n))) )
                elif op ==2:
                    result.store[opsize](__idx(n), div( self.load[opsize](__idx(n)) , rhs.load[opsize](__idx(n))) )
                elif op ==3:
                    result.store[opsize](__idx(n), mul( self.load[opsize](__idx(n)) , rhs.load[opsize](__idx(n))) )                    
                else:
                    print("Operation not supported")
            vectorize[nelts, same_op](self.size)
            return result            

        #tensor + row vec 
        if rhs.rank==1 and rhs.shape[0] == lhs_last_shape:
            fn rowvec_op(indices: DynamicVector[Int]):
                #get the stride and offset for this row
                let stride = __get_dim_product(self.shape, self.rank-1)
                let offset = self.index_to_offset(indices)

                #vectorize a strided load (row), multiply w/ rhs(vector), and do strided_store
                @parameter
                fn rowop[opsize: Int](n : Int):
                    let cur_offset = offset+(n*stride)                        
                    let row = self.load_stride[opsize](cur_offset, stride)
                    
                    if op == 0:
                        result.store_stride[opsize](cur_offset, stride, add(row, rhs.load[opsize](__idx(n))) )
                    elif op ==1:
                        result.store_stride[opsize](cur_offset, stride, sub(row, rhs.load[opsize](__idx(n))) )
                    elif op ==2:
                        result.store_stride[opsize](cur_offset, stride, div(row, rhs.load[opsize](__idx(n))) )
                    elif op ==3:
                        result.store_stride[opsize](cur_offset, stride, mul(row, rhs.load[opsize](__idx(n))) )                        
                    else:
                        print("Operation not supported")
                           
                vectorize[nelts, rowop](rhs.shape[0])

            #run this for all rows in tensor    
            self.op_over_dimension(self.rank - 2, rowvec_op)                    
            return result
        
        #tensor + column vec(X,1)
        if rhs.rank==2 and rhs.shape[1]==1 and rhs.shape[0] == self.shape[self.rank-2]:
            fn colvec_op(indices: DynamicVector[Int]):
                #get the stride and offset for this row
                let stride = __get_dim_product(self.shape, self.rank-1)
                let offset = self.index_to_offset(indices)

                #vectorize a strided load (row), apply op w/ rhs[i], and do strided_store
                @parameter
                fn colop[opsize: Int](n : Int):
                    let cur_offset = offset+(n*stride)                        
                    let row = self.load_stride[opsize](cur_offset, stride)

                    let val = SIMD[type, opsize](rhs.load[1](__idx(indices[self.rank-2], 0)))
                    if op == 0:
                        result.store_stride[opsize](cur_offset, stride, add(row, val) )
                    elif op ==1:
                        result.store_stride[opsize](cur_offset, stride, sub(row, val) )
                    elif op ==2:
                        result.store_stride[opsize](cur_offset, stride, div(row, val) )
                    elif op ==3:
                        result.store_stride[opsize](cur_offset, stride, mul(row, val) )                        
                    else:
                        print("Operation not supported")
                           
                vectorize[nelts, colop](self.shape[self.rank-1])

            #run this for all rows in tensor    
            self.op_over_dimension(self.rank - 2, colvec_op)                    
            return result
        
        #tensor+matrix
        if rhs.rank == 2:
            var result = Tensor[type](self.shape)
            @parameter
            fn tensor_op(indices: DynamicVector[Int]):
                #get the strides (lhs stride may be bigger as lhs may be > R2)
                let stride = __get_dim_product(self.shape, self.rank-1)
                let rhs_stride = rhs.shape[0] #only need __get_dim_product(rhs.shape, rhs.rank-1) for rank > 2

                #get the offset - the location of the start of lhs and rhs rows
                let lhs_offset = self.index_to_offset(indices)
                let rhs_offset = indices[len(indices)-2] #only need rhs.index_to_offset(indices[len(indices)-2]) for rank>2
                      
                #load lhs row, load rhs row, mul, and store in result
                @parameter
                fn rowop_X2[opsize: Int](n : Int): 
                    let cur_lhs_offset = lhs_offset+(n*stride)
                    let lhs_row = self.load_stride[opsize](cur_lhs_offset, stride)
                    let rhs_row = rhs.load_stride[opsize](rhs_offset+(n*rhs_stride), rhs_stride)

                    if op == 0:
                        let product = add[type, opsize](lhs_row, rhs_row)
                        result.store_stride[opsize](cur_lhs_offset, stride, product)                                                        
                    elif op ==1:
                        let product = sub[type, opsize](lhs_row, rhs_row)                        
                        result.store_stride[opsize](cur_lhs_offset, stride, product)
                    elif op ==2:
                        let product = div[type, opsize](lhs_row, rhs_row)                        
                        result.store_stride[opsize](cur_lhs_offset, stride, product)
                    elif op ==3:
                        let product = mul[type, opsize](lhs_row, rhs_row)                        
                        result.store_stride[opsize](cur_lhs_offset, stride, product)                        
                    else:
                        print("Operation not supported")
                    
                vectorize[nelts, rowop_X2](lhs_last_shape)            
                
            #run this for all matrices(last two dims) in the tensor
            self.op_over_dimension(self.rank - 2, tensor_op)
            return result
        
        #change to failure
        print("Shapes not supported for broadcast op. ", __vector_to_string(self.shape), __vector_to_string(rhs.shape))
        return Tensor[type].zeros(1)      
    
    fn __add__(self: Self, rhs: Self) -> Self:
        """Add two tensors together. See broadcast_op."""
        return self.broadcast_op[0](rhs)

    # Computes 
    fn __mul__(self: Self, rhs: Self) -> Self:
        """Computes the hadamard product (https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) of two tensors"""
        return self.broadcast_op[3](rhs)

    fn __sub__(self: Self, rhs: Self) -> Self:
        """Substract two tensors"""
        return self.broadcast_op[0](rhs)

    fn __truediv__(self: Self, rhs: Self) -> Self:
        """Computes the division of two tensors."""
        return self.broadcast_op[3](rhs)
    
    
    ### Reduce ops        
    
    fn op_over_dimension(self, last_dim_index: Int, op: fn(DynamicVector[Int]) capturing-> None, ignore_dim_index:Int = -1 ):            
        """Utility fn to loop over this Tensor and execute an _op_ at a specified index.
        
        This method will loop over each element for every dimension in _self.shape_ up to last_dim_index.
        For every index combination, it will execute a passed in _op_.
        
        For a Tensor with shape (2,2,2), op_over_dimension(1, my_op), will execute my_op 4 times:
          my_op(0,0), my_op(0,1), my_op(1,0) and my_op(1,1).
        
        Arguments
        ----------
        last_dim_index: Int
            The index of the last dimension to loop over. If _self.shape_ is (A,B,C) and 
            last_dim_index=1, then op_over_dimension will only loop over A and B.
        op: fn(DynamicVector[Int]) capturing-> None
            The fn to execute. The fn must accept an index (DynamicVector[Int]) representing
            the current index and return no value.
            
        Returns
        -------
        SIMD[type, nelts]
        """
   
    
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
                

                #For all dimensions, except the last one
                for i in range(last_dim_index, -1, -1):
                    #if not the first dim and reached end, reset current and add one to previous
                    if indices[i] == self.shape[i] and i!=0:
                        indices[i]=0
                        indices[i-1]+=1
    ### NN Ops
    
    
    #TODO:
    # - implement tensor(>R2) . matrix or vec (extra dimension is the batch)
    # - More detail in docstrings
    fn dot(self: Self, rhs: Self) -> Self:
        """Compute the dot product between this tensor and a rhs tensor.
           Currently supports vec.vec, matrix.matrix, matrix.vec"""
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
            if not(self.shape[1] == rhs.shape[0]): print("In dot MM shapes not aligned:", __vector_to_string(self.shape), __vector_to_string(rhs.shape))
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
            if not(self.shape[1] == rhs.shape[0]): print("In dot, 1M shapes not aligned:", __vector_to_string(self.shape), __vector_to_string(rhs.shape[0]))
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
        print("failed. dot")
        return Tensor[type].zeros(1)    
    
    #implement Relu, Sigmoid, LeakyRelu, other activitations?
    
    #imlement backward
    """
    Backward should:
    - called only on a Scalar tensor
    - traverse the computation graph from here, building a list of nodes
    - for every node - reversed in the computation graph, run backward (which will set grads)
    - backward needs:
    - - Every tensor needs to set it's backward if result of an OP
    - - Grads applies to each Value in data, 


    For backward consider:
    - either making Tensor register passable : remove the need for DynamicVector shape, 
        (use either a customer register_passable struct, or a DimList)
        - or making a TensorOpRef, that is very small (left, right, op) only, close to this
        https://docs.modular.com/mojo/programming-manual.html#register_passable-struct-decorator
        self may be unstable?
    - Using Type erasure, pointers to Int, then bitcasting pointer to Tensor when using
        - manually calling clean-up memory where needed (
    - 
    """