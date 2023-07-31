from DType import DType
from String import String
from Memory import memset_zero, memcpy
from Pointer import DTypePointer
from Vector import DynamicVector
from List import Dim, DimList, VariadicList

#this is part of a hack until traits, a Dimlist I can make register_passable w/ some utility functions
@value
@register_passable
struct VDimList:
    """Temp struct to keep a list of dimensions, with some helper functions"""
    var data: DTypePointer[DType.int8]
    var size: Int    
    var test1: Int
    var test2: Int

    fn __init__(l: VariadicList[Int]) -> Self:
        let size = len(l)
        let p = DTypePointer[DType.int8].alloc(size*2+1)
        #store dimensions and then products
        
        var j = size
        p.store(size, 1)            
        for i in range(size):
            p.store(i, l[i])
            p.store(j+1, p.load(j)*l[i])
            print("sotring: ", p.load(j), " * ", l[i], " at: ", j+1)            
            j+=1

        return Self(p, size, l[0], l[1])
    
    fn __init__(p: DTypePointer[DType.int8], l: VariadicList[Int]) -> Self:    
        let size = len(l)
        #store dimensions and then products
        
        var j = size
        p.store(size, 1)            
        for i in range(size):
            p.store(i, l[i])
            p.store(j+1, p.load(j)*l[i])
            print("sotring: ", p.load(j), " * ", l[i], " at: ", j+1)            
            j+=1        
        return Self(p, size, l[0], l[1])
        
    fn __init__(*dims: Int) -> Self:
        let l = VariadicList(dims)        
        return Self(l)
    
    fn __copyinit__(self) -> Self:
        let data = DTypePointer[DType.int8].alloc(self.size)
        memcpy(data, self.data, self.size)        
        return Self(data, self.size, self.test1, self.test2)
            
    fn zeros_like(self: Self) -> Self:
        let p = DTypePointer[DType.int8].alloc(self.size*2+1)
        memset_zero(p, self.size)
        return VDimList(p, self.size, self.test1, self.test2)
                    
    fn __del__(owned self: Self):
        self.data.free()
    
    fn __len__(self: Self) -> Int:
        return self.size
    
    fn __eq__(self: Self, rhs: Self) -> Bool:
        if self.size != rhs.__len__(): return False

        for i in range(self.size):
            if self[i]!=rhs[i]: return False

        return True
        
    fn __getitem__(self: Self, idx: Int) -> Int:
        return self.data.load(idx).to_int()

    fn __setitem__(self: Self, idx: Int, val:Int):
        return self.data.store(idx, val)

    fn get_product(self: Self, rank: Int) -> Int:
        var size = self[0]
        for i in range(1, rank):
            size *= self[i]
        return size
    
    fn get_products(self: Self) -> DynamicVector[Int]:
        var products = DynamicVector[Int]()
        #fetch products
        print("t:", self.test1, self.test2)
        for i in range(self.size, self.size*2+1):
            #print("Fetching product at ", i, " val: ", self[i])
            products.push_back(self[i])
        return products
    
    fn to_string(self: Self) -> String:
        var out = String("[")
        for i in range(self.size):
            out += String(self[i])
            out+= ", "
        return out[0:len(out)-2]+"]"        