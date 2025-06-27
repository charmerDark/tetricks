# Loop IR Design

# Functionality required
    Must be able to represent loop structure well enough to be analysed
        - Loop variable - can be same as einsum variable
        - loop range - inferred from einum onject passed
        - ability to do loop fusion, tiling, unrollling jamming etc


# Language Design

## Operations needed

    Einsum functionality 
    ~~Addition~~ Element wise operations


## candidates
### taco style
    - A(i,j) = B(i,k) + C(k,j) # Matrix Addition
    -A(i,j) = B(j,i ) # transpose
    -b = A(i,j) # sum over all elements
    -B(j) = A(i,j) # column sum
    -B(i) = A(i,j) # row sum 

- Basically taco syntax again
- How to show data types

### einsum notation style
    - ik, kj -> ij # matrix addition
    - ij ->ji # transpose
    -  ij-># sum over all elements
    - ij -> i # row sum
    - ij -> j # column sum
    - ik, k -> i #matrix vector multiplicaiton
    - ijk, ikl -> ijl # batched matrix multiplication


- how to show data types
- how to integrate into other operations?

### Other issues
- Need more information like matrix size for decisions like tiling. (how did torch comprehension handle this?)
- Is element wise addition the only other operation I need?


TACO command terminal seems to have ignored the data type issue completely??

### Possible candidate with element wise operations

    - (ij, jk -> ik, Res, A, B) + C # matrix multiplication and then element wise addition


### References

- Guide to basic einsum \([link](https://ajcr.net/Basic-guide-to-einsum/)\)
- Good einsum examples  \([link](https://rockt.ai/2018/04/30/einsum)\)
- Possible extension \([link](https://github.com/arogozhnikov/einops)\)
