type bop =
| Add
| Subtract
| Multiply
| Divide

type expr =
| Einsum of einsum
| BinOp of bop * expr * expr (*Does not support BinOp between two tensors but handles later in the IR stage*)
| Tensor of string

and einsum = {
  notation: string;
  operands: expr list;
}