type bop =
| Add
| Subtract
| Multiply
| Divide

type expr =
| Einsum of einsum
| BinOp of bop * expr * expr
| Tensor of string

and einsum = {
  notation: string;
  tensors: expr list;
}