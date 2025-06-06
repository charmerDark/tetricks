type einsum_notation = string list

type bop =
| Add
| Subtract
| Multiply
| Divide

type expr =
| Einsum of einsum
| BinOp of bop * expr * expr
| Variable of string

and einsum = {
  notation: einsum_notation;
  tensors: string list;
}