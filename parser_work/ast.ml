type bop =
| Add
| Subtract
| Multiply
| Divide
[@@deriving yojson]

type expr =
| Einsum of einsum
| BinOp of bop * expr * expr
| Tensor of string
[@@deriving yojson]

and einsum = {
  notation: string;
  tensors: expr list;
}
[@@deriving yojson]