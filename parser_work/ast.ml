type bop =
| Add
| Subtract
| Multiply
| Divide
[@@deriving yojson]

type tensor = string
[@@deriving yojson]

type expr =
| Einsum of einsum
| BinOp of bop * expr * expr
| Tensor of tensor
[@@deriving yojson]

and einsum = {
  notation: string;
  tensors: tensor list;
}
[@@deriving yojson]