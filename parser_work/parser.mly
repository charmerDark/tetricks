%{
open Ast

(* Helper function to parse einsum notation string *)
let parse_notation str =
  let parts = String.split_on_char '>' str in
  match parts with
  | [input_part; output_part] ->
      let inputs = String.split_on_char ',' (String.sub input_part 0 (String.length input_part - 1)) in
      let output = String.trim output_part in
      let output_list = if output = "" then [] else [output] in
      List.map String.trim inputs @ output_list
  | _ -> failwith ("Invalid einsum notation: " ^ str)
%}

%token <string> IDENT
%token <string> STRING
%token EINSUM COMMA PLUS MINUS TIMES DIVIDE LPAREN RPAREN
%token EOF

%left PLUS MINUS
%left TIMES DIVIDE

%type <Ast.expr> expr
%type <Ast.expr> einsum_expr
%type <string list> tensor_list

%start <Ast.expr> main

%%

main:
| e=expr EOF { e }

expr:
| e=einsum_expr { e }
| id=IDENT { Variable id }
| e1=expr PLUS e2=expr { BinOp (Add, e1, e2) }
| e1=expr MINUS e2=expr { BinOp (Subtract, e1, e2) }
| e1=expr TIMES e2=expr { BinOp (Multiply, e1, e2) }
| e1=expr DIVIDE e2=expr { BinOp (Divide, e1, e2) }
| LPAREN e=expr RPAREN { e }

einsum_expr:
| EINSUM LPAREN notation=STRING COMMA tensors=tensor_list RPAREN {
    Einsum { notation = parse_notation notation; tensors = tensors }
}

tensor_list:
| t=IDENT COMMA tl=tensor_list { t :: tl }
| t=IDENT { [t] }