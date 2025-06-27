%{
open Ast

let normalize_notation str =
  let trimmed = String.trim str in
  if String.contains trimmed '>' then
    (* Handle "ijk-> " -> "ijk->" *)
    let parts = String.split_on_char '>' trimmed in
    match parts with
    | [input; output] -> 
        let clean_input = String.trim input in
        let clean_output = String.trim output in
        clean_input ^ "->" ^ clean_output
    | _ -> trimmed
  else trimmed
%}

%token <string> IDENT
%token <string> STRING
%token EINSUM COMMA PLUS MINUS TIMES DIVIDE LPAREN RPAREN
%token EOF

%left PLUS MINUS
%left TIMES DIVIDE

%type <Ast.expr> expr
%type <Ast.expr> einsum_expr
%type <Ast.expr list> expr_list 
%start <Ast.expr> main


%%

main:
| e=expr EOF { e }

expr:
| e=einsum_expr { e }
| id=IDENT { Tensor id } 
| e1=expr PLUS e2=expr { BinOp (Add, e1, e2) }
| e1=expr MINUS e2=expr { BinOp (Subtract, e1, e2) }
| e1=expr TIMES e2=expr { BinOp (Multiply, e1, e2) }
| e1=expr DIVIDE e2=expr { BinOp (Divide, e1, e2) }
| LPAREN e=expr RPAREN { e }

einsum_expr:
| EINSUM LPAREN notation=STRING COMMA tensors=expr_list RPAREN {
    Einsum { notation = normalize_notation notation; tensors = tensors }
}

expr_list:
| e=expr COMMA el=expr_list { e :: el }
| e=expr { [e] }