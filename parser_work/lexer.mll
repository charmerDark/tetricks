{
open Parser
exception Error of string
}

let white = [' ' '\t']+

rule read = parse
| white { read lexbuf }
| "einsum" { EINSUM }
| ',' { COMMA }
| '+' { PLUS }
| '-' { MINUS }
| '*' { TIMES }
| '/' { DIVIDE }
| '"' { read_string "" lexbuf }
| '(' { LPAREN }
| ')' { RPAREN }
| ['a'-'z' 'A'-'Z']+ as id { IDENT id }
| eof { EOF }

and read_string acc = parse
| '"' { STRING acc }
| [^'"']+ as s { read_string (acc ^ s) lexbuf }
| eof { failwith "Unterminated string" }