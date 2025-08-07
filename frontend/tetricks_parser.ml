open Ast

let () =
  try
    let ast = Parser.main Lexer.read (Lexing.from_channel stdin) in
    let json = Ast.expr_to_yojson ast in
    
    Yojson.Safe.pretty_to_channel stdout json;
    print_newline ()
  with
  | Parser.Error ->
      Printf.eprintf "Parse error\n";
      exit 1
  | Lexer.Error msg ->
      Printf.eprintf "Lexical error: %s\n" msg;
      exit 1
  | e ->
      Printf.eprintf "Error: %s\n" (Printexc.to_string e);
      exit 1