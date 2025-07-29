let () =
  let ast = Parser.main Lexer.read (Lexing.from_channel stdin) in
  let json = Ast.expr_to_yojson ast in
  Yojson.Safe.pretty_to_channel stdout json