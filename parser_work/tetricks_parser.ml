open Printf

(* Function to convert token to string for display *)
let string_of_token = function
  | Parser.EINSUM -> "EINSUM"
  | Parser.COMMA -> "COMMA"
  | Parser.PLUS -> "PLUS"
  | Parser.MINUS -> "MINUS"
  | Parser.TIMES -> "TIMES"
  | Parser.DIVIDE -> "DIVIDE"
  | Parser.LPAREN -> "LPAREN"
  | Parser.RPAREN -> "RPAREN"
  | Parser.IDENT s -> sprintf "IDENT(%s)" s
  | Parser.STRING s -> sprintf "STRING(%s)" s
  | Parser.EOF -> "EOF"

(* Function to tokenize a string *)
let tokenize str =
  let lexbuf = Lexing.from_string str in
  let rec loop acc =
    match Lexer.read lexbuf with
    | Parser.EOF -> List.rev (Parser.EOF :: acc)
    | token -> loop (token :: acc)
  in
  loop []

(* Function to convert AST to string for display *)
let rec string_of_expr = function
  | Ast.Einsum e ->
      sprintf "Einsum{notation=\"%s\"; tensors=[%s]}"
        e.notation
        (String.concat "; " (List.map string_of_expr e.tensors))
  | Ast.BinOp (op, e1, e2) ->
      let op_str = match op with
        | Ast.Add -> "Add"
        | Ast.Subtract -> "Subtract"
        | Ast.Multiply -> "Multiply"
        | Ast.Divide -> "Divide"
      in
      sprintf "BinOp(%s, %s, %s)" op_str
        (string_of_expr e1) (string_of_expr e2)
  | Ast.Tensor name -> sprintf "Tensor(%s)" name

(* Function to parse a string *)
let parse_string str =
  let lexbuf = Lexing.from_string str in
  try
    Some (Parser.main Lexer.read lexbuf)
  with
  | e -> printf "Parse error: %s\n" (Printexc.to_string e); None

(* Function to analyze an expression *)
let analyze_expression expr =
  printf "=== Analyzing: %s ===\n" expr;
  
  (* Show tokens *)
  printf "Tokens: ";
  let tokens = tokenize expr in
  List.iter (fun token -> printf "%s " (string_of_token token)) tokens;
  printf "\n";
  
  (* Show AST *)
  match parse_string expr with
  | Some ast ->
      printf "AST: %s\n" (string_of_expr ast)
  | None ->
      printf "Failed to parse\n";
  printf "\n"

(* Test cases *)
let test_cases = [
  "A";
  "einsum(\"i->i\", A)";
  "einsum(\"ij,jk->ik\", A, B)";
  "einsum(\"ij,jk,kl->il\", A, B, C)";
  "einsum(\"ij->\", A)";
  "A + B";
  "einsum(\"ij,jk->ik\", A, B) + C";
  "einsum(\"ij,jk->ik\", A, B) * einsum(\"kl->l\", D)";
  "(A + B) * C";
  "einsum(\"lmnop -> lp\", Q + T)";  (* New test case! *)
]

(* Main function *)
let () =
  printf "=== Einsum Expression Analyzer ===\n\n";
  
  (* Test predefined cases *)
  printf "=== Testing predefined cases ===\n";
  List.iter analyze_expression test_cases;
  
  (* Read and analyze user input *)
  printf "=== Enter your expression ===\n";
  printf "Enter an expression to analyze: ";
  flush_all ();
  let user_input = read_line () in
  if user_input <> "" then (
    printf "\n=== Your input analysis ===\n";
    analyze_expression user_input
  );
  
  (* Interactive mode *)
  printf "=== Interactive mode ===\n";
  printf "Enter more expressions to analyze (empty line to exit):\n";
  try
    while true do
      printf "> ";
      flush_all ();
      let input = read_line () in
      if input = "" then (
        exit 0
      ) else
        analyze_expression input
    done
  with
  | End_of_file -> printf "\nGoodbye!\n"