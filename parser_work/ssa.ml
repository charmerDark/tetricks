(* SSA IR Definition *)
type ssa_value = 
  | Variable of string          (* Original variables: A, B, C *)
  | Temporary of int           (* Generated temporaries: %1, %2, %3 *)

type ssa_instruction =
  | Einsum of ssa_value * string list * ssa_value list  (* result = einsum(notation, operands) *)
  | BinOp of ssa_value * Ast.bop * ssa_value * ssa_value (* result = op lhs rhs *)

type ssa_program = ssa_instruction list

(* State for SSA generation *)
type ssa_state = {
  instructions: ssa_instruction list;
  next_temp: int;
}

(* Create a new temporary variable *)
let fresh_temp state =
  let temp = Temporary state.next_temp in
  let new_state = { state with next_temp = state.next_temp + 1 } in
  (temp, new_state)

(* Add an instruction to the program *)
let add_instruction state instr =
  { state with instructions = instr :: state.instructions }

(* Convert AST expression to SSA IR *)
let rec expr_to_ssa state expr =
  match expr with
  | Ast.Variable name ->
      (* Variables are just referenced, no instruction needed *)
      (Variable name, state)
      
  | Ast.Einsum einsum ->
      let result, state = fresh_temp state in
      let operands = List.map (fun name -> Variable name) einsum.tensors in
      let instr = Einsum (result, einsum.notation, operands) in
      let state = add_instruction state instr in
      (result, state)
      
  | Ast.BinOp (op, lhs, rhs) ->
      (* Convert left and right operands to SSA *)
      let lhs_val, state = expr_to_ssa state lhs in
      let rhs_val, state = expr_to_ssa state rhs in
      (* Create result temporary *)
      let result, state = fresh_temp state in
      let instr = BinOp (result, op, lhs_val, rhs_val) in
      let state = add_instruction state instr in
      (result, state)

(* Convert AST to SSA program *)
let ast_to_ssa ast =
  let initial_state = { instructions = []; next_temp = 1 } in
  let final_result, final_state = expr_to_ssa initial_state ast in
  (* Reverse to get instructions in execution order *)
  let program = List.rev final_state.instructions in
  (program, final_result)

(* Pretty printing functions *)
let string_of_ssa_value = function
  | Variable name -> name
  | Temporary id -> Printf.sprintf "%%t%d" id

let string_of_bop = function
  | Ast.Add -> "add"
  | Ast.Subtract -> "sub" 
  | Ast.Multiply -> "mul"
  | Ast.Divide -> "div"

let string_of_ssa_instruction = function
  | Einsum (result, notation, operands) ->
      Printf.sprintf "%s = einsum([%s], [%s])"
        (string_of_ssa_value result)
        (String.concat "; " notation)
        (String.concat "; " (List.map string_of_ssa_value operands))
  | BinOp (result, op, lhs, rhs) ->
      Printf.sprintf "%s = %s %s %s"
        (string_of_ssa_value result)
        (string_of_bop op)
        (string_of_ssa_value lhs)
        (string_of_ssa_value rhs)

let print_ssa_program program final_result =
  Printf.printf "=== SSA IR ===\n";
  List.iteri (fun i instr ->
    Printf.printf "%d: %s\n" i (string_of_ssa_instruction instr)
  ) program;
  Printf.printf "Final result: %s\n" (string_of_ssa_value final_result);
  Printf.printf "\n"

(* Extended analysis function that includes SSA generation *)
let analyze_expression_with_ssa expr =
  Printf.printf "=== Analyzing: %s ===\n" expr;
  
  (* Tokenization *)
  let tokenize str =
    let lexbuf = Lexing.from_string str in
    let rec loop acc =
      match Lexer.read lexbuf with
      | Parser.EOF -> List.rev (Parser.EOF :: acc)
      | token -> loop (token :: acc)
    in
    loop []
  in
  
  let string_of_token = function
    | Parser.EINSUM -> "EINSUM"
    | Parser.COMMA -> "COMMA"
    | Parser.PLUS -> "PLUS"
    | Parser.MINUS -> "MINUS"
    | Parser.TIMES -> "TIMES"
    | Parser.DIVIDE -> "DIVIDE"
    | Parser.LPAREN -> "LPAREN"
    | Parser.RPAREN -> "RPAREN"
    | Parser.IDENT s -> Printf.sprintf "IDENT(%s)" s
    | Parser.STRING s -> Printf.sprintf "STRING(%s)" s
    | Parser.EOF -> "EOF"
  in
  
  Printf.printf "Tokens: ";
  let tokens = tokenize expr in
  List.iter (fun token -> Printf.printf "%s " (string_of_token token)) tokens;
  Printf.printf "\n";
  
  (* Parsing *)
  let parse_string str =
    let lexbuf = Lexing.from_string str in
    try Some (Parser.main Lexer.read lexbuf)
    with e -> Printf.printf "Parse error: %s\n" (Printexc.to_string e); None
  in
  
  let rec string_of_expr = function
    | Ast.Einsum e -> 
        Printf.sprintf "Einsum{notation=[%s]; tensors=[%s]}"
          (String.concat "; " e.notation)
          (String.concat "; " e.tensors)
    | Ast.BinOp (op, e1, e2) ->
        let op_str = match op with
          | Ast.Add -> "Add"
          | Ast.Subtract -> "Subtract" 
          | Ast.Multiply -> "Multiply"
          | Ast.Divide -> "Divide"
        in
        Printf.sprintf "BinOp(%s, %s, %s)" op_str 
          (string_of_expr e1) (string_of_expr e2)
    | Ast.Variable v -> Printf.sprintf "Variable(%s)" v
  in
  
  match parse_string expr with
  | Some ast -> 
      Printf.printf "AST: %s\n" (string_of_expr ast);
      
      (* Generate SSA IR *)
      let ssa_program, final_result = ast_to_ssa ast in
      print_ssa_program ssa_program final_result;
      
  | None -> 
      Printf.printf "Failed to parse\n";
  Printf.printf "\n"