open Printf

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
  "A + B + C";
  "einsum(\"ij,jk->ik\", A, B) + einsum(\"kl,lm->km\", C, D)";
]

(* Main function *)
let () =
  printf "=== Einsum Expression Analyzer with SSA IR ===\n\n";
  
  (* Test predefined cases *)
  printf "=== Testing predefined cases ===\n";
  List.iter Ssa.analyze_expression_with_ssa test_cases;
  
  (* Read and analyze user input *)
  printf "=== Enter your expression ===\n";
  printf "Enter an expression to analyze: ";
  flush_all ();
  let user_input = read_line () in
  if user_input <> "" then (
    printf "\n=== Your input analysis ===\n";
    Ssa.analyze_expression_with_ssa user_input
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
        Ssa.analyze_expression_with_ssa input
    done
  with
  | End_of_file -> printf "\nGoodbye!\n"