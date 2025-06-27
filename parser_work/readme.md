To run test
#### First, make sure all your files are compiled
```
ocamllex lexer.mll
menhir parser.mly
ocamlc -c ast.ml
ocamlc -c parser.mli
ocamlc -c parser.ml
ocamlc -c lexer.ml
```

#### Compile the test program

`ocamlc -c tetricks_parser.ml`

#### Link everything together to create an executable
`ocamlc -o test ast.cmo parser.cmo lexer.cmo tetricks_parser.cmo`

#### Run the program
`./test`
