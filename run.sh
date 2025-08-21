input_file="inputs.txt"
frontend/_build/default/tetricks_frontend.exe < $input_file > ast.json
sed -i 's/-->/->/g' ast.json
./backend/tetricks_backend ast.json