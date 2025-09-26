"""
This program is a mapper that consumes IR representations from the tetricks backend and provdes a mapping after applying the required code hoisting for pointer arithmetic. 
"""
import re
import sys
from copy import deepcopy
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass


class IRNode:
    """Base class for all IR nodes
    IR Nodes are generated from the tetricks DSL for tensor operations.
    The DSL supports einsum operations and elementwise +, -, * and / operations
    that are broken down into SequenceNodes, LoopNodes and CalcNodes as necessary by the backend compiler.    
    """
    pass

@dataclass
class ShapeVariable:
    """
    Mirrors the shapevariable from tetricks backend - might not be necessary
    """
    tensor_name: str
    idx_posn: int

    def __str__(self):
        return f"{self.tensor_name}.shape[{self.idx_posn}]"

@dataclass
class LoopNode(IRNode):
    loop_variable: str
    shape_var: ShapeVariable
    body: Optional[IRNode] = None
    start: int = 0
    increment: int = 1
    end_offset: int = 0

    def __str__(self):
        return (f'LoopNode(var="{self.loop_variable}", shape_var={self.shape_var}, '
                f'start={self.start}, increment={self.increment}, end_offset={self.end_offset})')


@dataclass
class CalcNode(IRNode):
    """Represents a calculation node with a single code line"""
    code_line: str
    
    def __str__(self):
        return f'CalcNode("{self.code_line}")'
    


@dataclass
class SequenceNode(IRNode):
    """Represents a sequence of operations with temp declarations"""
    operations: List[IRNode]
    temp_declarations: Dict[str, str]  # name -> declaration
    result_temp: str = ""
    
    def __init__(self):
        self.operations = []
        self.temp_declarations = {}
        self.result_temp = ""
    
    def __str__(self):
        return f'SequenceNode({len(self.operations)} operations, {len(self.temp_declarations)} temps)'


class IRParseError(Exception):
    pass


class IRParser:
    """Parser for IR text format"""
    
    def __init__(self, text: str):
        self.lines = text.strip().split('\n')
        self.current_line = 0
        
    def parse(self) -> IRNode:
        """Parse the IR text and return the root node"""
        try:
            return self._parse_node()
        except IndexError:
            raise IRParseError("Unexpected end of input")
    
    def _current_line_content(self) -> str:
        """Get current line content, stripping whitespace"""
        if self.current_line >= len(self.lines):
            return ""
        return self.lines[self.current_line].strip()
    
    def _advance_line(self):
        """Move to next line"""
        self.current_line += 1
    
    def _get_indentation(self, line: str) -> int:
        """Get indentation level of a line"""
        return len(line) - len(line.lstrip())
    
    def _skip_empty_and_comments(self):
        """Skip empty lines and comment-only lines"""
        while self.current_line < len(self.lines):
            line = self._current_line_content()
            if line == "" or line.startswith("//"):
                self._advance_line()
            else:
                break
    
    def _parse_node(self) -> IRNode:
        """Parse a single IR node
            Entry point to specialised node parsers
        """
        self._skip_empty_and_comments()
        
        if self.current_line >= len(self.lines):
            raise IRParseError("No content to parse")
            
        line = self._current_line_content()
        
        if line.startswith("SequenceNode {"):
            return self._parse_sequence_node()
        elif line.startswith("LoopNode {"):
            return self._parse_loop_node()
        elif line.startswith("CalcNode {"):
            return self._parse_calc_node()
        else:
            raise IRParseError(f"Unknown node type at line {self.current_line + 1}: {line}")
    
    def _parse_sequence_node(self) -> SequenceNode:
        """Parse a SequenceNode"""
        # Consume the opening line
        self._advance_line()
        
        node = SequenceNode()
        
        while self.current_line < len(self.lines):
            self._skip_empty_and_comments()
            
            if self.current_line >= len(self.lines):
                break
                
            line = self._current_line_content()
            
            # Check for closing brace
            if line == "}":
                self._advance_line()
                break
            
            # Parse temp declarations
            if self._is_temp_declaration(line):
                name, decl = self._parse_temp_declaration(line)
                node.temp_declarations[name] = decl
                self._advance_line()
            
            # Parse result temp comment
            elif line.startswith("// Result stored in:"):
                node.result_temp = self._extract_result_temp(line)
                self._advance_line()
            
            # Parse child nodes (operations)
            elif any(line.startswith(node_type) for node_type in ["SequenceNode {", "LoopNode {", "CalcNode {"]):
                child_node = self._parse_node()
                node.operations.append(child_node)
            
            else:
                # Skip unknown lines or advance to avoid infinite loop
                self._advance_line()
        
        return node
    
    def _parse_loop_node(self) -> LoopNode:
        self._advance_line()
        loop_var = ""
        shape_var = None
        start = 0
        increment = 1
        end_offset = 0
        body = None

        while self.current_line < len(self.lines):
            self._skip_empty_and_comments()
            if self.current_line >= len(self.lines):
                break
            line = self._current_line_content()
            if line == "}":
                self._advance_line()
                break

            if line.startswith("variable:"):
                loop_var = self._extract_quoted_value(line, "variable:")
                self._advance_line()
            elif line.startswith("shape_var:"):
                # Parse e.g. "A.shape[0]"
                shape_str = line.replace("shape_var:", "").strip()
                match = re.match(r"(\w+)\.shape\[(\d+)\]", shape_str)
                if match:
                    tensor_name = match.group(1)
                    idx_posn = int(match.group(2))
                    shape_var = ShapeVariable(tensor_name, idx_posn)
                else:
                    raise IRParseError(f"Invalid shape_var format: {shape_str}")
                self._advance_line()
            elif line.startswith("start:"):
                start = int(line.replace("start:", "").strip())
                self._advance_line()
            elif line.startswith("increment:"):
                increment = int(line.replace("increment:", "").strip())
                self._advance_line()
            elif line.startswith("end_offset:"):
                end_offset = int(line.replace("end_offset:", "").strip())
                self._advance_line()
            elif line.startswith("body:"):
                self._advance_line()
                self._skip_empty_and_comments()
                if self.current_line < len(self.lines):
                    body = self._parse_node()
            elif any(line.startswith(node_type) for node_type in ["SequenceNode {", "LoopNode {", "CalcNode {"]):
                body = self._parse_node()
            else:
                self._advance_line()

        if shape_var is None:
            raise IRParseError("LoopNode missing shape_var")
        return LoopNode(loop_variable=loop_var, shape_var=shape_var, body=body,
                        start=start, increment=increment, end_offset=end_offset)


    def _parse_calc_node(self) -> CalcNode:
        """Parse a CalcNode"""
        line = self._current_line_content()
        
        # Extract code from CalcNode { "code" } format
        match = re.search(r'CalcNode\s*\{\s*"(.*)"\s*\}', line)
        if match:
            code = match.group(1)
            # Handle escaped quotes
            code = code.replace('\\"', '"')
        else:
            raise IRParseError(f"Invalid CalcNode format at line {self.current_line + 1}: {line}")
        
        self._advance_line()
        return CalcNode(code_line=code)
    
    def _is_temp_declaration(self, line: str) -> bool:
        """Check if line is a temp variable declaration
            Currently ased off the fact that backend names all temps as tempX. slightly hacky
        """
        # Look for patterns like "double tempX[...];" 
        return (line.startswith("double ") and 
                ("temp" in line or line.endswith(";")))
    
    def _parse_temp_declaration(self, line: str) -> Tuple[str, str]:
        """Parse temp declaration and extract name and full declaration
        """
        # Extract variable name (everything between "double " and first "[" or "=")
        match = re.search(r'double\s+(\w+)', line)
        if match:
            name = match.group(1)
            return name, line
        else:
            raise IRParseError(f"Invalid temp declaration: {line}")
    
    def _extract_result_temp(self, line: str) -> str:
        """Extract result temp name from comment"""
        # Pattern: "// Result stored in: tempX"
        match = re.search(r'//\s*Result stored in:\s*(\w+)', line)
        if match:
            return match.group(1)
        return ""
    
    def _extract_quoted_value(self, line: str, prefix: str) -> str:
        """Extract quoted value after a prefix"""
        # Remove prefix and find quoted content
        content = line.replace(prefix, "").strip()
        match = re.search(r"'([^']*)'", content)
        if match:
            return match.group(1)
        return content.strip("'\"")

def pretty_print_ir(node: IRNode, indent: int = 0) -> str:
    indent_str = "  " * indent

    if isinstance(node, SequenceNode):
        result = f"{indent_str}SequenceNode {{\n"
        if node.temp_declarations:
            result += f"{indent_str}  // Temporary declarations:\n"
            for name, decl in node.temp_declarations.items():
                result += f"{indent_str}  {decl}\n"
            result += "\n"
        for i, op in enumerate(node.operations):
            result += pretty_print_ir(op, indent + 1)
            if i < len(node.operations) - 1:
                result += "\n"
        if node.result_temp:
            result += f"\n{indent_str}  // Result: {node.result_temp}\n"
        result += f"{indent_str}}}\n"

    elif isinstance(node, LoopNode):
        result = f"{indent_str}LoopNode {{\n"
        result += f"{indent_str}  variable: '{node.loop_variable}'\n"
        result += f"{indent_str}  shape_var: {node.shape_var}\n"
        result += f"{indent_str}  start: {node.start}\n"
        result += f"{indent_str}  increment: {node.increment}\n"
        result += f"{indent_str}  end_offset: {node.end_offset}\n"
        if node.body:
            result += f"{indent_str}  body:\n"
            result += pretty_print_ir(node.body, indent + 2)
        result += f"{indent_str}}}\n"

    elif isinstance(node, CalcNode):
        result = f"{indent_str}CalcNode {{ \"{node.code_line}\" }}\n"

    else:
        result = f"{indent_str}Unknown node type: {type(node)}\n"

    return result


def parse_ir_file(filename: str) -> IRNode:
    """Parse IR from a file"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        parser = IRParser(content)
        return parser.parse()
        
    except FileNotFoundError:
        raise IRParseError(f"File not found: {filename}")
    except Exception as e:
        raise IRParseError(f"Error parsing file {filename}: {str(e)}")

def pointer_arithmetic_hoisting_pass(node: IRNode, loop_vars=None) -> IRNode:
    """
    Performs pointer arithmetic hoisting for all arrays in innermost loops of the IR.

    This pass transforms the IR to make innermost loops more suitable for hardware mapping
    (e.g., CGRA acceleration) by:
      - Hoisting the computation of base pointers for all arrays accessed in the innermost loop
        to just outside the innermost loop.
      - Replacing multi-dimensional array accesses in the innermost loop with pointer dereferences
        (e.g., '*ptr_A' instead of 'A[i][j][k][l]').
      - Inserting pointer bumping statements (e.g., 'ptr_A += stride_A_2;') after each use
        in the innermost loop.

    The transformation is applied recursively to the entire IR tree, and supports:
      - Multiple innermost loops in the same or different loop nests.
      - Multiple array accesses (inputs and outputs) in a single CalcNode.
      - Wrapping multiple statements in SequenceNodes as needed to preserve IR structure.

    Parameters
    ----------
    node : IRNode
        The root of the IR tree to transform. This can be a SequenceNode, LoopNode, or CalcNode.
    loop_vars : list of str, optional
        The stack of loop variables currently in scope (used internally for recursion).
        Users should not set this parameter.

    Returns
    -------
    IRNode
        A new IR tree with pointer arithmetic hoisted for all arrays in innermost loops.

    Algorithm
    ---------
    - Recursively traverse the IR tree.
    - For each innermost LoopNode (i.e., a loop whose body is a CalcNode or a SequenceNode of only CalcNodes):
        - Identify all arrays accessed in the loop body.
        - For each array:
            - Insert a pointer setup CalcNode (e.g., 'ptr_A = &A[i][j][0][l];') before the innermost loop,
              with the innermost index set to 0.
            - In the innermost loop body, replace all accesses to that array with '*ptr_A'.
            - After each use, insert a pointer bump CalcNode (e.g., 'ptr_A += stride_A_2;'), where the stride
              variable is named according to the array and the innermost dimension.
        - If the loop body or its parent now contains multiple statements, wrap them in a SequenceNode.
    - For all other nodes, recursively process their children and rebuild the IR.

    Naming Conventions
    ------------------
    - Pointer variables: 'ptr_<arrayname>' (e.g., 'ptr_A').
    - Stride variables: 'stride_<arrayname>_<dim>' (e.g., 'stride_A_2'), where <dim> is the index of the innermost loop variable in the array access.

    Limitations
    -----------
    - Only innermost loops are transformed; outer loops are left unchanged except for pointer setup insertion.
    - Stride variables are assumed to exist or be defined elsewhere.
    - Only supports IRs composed of SequenceNode, LoopNode, and CalcNode.
    - Does not handle imperfect loop nests or advanced pointer bumping (e.g., for vectorized inner loops).

    Examples
    --------
    Before:
        LoopNode('i', ...)
          LoopNode('j', ...)
            LoopNode('k', ...)
              CalcNode("temp1[i][j] += A[i][j][k];")

    After:
        LoopNode('i', ...)
          LoopNode('j', ...)
            SequenceNode([
                CalcNode("ptr_A = &A[i][j][0];"),
                CalcNode("ptr_temp1 = &temp1[i][j];"),
                LoopNode('k', ...)
                  SequenceNode([
                      CalcNode("*ptr_temp1 += *ptr_A;"),
                      CalcNode("ptr_A += stride_A_2;")
                  ])
            ])

    """
    # ... implementation ...

    if loop_vars is None:
        loop_vars = []

    # Helper: Find all array accesses in a code line
    def find_array_accesses(code_line):
        # Matches e.g. A[i][j][k], temp1[i][l], etc.
        pattern = r'([A-Za-z_]\w*)\s*(\[[^\]]+\])+'
        matches = re.finditer(pattern, code_line)
        accesses = []
        for m in matches:
            array_name = m.group(1)
            indices = re.findall(r'\[([^\]]+)\]', m.group(0))
            accesses.append((array_name, indices))
        return accesses

    # Helper: Replace array accesses with pointer dereference in code line
    def replace_with_pointer(code_line, arrays):
        # For each array, replace all occurrences of array access with *ptr_<array>
        for array_name, indices in arrays:
            # Build regex for this array access
            access_pattern = re.escape(array_name) + r'(\s*(\[[^\]]+\])+)'  # e.g. A[[...]]
            code_line = re.sub(access_pattern, f'*ptr_{array_name}', code_line)
        return code_line

    # Helper: Build pointer setup code
    def pointer_setup_code(array_name, indices, innermost_var):
        # Set innermost index to 0 for pointer setup
        setup_indices = []
        for idx in indices:
            if idx == innermost_var:
                setup_indices.append('0')
            else:
                setup_indices.append(idx)
        return f'ptr_{array_name} = &{array_name}[' + ']['.join(setup_indices) + '];'

    # Helper: Build pointer bump code
    def pointer_bump_code(array_name, innermost_dim):
        return f'ptr_{array_name} += stride_{array_name}_{innermost_dim};'

    # Recursive traversal
    if isinstance(node, SequenceNode):
        new_ops = []
        for op in node.operations:
            new_ops.append(pointer_arithmetic_hoisting_pass(op, loop_vars))
        new_seq = SequenceNode()
        new_seq.operations = new_ops
        new_seq.temp_declarations = deepcopy(node.temp_declarations)
        new_seq.result_temp = node.result_temp
        return new_seq

    elif isinstance(node, LoopNode):
        # Add this loop variable to the stack
        new_loop_vars = loop_vars + [node.loop_variable]
        # Recursively process the body
        new_body = pointer_arithmetic_hoisting_pass(node.body, new_loop_vars ) if node.body else None

        # Check if this is an innermost loop (body is a CalcNode or SequenceNode of only CalcNodes)
        is_innermost = False
        if isinstance(new_body, CalcNode):
            is_innermost = True
        elif isinstance(new_body, SequenceNode):
            is_innermost = all(isinstance(op, CalcNode) for op in new_body.operations)

        if is_innermost:
            # Gather all arrays in all CalcNodes in the body
            if isinstance(new_body, CalcNode):
                calc_nodes = [new_body]
            else:
                calc_nodes = new_body.operations

            all_arrays = []
            for calc in calc_nodes:
                all_arrays += find_array_accesses(calc.code_line)
            # Remove duplicates
            seen = set()
            unique_arrays = []
            for arr in all_arrays:
                if arr[0] not in seen:
                    unique_arrays.append(arr)
                    seen.add(arr[0])

            # Pointer setup code (before the loop)
            pointer_setups = []
            for array_name, indices in unique_arrays:
                pointer_setups.append(CalcNode(pointer_setup_code(array_name, indices, node.loop_variable)))

            # Transform CalcNodes: replace array accesses with *ptr_<array>, add pointer bump after each
            new_calc_nodes = []
            for calc in calc_nodes:
                arrays_in_this = find_array_accesses(calc.code_line)
                # Replace all accesses in this line
                new_code = replace_with_pointer(calc.code_line, arrays_in_this)
                new_calc_nodes.append(CalcNode(new_code))
                # Add pointer bump for each array used in this line
                for array_name, indices in arrays_in_this:
                    # Find which index is the innermost (should be the last index)
                    if node.loop_variable in indices:
                        innermost_dim = indices.index(node.loop_variable)
                    else:
                        # Fallback: bump the last dimension
                        innermost_dim = len(indices) - 1
                    new_calc_nodes.append(CalcNode(pointer_bump_code(array_name, innermost_dim)))

            # Wrap new_calc_nodes in a SequenceNode if more than one
            if len(new_calc_nodes) == 1:
                new_innermost_body = new_calc_nodes[0]
            else:
                seq = SequenceNode()
                seq.operations = new_calc_nodes
                new_innermost_body = seq

            # The new loop body is the transformed innermost body
            new_loop = LoopNode(
                loop_variable=node.loop_variable,
                shape_var=deepcopy(node.shape_var),
                body=new_innermost_body,
                start=node.start,
                increment=node.increment,
                end_offset=node.end_offset
            )

            # If pointer_setups is not empty, wrap pointer setups + loop in a SequenceNode
            if pointer_setups:
                seq = SequenceNode()
                seq.operations = pointer_setups + [new_loop]
                return seq
            else:
                return new_loop

        else:
            # Not innermost: just rebuild the loop with the transformed body
            return LoopNode(
                loop_variable=node.loop_variable,
                shape_var=deepcopy(node.shape_var),
                body=new_body,
                start=node.start,
                increment=node.increment,
                end_offset=node.end_offset
            )

    elif isinstance(node, CalcNode):
        # No transformation needed
        return deepcopy(node)

    else:
        # Unknown node type
        return deepcopy(node)


class State:
    nrow:int
    ncol:int
    registers:List[List[Tuple[str,str]]] #value held by two registers - assumed two for now
    operations: List[List[str]]

    def __init__(self, nrow, ncol):
        self.nrow = nrow
        self.ncol = ncol
        self.operations = [ ["NOP"]* self.ncol] * self.nrow
        self.registers = [(None, None) * self.ncol ] * self.nrow

class Mapping:
    timestep: int
    mapping:Tuple[int, State]


        


def main():
    """Main function for command line usage"""
    if len(sys.argv) != 2:
        print("Usage: python ir_parser.py <ir_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        ir_root = parse_ir_file(filename)
        print("=== Parsed IR Structure ===")
        print(pretty_print_ir(ir_root))
        
        print("\n=== Analysis ===")
        print(f"Root node type: {type(ir_root).__name__}")
        
        if isinstance(ir_root, SequenceNode):
            print(f"Number of operations: {len(ir_root.operations)}")
            print(f"Number of temp declarations: {len(ir_root.temp_declarations)}")
            print(f"Result temp: {ir_root.result_temp}")
        
        root = pointer_arithmetic_hoisting_pass(ir_root)
        print("\n=== After code hoisting ===")

        print(pretty_print_ir(root))
        if isinstance(root, SequenceNode):
            print(f"Number of operations: {len(ir_root.operations)}")
            print(f"Number of temp declarations: {len(ir_root.temp_declarations)}")
            print(f"Result temp: {ir_root.result_temp}")


    except IRParseError as e:
        print(f"Parse error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()