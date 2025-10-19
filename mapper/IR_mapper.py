"""
This program is a mapper that consumes IR representations from the tetricks backend and provdes a mapping after applying the required code hoisting for pointer arithmetic. 
"""

import re
import sys
from copy import deepcopy
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

import argparse

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
        return (
            f'LoopNode(var="{self.loop_variable}", shape_var={self.shape_var}, '
            f"start={self.start}, increment={self.increment}, end_offset={self.end_offset})"
        )


@dataclass
class CalcNode(IRNode):
    """Represents a calculation node with a single code line"""

    code_line: str
    tag: Optional[str] = field(default=None, init=False)

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
        return f"SequenceNode({len(self.operations)} operations, {len(self.temp_declarations)} temps)"


class IRParseError(Exception):
    pass


class IRParser:
    """Parser for IR text format"""

    def __init__(self, text: str):
        self.lines = text.strip().split("\n")
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
            raise IRParseError(
                f"Unknown node type at line {self.current_line + 1}: {line}"
            )

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
            elif any(
                line.startswith(node_type)
                for node_type in ["SequenceNode {", "LoopNode {", "CalcNode {"]
            ):
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
            elif any(
                line.startswith(node_type)
                for node_type in ["SequenceNode {", "LoopNode {", "CalcNode {"]
            ):
                body = self._parse_node()
            else:
                self._advance_line()

        if shape_var is None:
            raise IRParseError("LoopNode missing shape_var")
        return LoopNode(
            loop_variable=loop_var,
            shape_var=shape_var,
            body=body,
            start=start,
            increment=increment,
            end_offset=end_offset,
        )

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
            raise IRParseError(
                f"Invalid CalcNode format at line {self.current_line + 1}: {line}"
            )

        self._advance_line()
        return CalcNode(code_line=code)

    def _is_temp_declaration(self, line: str) -> bool:
        """Check if line is a temp variable declaration
        Currently ased off the fact that backend names all temps as tempX. slightly hacky
        """
        # Look for patterns like "double tempX[...];"
        return line.startswith("double ") and ("temp" in line or line.endswith(";"))

    def _parse_temp_declaration(self, line: str) -> Tuple[str, str]:
        """Parse temp declaration and extract name and full declaration"""
        # Extract variable name (everything between "double " and first "[" or "=")
        match = re.search(r"double\s+(\w+)", line)
        if match:
            name = match.group(1)
            return name, line
        else:
            raise IRParseError(f"Invalid temp declaration: {line}")

    def _extract_result_temp(self, line: str) -> str:
        """Extract result temp name from comment"""
        # Pattern: "// Result stored in: tempX"
        match = re.search(r"//\s*Result stored in:\s*(\w+)", line)
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
        tag_str = f" [tag: {node.tag}]" if getattr(node, 'tag', None) else ""
        result = f'{indent_str}CalcNode {{ "{node.code_line} "{tag_str} }}\n'

    else:
        result = f"{indent_str}Unknown node type: {type(node)}\n"

    return result


def parse_ir_file(filename: str) -> IRNode:
    """Parse IR from a file"""
    try:
        with open(filename, "r") as f:
            content = f.read()

        parser = IRParser(content)
        return parser.parse()

    except FileNotFoundError:
        raise IRParseError(f"File not found: {filename}")
    except Exception as e:
        raise IRParseError(f"Error parsing file {filename}: {str(e)}")


def extract_array_shapes_from_decls(temp_declarations):
    """
    Given a dict of temp declarations, returns a dict mapping array names to shape lists.
    Example: {'temp1': 'double temp1[D0][D3];'} -> {'temp1': ['D0', 'D3']}
    """
    array_shapes = {}
    for name, decl in temp_declarations.items():
        # Match e.g. double temp1[D0][D3];
        m = re.match(
            r".*?(\w+)\s*\[([^\]]+)\](?:\s*\[([^\]]+)\])?(?:\s*\[([^\]]+)\])?(?:\s*\[([^\]]+)\])?",
            decl,
        )
        if m:
            arr_name = m.group(1)
            dims = [g for g in m.groups()[1:] if g is not None]
            array_shapes[arr_name] = dims
    return array_shapes

def extract_tensor_shapes_from_loops(node: IRNode) -> Dict[str, List[str]]:
    """
    Traverse the IR and collect shape information from loop bounds.
    Returns dict mapping tensor_name -> list of dimension expressions
    """
    tensor_shapes = {}
    
    def collect_shapes(n):
        if isinstance(n, SequenceNode):
            for op in n.operations:
                collect_shapes(op)
        
        elif isinstance(n, LoopNode):
            # Extract tensor name and dimension from shape_var
            tensor_name = n.shape_var.tensor_name
            idx_posn = n.shape_var.idx_posn
            shape_expr = str(n.shape_var)  # e.g., "A.shape[0]"
            
            if tensor_name not in tensor_shapes:
                tensor_shapes[tensor_name] = {}
            
            tensor_shapes[tensor_name][idx_posn] = shape_expr
            
            if n.body:
                collect_shapes(n.body)
        
        elif isinstance(n, CalcNode):
            pass  # No shape info in calc nodes
    
    collect_shapes(node)
    
    # Convert dict of dicts to dict of lists
    result = {}
    for tensor_name, dims_dict in tensor_shapes.items():
        if dims_dict:
            max_dim = max(dims_dict.keys())
            shape_list = [dims_dict.get(i, '?') for i in range(max_dim + 1)]
            result[tensor_name] = shape_list
    
    return result


def compute_stride_symbolic(array_shape: List[str], innermost_dim: int) -> str:
    """
    Compute stride for the innermost dimension.
    For dimension d, stride = product of all dimensions after d.
    Returns a symbolic expression as a string.
    """
    if innermost_dim >= len(array_shape) - 1:
        # Last dimension has stride 1
        return "1"
    
    # Collect dimensions after innermost_dim
    dims_after = array_shape[innermost_dim + 1:]
    
    if len(dims_after) == 1:
        return dims_after[0]
    else:
        # Create multiplication expression: D2 * D3 * D4
        return " * ".join(dims_after)

import re
from copy import deepcopy
from typing import List, Dict, Tuple, Optional

def find_innermost_loop(loop_node: 'LoopNode') -> Tuple['LoopNode', List['LoopNode']]:
    """
    Find the innermost loop and return it along with the path from root to innermost.
    Returns: (innermost_loop, path_list)
    """
    path = [loop_node]
    current = loop_node
    
    while current.body and isinstance(current.body, LoopNode):
        current = current.body
        path.append(current)
    
    return current, path


def parse_calc_statement(code_line: str) -> Dict:
    """
    Parse a calculation statement and extract components.
    Example: "temp1[i][l] += A[i][j][k][l];" 
    Returns: {
        'lhs_array': 'temp1',
        'lhs_indices': ['i', 'l'],
        'rhs_arrays': [('A', ['i', 'j', 'k', 'l'])],
        'operation': '+='
    }
    """
    # Remove semicolon and whitespace
    stmt = code_line.strip().rstrip(';').strip()
    
    # Detect operation type
    if '+=' in stmt:
        operation = '+='
        lhs, rhs = stmt.split('+=')
    elif '=' in stmt:
        operation = '='
        lhs, rhs = stmt.split('=', 1)
    else:
        raise ValueError(f"Cannot parse statement: {code_line}")
    
    lhs = lhs.strip()
    rhs = rhs.strip()
    
    # Parse LHS
    lhs_array, lhs_indices = get_array_indices_from_expr(lhs)
    
    # Parse RHS - find all array accesses
    rhs_arrays = []
    # Pattern to match array access: word followed by bracketed indices
    pattern = r'(\w+)(\[[^\]]+\])+' 
    for match in re.finditer(pattern, rhs):
        array_expr = match.group(0)
        array_name, indices = get_array_indices_from_expr(array_expr)
        rhs_arrays.append((array_name, indices))
    
    return {
        'lhs_array': lhs_array,
        'lhs_indices': lhs_indices,
        'rhs_arrays': rhs_arrays,
        'operation': operation,
        'full_rhs': rhs
    }


def get_array_indices_from_expr(expr: str) -> Tuple[str, List[str]]:
    """
    Extract array name and indices from expression like 'A[i][j][k]'
    Returns: ('A', ['i', 'j', 'k'])
    """
    # Match array name
    match = re.match(r'(\w+)', expr)
    if not match:
        raise ValueError(f"Cannot extract array name from: {expr}")
    
    array_name = match.group(1)
    
    # Extract all indices
    indices = re.findall(r'\[([^\]]+)\]', expr)
    
    return array_name, indices


def determine_hoist_level(indices: List[str], loop_path: List['LoopNode']) -> int:
    """
    Determine which loop level to hoist a pointer declaration to.
    Returns the index in loop_path where all required indices FIRST become available.
    We want the shallowest (outermost) level where all indices exist.
    """
    # Find the shallowest (outermost) loop that contains all required indices
    for i in range(len(loop_path)):
        # Get all loop variables up to and including this level
        available_vars = {loop.loop_variable for loop in loop_path[:i+1]}
        
        # Check if all indices are available at this level
        if all(idx in available_vars for idx in indices):
            return i
    
    # If no match, shouldn't happen but return outermost as fallback
    return 0


def build_pointer_init(array_name: str, indices: List[str], innermost_var: str, 
                       loop_path: List['LoopNode']) -> str:
    """
    Build pointer initialization statement.
    For arrays dependent on innermost loop var, replace innermost index with 0.
    """
    init_indices = []
    for idx in indices:
        if idx == innermost_var:
            init_indices.append('0')
        else:
            init_indices.append(idx)
    
    indices_str = ''.join(f'[{idx}]' for idx in init_indices)
    return f"double *ptr_{array_name} = &{array_name}{indices_str};"


def build_stride_decl(array_name: str, array_shape: List[str], 
                      innermost_dim_index: int) -> str:
    """
    Build stride declaration for incrementing pointer.
    Stride = product of all dimensions after innermost_dim_index
    """
    stride_expr = compute_stride_symbolic(array_shape, innermost_dim_index)
    return f"const int stride_{array_name}_{innermost_dim_index} = {stride_expr};"


def compute_stride_symbolic(array_shape: List[str], innermost_dim: int) -> str:
    """
    Compute stride for the innermost dimension.
    For dimension d, stride = product of all dimensions after d.
    Returns a symbolic expression as a string.
    """
    if innermost_dim >= len(array_shape) - 1:
        # Last dimension has stride 1
        return "1"
    
    # Collect dimensions after innermost_dim
    dims_after = array_shape[innermost_dim + 1:]
    
    if len(dims_after) == 1:
        return dims_after[0]
    else:
        # Create multiplication expression: D2 * D3 * D4
        return " * ".join(dims_after)


def transform_loop_nest(loop_node: 'LoopNode', array_shapes: Dict[str, List[str]]) -> 'LoopNode':
    """
    Transform a single loop nest by hoisting pointer arithmetic.
    """
    # Step 1: Find innermost loop
    innermost_loop, loop_path = find_innermost_loop(loop_node)
    innermost_var = innermost_loop.loop_variable
    
    # Step 2: Extract calculations from innermost loop body
    calc_nodes = []
    if isinstance(innermost_loop.body, CalcNode):
        calc_nodes = [innermost_loop.body]
    elif isinstance(innermost_loop.body, SequenceNode):
        calc_nodes = [op for op in innermost_loop.body.operations if isinstance(op, CalcNode)]
    else:
        # No calculations to transform
        return deepcopy(loop_node)
    
    # Only transform if we have calculations
    if not calc_nodes:
        return deepcopy(loop_node)
    
    # Process the first calc node (assuming single operation per innermost loop)
    calc_node = calc_nodes[0]
    parsed = parse_calc_statement(calc_node.code_line)

    # Step 3: Determine what needs to be hoisted
    lhs_array = parsed['lhs_array']
    lhs_indices = parsed['lhs_indices']
    rhs_arrays = parsed['rhs_arrays']
    
    # Track declarations to insert at each level
    # level_decls[i] = list of declaration strings for loop at index i
    level_decls = [[] for _ in loop_path]
    
    # Arrays that need pointer updates (depend on innermost loop var)
    arrays_to_update = []
    
    # Step 4: Process LHS array
    lhs_ptr_init = build_pointer_init(lhs_array, lhs_indices, innermost_var, loop_path)

    # Check if LHS depends on innermost var (BEFORE any transformation)
    lhs_depends_on_innermost = innermost_var in lhs_indices

    if lhs_depends_on_innermost:
        # Both pointer and stride go one level before innermost
        level_decls[len(loop_path) - 2].append(lhs_ptr_init)
        
        dim_index = lhs_indices.index(innermost_var)
        if lhs_array in array_shapes:
            stride_decl = build_stride_decl(lhs_array, array_shapes[lhs_array], dim_index)
            level_decls[len(loop_path) - 2].append(stride_decl)
            arrays_to_update.append((lhs_array, dim_index))
    else:
        # Pointer doesn't depend on innermost var, hoist normally
        lhs_hoist_level = determine_hoist_level(lhs_indices, loop_path)
        level_decls[lhs_hoist_level].append(lhs_ptr_init)
    
# Step 5: Process RHS arrays
    for rhs_array, rhs_indices in rhs_arrays:
        rhs_ptr_init = build_pointer_init(rhs_array, rhs_indices, innermost_var, loop_path)
        
        # Check if this array depends on innermost var (BEFORE any transformation)
        rhs_depends_on_innermost = innermost_var in rhs_indices
        
        if rhs_depends_on_innermost:
            # Both pointer and stride go one level before innermost
            level_decls[len(loop_path) - 2].append(rhs_ptr_init)
            
            dim_index = rhs_indices.index(innermost_var)
            if rhs_array in array_shapes:
                stride_decl = build_stride_decl(rhs_array, array_shapes[rhs_array], dim_index)
                level_decls[len(loop_path) - 2].append(stride_decl)
                arrays_to_update.append((rhs_array, dim_index))
        else:
            # Pointer doesn't depend on innermost var, hoist normally
            rhs_hoist_level = determine_hoist_level(rhs_indices, loop_path)
            level_decls[rhs_hoist_level].append(rhs_ptr_init)
    
    # Step 6: Transform the innermost calculation
    new_calc_operations = []
    
    # Replace array accesses with pointer dereferences
    transformed_calc = calc_node.code_line
    
    # Replace LHS
    lhs_pattern = f"{lhs_array}" + r'\[([^\]]+)\]' * len(lhs_indices)
    transformed_calc = re.sub(lhs_pattern, f"*ptr_{lhs_array}", transformed_calc)
    
    # Replace RHS arrays
    for rhs_array, rhs_indices in rhs_arrays:
        rhs_pattern = f"{rhs_array}" + r'\[([^\]]+)\]' * len(rhs_indices)
        transformed_calc = re.sub(rhs_pattern, f"*ptr_{rhs_array}", transformed_calc)
    
    # Create new calc node with tag
    new_calc = CalcNode(code_line=transformed_calc)
    if parsed['operation'] == '+=':
        new_calc.tag = 'einsum'
    else:
        new_calc.tag = 'elementwise'
    
    new_calc_operations.append(new_calc)
    
    # Add pointer update statements
    for array_name, dim_index in arrays_to_update:
        update_stmt = f"ptr_{array_name} += stride_{array_name}_{dim_index};"
        update_node = CalcNode(code_line=update_stmt)
        update_node.tag = 'update'
        new_calc_operations.append(update_node)
    
    # Wrap in SequenceNode
    new_innermost_body = SequenceNode()
    new_innermost_body.operations = new_calc_operations
    
    # Step 7: Rebuild loop nest from inside out
    # Start with the transformed innermost body
    current_body = new_innermost_body
    
    # Work backwards through the loop path
    for i in range(len(loop_path) - 1, -1, -1):
        loop = loop_path[i]
        
        # Check if we need to add declarations at this level
        # Declarations at level i should go INSIDE loop i, BEFORE the next inner content
        if level_decls[i]:
            # Wrap the current_body with declarations
            wrapper = SequenceNode()
            for decl in level_decls[i]:
                wrapper.operations.append(CalcNode(code_line=decl))
            wrapper.operations.append(current_body)
            current_body = wrapper
        
        # Create the loop at this level with current_body as its body
        new_loop = LoopNode(
            loop_variable=loop.loop_variable,
            shape_var=deepcopy(loop.shape_var),
            start=loop.start,
            increment=loop.increment,
            end_offset=loop.end_offset,
            body=current_body
        )
        
        current_body = new_loop

    return current_body


def pointer_arithmetic_hoisting_pass(node: 'IRNode', array_shapes: Optional[Dict[str, List[str]]] = None) -> 'IRNode':
    """
    Main function to apply pointer arithmetic hoisting to entire IR tree.
    
    Args:
        node: Root IR node
        array_shapes: Dictionary mapping array names to their shapes (list of dimension expressions)
    
    Returns:
        Transformed IR tree with hoisted pointer arithmetic
    """
    if array_shapes is None:
        array_shapes = {}
    
    if isinstance(node, SequenceNode):
        # Create new SequenceNode
        new_seq = SequenceNode()
        new_seq.temp_declarations = deepcopy(node.temp_declarations)
        new_seq.result_temp = node.result_temp
        
        # Extract array shapes from temp declarations if not provided
        if not array_shapes:
            from __main__ import extract_array_shapes_from_decls, extract_tensor_shapes_from_loops
            temp_shapes = extract_array_shapes_from_decls(node.temp_declarations)
            tensor_shapes = extract_tensor_shapes_from_loops(node)
            array_shapes = {**temp_shapes, **tensor_shapes}
        
        # Process each operation
        for op in node.operations:
            if isinstance(op, LoopNode):
                # Transform this loop nest
                transformed = transform_loop_nest(op, array_shapes)
                new_seq.operations.append(transformed)
            else:
                # Recursively process
                new_seq.operations.append(pointer_arithmetic_hoisting_pass(op, array_shapes))
        
        return new_seq
    
    elif isinstance(node, LoopNode):
        # Entry point for a standalone loop nest
        return transform_loop_nest(node, array_shapes)
    
    elif isinstance(node, CalcNode):
        # Leaf node - just copy
        return deepcopy(node)
    
    else:
        # Unknown node type - copy as is
        return deepcopy(node)

class State:
    nrow: int
    ncol: int
    registers: List[
        List[Tuple[str, str]]
    ]  # value held by two registers - assumed two for now
    operations: List[List[str]]

    def __init__(self, nrow, ncol):
        self.nrow = nrow
        self.ncol = ncol
        self.operations = [["NOP"] * self.ncol] * self.nrow
        self.registers = [(None, None) * self.ncol] * self.nrow


class Mapping:
    timestep: int
    mapping: Tuple[int, State]

    def pretty_print(self):
        if self.mapping:
            for timestep, ops in self.mapping:
                print(f'Timestep {timestep} : {ops}')


def mac_load_routine(
    sum_term: str, mult_terms: List[str], nrow: int, ncol: int
) -> Mapping:
    pass


def mac_routine(sum_term: str, mult_terms: List[str], nrow: int, ncol: int) -> Mapping:
    pass


def elementwise_routine(terms: List[str], op: str) -> Mapping:
    pass


def mapper(nrow: int, ncol: int, root: IRNode) -> Mapping:
    """
    Function takes in root node and traverses it till innermost loops are found.
    Innermost loops must be accelerated on nrow x ncol CGRAs
    schedules operations by checking tag field on CalcNodes.
    Implements mapping by calling routines from schedule as needed.

    """
    ops_q = []  # queue to which operations will be added as needed
    mapping = Mapping(nrow, ncol)

    def traverse(node:IRNode, mapping):

        if isinstance(node, LoopNode):

            if  isinstance(node.body, SequenceNode) and all(isinstance(op,CalcNode) for op in node.body.operations) :
                ## in innermost loop
                for op in node.body.operations:
                    if not op.tag:
                        print("Reached innermost loop operation not tagged. please check IR")
                    elif  op.tag == "einsum":
                        mapping = mac_load_routine(mapping)
                    elif op.tag == "elementwise":
                        mapping = elementwise_routine(mapping)
                    elif op.tag == "update":
                        update_instruction_count+=1
                        if update_instruction_count == mapping.nrow:
                            pass

def main():

    parser = argparse.ArgumentParser(description="Tetricks IR Mapper")
    parser.add_argument("ir_file", help="Input IR file")
    parser.add_argument("--nrow", type=int, default=3, help="Number of CGRA rows (default: 3)")
    parser.add_argument("--ncol", type=int, default=3, help="Number of CGRA columns (default: 3)")
    args = parser.parse_args()

    filename = args.ir_file
    nrow = args.nrow
    ncol = args.ncol

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
            array_shapes = extract_array_shapes_from_decls(ir_root.temp_declarations)
        else:
            array_shapes = {}

        # Only need to pass temp array shapes from declarations
        array_shapes = extract_array_shapes_from_decls(ir_root.temp_declarations)

        # The function now handles everything internally
        root = pointer_arithmetic_hoisting_pass(ir_root)
        print("\n=== After code hoisting ===")
        print(pretty_print_ir(root))

        mapping = mapper(nrow, ncol, root)


    except IRParseError as e:
        print(f"Parse error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
