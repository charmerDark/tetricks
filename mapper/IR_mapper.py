"""
This program is a mapper that consumes IR representations from the tetricks backend and provdes a mapping after applying the required code hoisting for pointer arithmetic. 
"""
import re
import sys
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
class CalcNode(IRNode):
    """Represents a calculation node with a single code line"""
    code_line: str
    
    def __str__(self):
        return f'CalcNode("{self.code_line}")'
    



@dataclass 
class LoopNode(IRNode):
    """Represents a loop node with variable, bounds, and body"""
    loop_variable: str
    bounds: str
    body: Optional[IRNode] = None
    
    def __str__(self):
        return f'LoopNode(var="{self.loop_variable}", bounds="{self.bounds}")'


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
    """Custom exception for parsing errors"""
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
        """Parse a LoopNode"""
        # Consume the opening line
        self._advance_line()
        
        loop_var = ""
        bounds = ""
        body = None
        
        while self.current_line < len(self.lines):
            self._skip_empty_and_comments()
            
            if self.current_line >= len(self.lines):
                break
                
            line = self._current_line_content()
            
            # Check for closing brace
            if line == "}":
                self._advance_line()
                break
            
            # Parse variable
            if line.startswith("variable:"):
                loop_var = self._extract_quoted_value(line, "variable:")
                self._advance_line()
            
            # Parse bounds
            elif line.startswith("bounds:"):
                bounds = line.replace("bounds:", "").strip()
                self._advance_line()
            
            # Parse body
            elif line.startswith("body:"):
                self._advance_line()
                self._skip_empty_and_comments()
                if self.current_line < len(self.lines):
                    body = self._parse_node()
            
            # Parse direct child nodes
            elif any(line.startswith(node_type) for node_type in ["SequenceNode {", "LoopNode {", "CalcNode {"]):
                body = self._parse_node()
            
            else:
                self._advance_line()
        
        return LoopNode(loop_variable=loop_var, bounds=bounds, body=body)
    
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
    """Pretty print the parsed IR structure"""
    indent_str = "  " * indent
    
    if isinstance(node, SequenceNode):
        result = f"{indent_str}SequenceNode {{\n"
        
        # Print temp declarations
        if node.temp_declarations:
            result += f"{indent_str}  // Temporary declarations:\n"
            for name, decl in node.temp_declarations.items():
                result += f"{indent_str}  {decl}\n"
            result += "\n"
        
        # Print operations
        for i, op in enumerate(node.operations):
            result += pretty_print_ir(op, indent + 1)
            if i < len(node.operations) - 1:
                result += "\n"
        
        # Print result temp
        if node.result_temp:
            result += f"\n{indent_str}  // Result: {node.result_temp}\n"
        
        result += f"{indent_str}}}\n"
        
    elif isinstance(node, LoopNode):
        result = f"{indent_str}LoopNode {{\n"
        result += f"{indent_str}  variable: '{node.loop_variable}'\n"
        result += f"{indent_str}  bounds: {node.bounds}\n"
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

def pointer_arithmetic_hoisting_pass(node: IRNode)-> IRNode:
    """
    Code Hoisting and Pointer Bumping Pass for Tetricks IR
    ------------------------------------------------------

    This pass optimises loop nests by hoisting invariant address computations
    and converting array accesses in the innermost loops to pointer-based traversal.

    Principles:
    - For each array access in a loop nest, identify which indices are constant
    (i.e., do not depend on the innermost loop variable).
    - Hoist the computation of the base pointer for such accesses outside the innermost loop.
    - In the innermost loop, replace multi-dimensional array accesses with pointer dereferences,
    and increment the pointer by the appropriate stride (pointer bumping).
    - Output array references (e.g., temp1[i][l]) are also hoisted as far out as possible.

    Benefits:
    - Reduces redundant address calculations in inner loops.
    - Enables efficient hardware mapping and vectorisation.
    - Improves memory access patterns and throughput.

    Example:

    Original IR:
    -------------
    LoopNode {
    variable: 'i'
    body:
        LoopNode {
        variable: 'j'
        body:
            LoopNode {
            variable: 'l'
            body:
                LoopNode {
                variable: 'k'
                body:
                    CalcNode { "temp1[i][l] += A[i][j][k][l];" }
                }
            }
        }
    }

    After Hoisting:
    ---------------
    LoopNode {
    variable: 'i'
    body:
        LoopNode {
        variable: 'j'
        body:
            LoopNode {
            variable: 'l'
            body:
                SequenceNode {
                operations: [
                    CalcNode { "pts_A = &A[i][j][0][l];" },
                    CalcNode { "pts_temp1 = &temp1[i][l];" },
                    LoopNode {
                    variable: 'k'
                    body:
                        SequenceNode {
                        operations: [
                            CalcNode { "*pts_temp1 += *pts_A;" },
                            CalcNode { "pts_A += stride_A_2;" }
                        ]
                        }
                    }
                ]
                }
            }
        }
    }

    Here, the base pointers for A and temp1 are computed once per (i, j, l),
    and the innermost loop over k uses pointer dereference and pointer bumping.

    This pass preserves the IR structure: each LoopNode has a single body,
    and multiple operations are wrapped in a SequenceNode as needed.
    """
    pass



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
        
    except IRParseError as e:
        print(f"Parse error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()