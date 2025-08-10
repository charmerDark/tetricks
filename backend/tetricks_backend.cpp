/*
Code generating C(pp) compiler. Reads AST from JSON format, lowers to loop IR, optimises and generates code.
*/
#include <iostream>
#include <memory>
#include <set>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

#include <fstream>
#include "json.hpp"
using json = nlohmann::json;


enum BinOpKind { Add, Subtract, Multiply, Divide };


//for reading from json
BinOpKind parseBinOpKind(const string &opStr) {
  if (opStr == "Add") return Add;
  if (opStr == "Subtract") return Subtract;
  if (opStr == "Multiply") return Multiply;
  if (opStr == "Divide") return Divide;
  throw runtime_error("Unknown binary operation: " + opStr);
}


struct expr {
  /*
  AST nodes representation
  */
  virtual ~expr() = default;
};

struct Tensor : expr {
  string name;
  int num_dims = -1;
  bool dims_inferred = false;
  
  // Track which tensor/dimension determines each of our dimensions - mainly for temp variables generated in the process
  struct DimensionSource {
    shared_ptr<Tensor> source_tensor;
    int source_dim_index;
    char loop_variable; 
    DimensionSource() = default;
    DimensionSource(shared_ptr<Tensor> tensor, int dim, char var)
        : source_tensor(tensor), source_dim_index(dim), loop_variable(var) {}
};

  
  vector<DimensionSource> dim_sources;  // One per dimension
  
  Tensor(const string &n) : name(n) {}
  Tensor(const string &n, int dims, bool inferred = false): name(n), num_dims(dims), dims_inferred(inferred) {
      dim_sources.resize(dims);  // Initialize empty sources
  }
  
  // Helper to set dimension source for intermediary temp tensors
  void setDimensionSource(int our_dim, shared_ptr<Tensor> source_tensor, int source_dim, char loop_var) {
    if (our_dim >= 0 && our_dim < dim_sources.size()) {
        dim_sources[our_dim] = DimensionSource(source_tensor, source_dim, loop_var);
    }
}

  
  // Helper to get dimension reference for code generation
  string getDimensionRef(int dim_index) const {
      if (dim_index >= 0 && dim_index < dim_sources.size()) {
          const auto& source = dim_sources[dim_index];
          if (source.source_tensor) {
              return source.source_tensor->name + ".shape[" + to_string(source.source_dim_index) + "]";
          }
      }
      // Fallback for original tensors or unknown sources
      return name + ".shape[" + to_string(dim_index) + "]";
  }
};

struct EinsumNode : expr {
  /*
  indexes;
  */
  vector<string> indexes;
  vector<shared_ptr<Tensor>> inputs; 

  EinsumNode(const vector<string> &idx, const vector<shared_ptr<Tensor>> &inp_tensors) {
          if (idx.size() == inp_tensors.size() + 1) {
              indexes = idx;
              inputs = inp_tensors;
          } else {
              throw std::invalid_argument(
                  "Input indices don't match the number of inputs provided.");
          }
      }
};

//to help with parsing notation from ast where the notation is
// just stored as string to EinsumNode where it is stored as 
//vector of stings for each tensor
vector<string> splitEinsumNotation(const string &notation) {
  size_t arrowPos = notation.find("->");
  if (arrowPos == string::npos) {
    throw runtime_error("Einsum notation must contain '->'");
  }

  string inputPart = notation.substr(0, arrowPos);
  string outputPart = notation.substr(arrowPos + 2);

  vector<string> result;
  stringstream ss(inputPart);
  string token;
  while (getline(ss, token, ',')) {
    result.push_back(token);
  }

  result.push_back(outputPart);
  return result;
}



struct BinOpNode : expr {
  BinOpKind opn;
  shared_ptr<expr> inp1;
  shared_ptr<expr> inp2;
  BinOpNode(BinOpKind operation, shared_ptr<expr> input1, shared_ptr<expr> input2): opn(operation), inp1(input1), inp2(input2) {}
};


class DimensionInferenceEngine {
  private:
      // Helper to find tensor by name in the AST
      std::vector<std::shared_ptr<Tensor>> findAllTensors(std::shared_ptr<expr> ast) {
          std::vector<std::shared_ptr<Tensor>> tensors;
          collectTensors(ast, tensors);
          return tensors;
      }
      
      void collectTensors(std::shared_ptr<expr> ast, std::vector<std::shared_ptr<Tensor>>& tensors) {
          if (auto tensor = std::dynamic_pointer_cast<Tensor>(ast)) {
              tensors.push_back(tensor);
          }
          
          if (auto binop = std::dynamic_pointer_cast<BinOpNode>(ast)) {
              collectTensors(binop->inp1, tensors);
              collectTensors(binop->inp2, tensors);
          }
          
          if (auto einsum = std::dynamic_pointer_cast<EinsumNode>(ast)) {
              // Note: EinsumNode stores Tensor objects by value, not shared_ptr
              // We'll need to modify this - see below
          }
      }
      
  public:      
      // Infer dimensions from einsum notation
      void inferFromEinsum(const EinsumNode* einsum) {
          // For each input tensor, count unique characters in its index pattern
          for (size_t i = 0; i < einsum->inputs.size(); i++) {
              const std::string& indexPattern = einsum->indexes[i];
              auto tensor = einsum->inputs[i];
              
              // Count unique characters = number of dimensions
              std::set<char> uniqueIndices;
              for (char c : indexPattern) {
                  uniqueIndices.insert(c);
              }
              
              int dims = uniqueIndices.size();
              if (tensor->num_dims != -1 && tensor->num_dims != dims) {
                  throw std::runtime_error("Dimension conflict for tensor " + tensor->name);
              }
        tensor->num_dims = dims;
        tensor->dims_inferred = true;
          }
      }
      
      // Propagate dimensions through binary operations
      void inferFromBinOp(const BinOpNode* binop) {
          // For element-wise operations, both operands must have same dimensions
          
          // Get dimension info for both operands
          auto leftDims = getExpressionDims(binop->inp1);
          auto rightDims = getExpressionDims(binop->inp2);
          
          // Determine the common dimensionality
          int commonDims = -1;
          if (leftDims != -1 && rightDims != -1) {
              // Both known - they must match
              if (leftDims != rightDims) {
                  throw std::runtime_error("Dimension mismatch in binary operation: " + std::to_string(leftDims) + " vs " + std::to_string(rightDims));
              }
              commonDims = leftDims;
          } else if (leftDims != -1) {
              // Left known, propagate to right
              commonDims = leftDims;
          } else if (rightDims != -1) {
              // Right known, propagate to left
              commonDims = rightDims;
          }
          // If both unknown, we can't infer anything - should not get o this case as frontend parser should throw error for two tensor binops
          
          if (commonDims != -1) {
              // Propagate to any tensor operands
              propagateToTensors(binop->inp1, commonDims);
              propagateToTensors(binop->inp2, commonDims);
          }
      }
      
      // Get dimensions of an expression (could be tensor, einsum result, etc.)
      int getExpressionDims(std::shared_ptr<expr> expression) {
          if (auto tensor = std::dynamic_pointer_cast<Tensor>(expression)) {
              return tensor->num_dims;
          }
          
          if (auto einsum = std::dynamic_pointer_cast<EinsumNode>(expression)) {
              // Output dimensions = unique chars in output pattern
              const std::string& outputPattern = einsum->indexes.back();
              std::set<char> uniqueIndices;
              for (char c : outputPattern) {
                  uniqueIndices.insert(c);
              }
              return uniqueIndices.size();
          }
          
          if (auto binop = std::dynamic_pointer_cast<BinOpNode>(expression)) {
              // Binary op result has same dims as operands
              int leftDims = getExpressionDims(binop->inp1);
              int rightDims = getExpressionDims(binop->inp2);
              
              if (leftDims != -1) return leftDims;
              if (rightDims != -1) return rightDims;
          }
          
          return -1; // Unknown
      }
      
      // Propagate dimension info to tensor nodes in an expression
      void propagateToTensors( std::shared_ptr<expr> expression, int dims) {
          if (auto tensor = std::dynamic_pointer_cast<Tensor>(expression)) {
            if (tensor->num_dims != -1 && tensor->num_dims != dims) {
              throw std::runtime_error("Dimension conflict for tensor " + tensor->name + 
                                     ": previously " + std::to_string(tensor->num_dims) + 
                                     ", now " + std::to_string(dims));
            }
          tensor->num_dims = dims;
          tensor->dims_inferred = true;
          }
          // Note: We don't recurse into Einsum or BinOp here because
          // we want to maintain the tree structure for later processing
      }
      
      // Main inference function - call this after parsing AST
      void inferAllDimensions(std::shared_ptr<expr> ast) {
          // Two-pass approach:
          // Pass 1: Extract explicit information from Einsum nodes
          extractExplicitDimensions(ast);
          
          // Pass 2: Propagate through binary operations
          propagateDimensions(ast);
      }
      
      // Pass 1: Extract dimensions from Einsum nodes
      void extractExplicitDimensions(std::shared_ptr<expr> ast) {
          if (auto einsum = std::dynamic_pointer_cast<EinsumNode>(ast)) {
              inferFromEinsum(einsum.get());
          }
          
          if (auto binop = std::dynamic_pointer_cast<BinOpNode>(ast)) {
              extractExplicitDimensions(binop->inp1);
              extractExplicitDimensions( binop->inp2);
          }
          
          // Tensor nodes don't provide explicit dimension info
      }
      
      // Pass 2: Propagate dimensions through binary operations
      void propagateDimensions( std::shared_ptr<expr> ast) {
          if (auto binop = std::dynamic_pointer_cast<BinOpNode>(ast)) {
              // First recurse to children
              propagateDimensions(binop->inp1);
              propagateDimensions(binop->inp2);
              
              // Then infer for this binary operation
              inferFromBinOp(binop.get());
          }
          
          // No propagation needed for Einsum or Tensor nodes
      }
      
      // Utility: Print all inferred dimensions (for debugging)
      void printAllDimensions(std::shared_ptr<expr> ast) const {
          auto tensors = const_cast<DimensionInferenceEngine*>(this)->findAllTensors(ast);
          for (auto tensor : tensors) {
              std::cout << tensor->name << ": " << tensor->num_dims << " dimensions";
              if (tensor->dims_inferred) {
                  std::cout << " (inferred)";
              }
              std::cout << std::endl;
          }
      }
  };
  

struct shape_variable {
  /*
  Helper struct that represents a shape variable used to store the range of each
  loop in Einsum calculation.
  */
  shared_ptr<Tensor> tensor_name;
  int idx_posn;
  shape_variable(const shared_ptr<Tensor> t_name, int pos): tensor_name(t_name), idx_posn(pos) {}
};

struct IRNode {
  /*
  Represents nodes in the IR.
  can be LoopNode or CalcNode
  */
  virtual ~IRNode() = default;
};

struct CalcNode : IRNode {
  /*
  IR node representing calculations
  */
  string code_line;
  CalcNode(const string &line) : code_line(line) {}
};

struct LoopNode : IRNode {
  /*
  Loop IR for loops to implement operations.
  EinsumNode("ij->ji",A) would have LoopNode(i, A[0], LoopNode(j,A[1],
  Calcnode(temp1[j][i] = A[i][j]) ))
  */
  char loop_variable;
  shape_variable shape_var;
  int start = 0;
  int increment = 1;
  int end_offset = 0;
  shared_ptr<IRNode> body;

  LoopNode(char var, const shape_variable &shape, shared_ptr<IRNode> b): loop_variable(var), shape_var(shape), body(b) {}

  //constructor to be used by loop unrolling pass
  LoopNode(char var, const shape_variable &shape, shared_ptr<IRNode> b, int start_val, int increment_val, int end_offset_val): loop_variable(var), shape_var(shape), body(b),start(start_val),increment(increment_val),end_offset(end_offset_val) {}
};

struct SequenceNode : IRNode {
  vector<shared_ptr<IRNode>> operations;
  map<string, string> temp_declarations; // temp_name -> declaration (e.g., "double temp1[10][20];")
  string result_temp; // Which temp variable holds the final result
  
  SequenceNode() = default;
  
  void addOperation(shared_ptr<IRNode> op) {
      operations.push_back(op);
  }
  
  void addTempDeclaration(const string& temp_name, const string& declaration) {
      temp_declarations[temp_name] = declaration;
  }
  
  void setResultTemp(const string& temp) {
      result_temp = temp;
  }
};

// Print IR structure (abstract representation for debugging)
void printIR(shared_ptr<IRNode> node, int indent = 0) {
  if (!node)
    return;

  string indentStr(indent * 2, ' ');

  if (auto seqNode = dynamic_pointer_cast<SequenceNode>(node)) {
    cout << indentStr << "SequenceNode {" << endl;
    
    // Print temp declarations first (matches code generation order)
    if (!seqNode->temp_declarations.empty()) {
      cout << indentStr << "  // Temporary variable declarations" << endl;
      for (const auto& decl : seqNode->temp_declarations) {
        cout << indentStr << "  " << decl.second << endl;
      }
      cout << endl;
    }
    
    // Print operations in sequence
    if (!seqNode->operations.empty()) {
      for (size_t i = 0; i < seqNode->operations.size(); i++) {
        printIR(seqNode->operations[i], indent + 1);
        if (i < seqNode->operations.size() - 1) {
          cout << endl; // Add spacing between operations
        }
      }
    }
    
    // Print result info at the end
    if (!seqNode->result_temp.empty()) {
      cout << endl << indentStr << "  // Result stored in: " << seqNode->result_temp << endl;
    }
    
    cout << indentStr << "}" << endl;
  } else if (auto loopNode = dynamic_pointer_cast<LoopNode>(node)) {
    cout << indentStr << "LoopNode {" << endl;
    cout << indentStr << "  variable: '" << loopNode->loop_variable << "'"
         << endl;
    cout << indentStr << "  bounds: " << loopNode->shape_var.tensor_name->name
         << ".shape[" << loopNode->shape_var.idx_posn << "]" << endl;
    cout << indentStr << "  body:" << endl;
    printIR(loopNode->body, indent + 2);
    cout << indentStr << "}" << endl;
  } else if (auto calcNode = dynamic_pointer_cast<CalcNode>(node)) {
    cout << indentStr << "CalcNode { \"" << calcNode->code_line << "\" }"
         << endl;
  }
}
// Generate C++ code from IR
void generateCode(shared_ptr<IRNode> node, const string& resultTempName, int indent = 0, int unrollFactor = 1) {
  
    if (!node)
        return;

    string indentStr(indent * 2, ' ');

    if (auto seqNode = dynamic_pointer_cast<SequenceNode>(node)) {
        // For nested sequence nodes, recurse with same substitution
        for (auto& operation : seqNode->operations) {
            generateCode(operation, resultTempName, indent);
        }
        return;
    }

    if (auto loopNode = dynamic_pointer_cast<LoopNode>(node)) {

        string shapeExpr = loopNode->shape_var.tensor_name->name + ".shape[" + to_string(loopNode->shape_var.idx_posn) + "]";
        string loopVar = string(1, loopNode->loop_variable);

        if (loopNode->start == 0 && loopNode->increment > 1) {
            // Main unrolled loop
            cout << indentStr << "for (int " << loopVar << " = 0; "
                 << loopVar << " < " << shapeExpr << " - " << (loopNode->end_offset) << "; "
                 << loopVar << " += " << loopNode->increment << ") {" << endl;
            generateCode(loopNode->body, resultTempName, indent + 1, unrollFactor);
            cout << indentStr << "}" << endl;
        }
        // Handle remainder loop
        else if (loopNode->start == -1) {
            // Remainder loop: start at N - (N % unrollFactor)
            cout << indentStr << "for (int " << loopVar << " = " << shapeExpr << " - (" << shapeExpr << " % " << unrollFactor << "); "
                 << loopVar << " < " << shapeExpr << "; "
                 << loopVar << "++) {" << endl;
            generateCode(loopNode->body, resultTempName, indent + 1, unrollFactor);
            cout << indentStr << "}" << endl;
        }
        // Handle standard loop
        else {
            cout << indentStr << "for (int " << loopVar << " = " << loopNode->start << "; "
                 << loopVar << " < " << shapeExpr << "; "
                 << loopVar << " += " << loopNode->increment << ") {" << endl;
            generateCode(loopNode->body, resultTempName, indent + 1, unrollFactor);
            cout << indentStr << "}" << endl;
        }
    } else if (auto calcNode = dynamic_pointer_cast<CalcNode>(node)) {
        string codeLine = calcNode->code_line;
        
        // Replace result temp name with output.data
        if (!resultTempName.empty()) {
            size_t pos = 0;
            while ((pos = codeLine.find(resultTempName, pos)) != string::npos) {
                codeLine.replace(pos, resultTempName.length(), "output.data");
                pos += strlen("output.data");
            }
        }
        
        cout << indentStr << codeLine << endl;
    }
}

// Helper function to generate kernel function signature
void generateKernelSignature(shared_ptr<SequenceNode> sequence, const vector<string>& inputTensorNames) {
  cout << "void kernel(";
  
  // Output parameter first
  cout << "tetricks_tensor& __restrict__ output";
  
  // Input parameters
  for (size_t i = 0; i < inputTensorNames.size(); i++) {
      cout << ", const tetricks_tensor& __restrict__ " << inputTensorNames[i];
  }
  
  cout << ") {" << endl;
}

void extractTensorNamesRecursive(shared_ptr<expr> ast, set<string>& names) {
  if (auto tensor = dynamic_pointer_cast<Tensor>(ast)) {
      // Only include original tensors (not generated temps)
      if (!tensor->dims_inferred || tensor->name.find("temp") != 0) {
          names.insert(tensor->name);
      }
  }
  
  if (auto binop = dynamic_pointer_cast<BinOpNode>(ast)) {
      extractTensorNamesRecursive(binop->inp1, names);
      extractTensorNamesRecursive(binop->inp2, names);
  }
  
  if (auto einsum = dynamic_pointer_cast<EinsumNode>(ast)) {
      for (auto& input : einsum->inputs) {
          names.insert(input->name);
      }
  }
}

void generateKernelCode(shared_ptr<SequenceNode> sequence, const vector<string>& inputTensorNames, int unrollFactor = 1) {
    // Generate function signature
    generateKernelSignature(sequence, inputTensorNames);
    cout << endl;
    
    // Generate temp declarations
    if (!sequence->temp_declarations.empty()) {
        cout << "  // Temporary variable declarations" << endl;
        for (auto& decl : sequence->temp_declarations) {
            cout << "  " << decl.second << endl;
        }
        cout << endl;
    }
    
    // Generate operations in sequence
    for (auto& operation : sequence->operations) {
      generateCode(operation, sequence->result_temp, 1, unrollFactor);  // Start with indent level 1
    }
    
    cout << "}" << endl;
}

// Helper function to extract input tensor names from AST
vector<string> extractInputTensorNames(shared_ptr<expr> ast) {
    set<string> uniqueNames;  // Use set to avoid duplicates
    extractTensorNamesRecursive(ast, uniqueNames);
    
    vector<string> result(uniqueNames.begin(), uniqueNames.end());
    return result;
}

// Helper function to get unique loop variables from einsum notation
set<char> getLoopVariables(const vector<string> &indexes) {
    set<char> variables;
    // Get variables from input tensor indexes (all except last)
    for (size_t i = 0; i < indexes.size() - 1; i++) {
      for (char c : indexes[i]) {
        variables.insert(c);
      }
    }
    return variables;
}

// Helper function to find position of character in string
int findPosition(const string &str, char c) {
    for (size_t i = 0; i < str.size(); i++) {
      if (str[i] == c) {
        return i;
      }
    }
    return -1; // not found
}

// Helper function to generate tensor access string
string generateTensorAccess(const string &tensorName, const string &indexPattern, const vector<char> &loopVars) {

    stringstream ss;
    ss << tensorName;
    for (char indexChar : indexPattern) {
      ss << "[" << indexChar << "]";
    }
    return ss.str();
}

// Helper function to generate calculation code for einsum
string generateEinsumCalc(const EinsumNode *einsum, const string &tempName) {
  stringstream ss;

  // Generate left side (output tensor access)
  string outputIndex = einsum->indexes.back(); // last index is output
    ss << tempName;
    if (!outputIndex.empty()) { // checking for cases where output contracts to a single variable
      for (char c : outputIndex) {
        ss << "[" << c << "]";
      }
    }
    

    ss << " += ";

    // Generate right side (multiplication of all input tensors)
    for (size_t i = 0; i < einsum->inputs.size(); i++) {
      if (i > 0)
        ss << " * ";
      ss << generateTensorAccess(einsum->inputs[i]->name, einsum->indexes[i], {});
    }

    ss << ";";
    return ss.str();
}

class IRGenerationContext {
  private:
      int temp_counter = 1;
      
  public:
      shared_ptr<Tensor> generateTempTensor(int num_dims) {
          string name = "temp" + to_string(temp_counter++);
          auto tensor = make_shared<Tensor>(name, num_dims, true);  // dims inferred = true
          return tensor;
      }
      
      void reset() { temp_counter = 1; }
  };

// Function to compute result dimensions of any expression
int getResultDimensions(shared_ptr<expr> node) {
  if (auto tensor = dynamic_pointer_cast<Tensor>(node)) {
      return tensor->num_dims;
  }
  
  if (auto einsum = dynamic_pointer_cast<EinsumNode>(node)) {
      // Count unique characters in output pattern
      const string& outputPattern = einsum->indexes.back();
      set<char> uniqueChars;
      for (char c : outputPattern) {
          uniqueChars.insert(c);
      }
      
      int dims = uniqueChars.size();
      
      // TODO: Handle total contraction case (empty output pattern)
      if (outputPattern.empty()) {
          // This is a scalar result (total contraction)
          // For now, treat as 0-dimensional
          return 0;  // Could also return -1 to indicate scalar
      }
      
      return dims;
  }
  
  if (auto binop = dynamic_pointer_cast<BinOpNode>(node)) {
      // For element-wise operations, result has same dimensions as operands
      int leftDims = getResultDimensions(binop->inp1);
      int rightDims = getResultDimensions(binop->inp2);
      
      // Verify dimensions match (they should after dimension inference)
      if (leftDims != -1 && rightDims != -1 && leftDims != rightDims) {
          throw runtime_error("Dimension mismatch in binary operation: " + 
                            to_string(leftDims) + " vs " + to_string(rightDims));
      }
      
      return (leftDims != -1) ? leftDims : rightDims;
  }
  
  return -1;  // Unknown
}

// Helper to check if expression results in scalar (total contraction)
bool isScalarResult(shared_ptr<expr> node) {
  if (auto einsum = dynamic_pointer_cast<EinsumNode>(node)) {
      return einsum->indexes.back().empty();  // Empty output pattern = scalar
  }
  return false;
}

// Helper function to convert einsum to nested loops
shared_ptr<IRNode> convertEinsumToLoops(const EinsumNode *einsum, const string &tempName, IRGenerationContext& ctx) {
    // 1. Identify output and contracted indices
    const string& outputPattern = einsum->indexes.back();
    set<char> outputIndices(outputPattern.begin(), outputPattern.end());

    // Collect all indices from input patterns
    set<char> allIndices;
    for (size_t i = 0; i < einsum->indexes.size() - 1; i++) {
        for (char c : einsum->indexes[i]) {
            allIndices.insert(c);
        }
    }

    // Contracted indices = all - output
    vector<char> contractedIndices;
    for (char c : allIndices) {
        if (outputIndices.find(c) == outputIndices.end()) {
            contractedIndices.push_back(c);
        }
    }

    // 2. Build ordered list: output indices (in outputPattern order), then contracted
    vector<char> orderedVars;
    for (char c : outputPattern) {
        orderedVars.push_back(c);
    }
    for (char c : contractedIndices) {
        orderedVars.push_back(c);
    }

    // 3. Build nested loops in this order
    shared_ptr<IRNode> innermost = make_shared<CalcNode>(generateEinsumCalc(einsum, tempName));
    shared_ptr<IRNode> current = innermost;

    for (int i = orderedVars.size() - 1; i >= 0; i--) {
        char loopVar = orderedVars[i];
        // Find a tensor that has this loop variable to determine bounds
        shared_ptr<Tensor> boundsTensor = nullptr;
        int boundsPosition = -1;

        for (size_t j = 0; j < einsum->inputs.size(); j++) {
            int pos = findPosition(einsum->indexes[j], loopVar);
            if (pos != -1) {
                boundsTensor = einsum->inputs[j];
                boundsPosition = pos;
                break;
            }
        }

        if (boundsTensor == nullptr) {
            throw runtime_error("Could not find bounds for loop variable: " + string(1, loopVar));
        }

        shape_variable shapeVar(boundsTensor, boundsPosition);
        current = make_shared<LoopNode>(loopVar, shapeVar, current);
    }

    return current;
}


// Helper function to convert BinOpKind to operator string
string binOpToString(BinOpKind op) {
  switch (op) {
    case Add: return "+";
    case Subtract: return "-";
    case Multiply: return "*";
    case Divide: return "/";
    default: throw runtime_error("Unknown binary operation");
  }
}

// Helper function to generate element-wise calculation code
string generateElementwiseCalc(shared_ptr<Tensor> resultTensor, shared_ptr<Tensor> leftTensor, shared_ptr<Tensor> rightTensor, BinOpKind op) {

    stringstream ss;
    // Generate left side (result tensor access)
    ss << resultTensor->name;
    for (int i = 0; i < resultTensor->num_dims; i++) {
        char loopVar;

        if (i < resultTensor->dim_sources.size() && resultTensor->dim_sources[i].loop_variable != '\0'){
            loopVar = resultTensor->dim_sources[i].loop_variable;
        } else {
            loopVar = 'i' + i; //default if inference doesnt work - code should nto reach here ideally.
        }

      ss << "[" << loopVar << "]";
    }
    
    ss << " = ";
    
    // Generate right side (left operand)
    ss << leftTensor->name;
    for (int i = 0; i < leftTensor->num_dims; i++) {
        char loopVar;
        if (i < resultTensor->dim_sources.size() && 
            resultTensor->dim_sources[i].loop_variable != '\0') {
            loopVar = resultTensor->dim_sources[i].loop_variable;
        } else {
            loopVar = 'i' + i;
        }
        ss << "[" << loopVar << "]";
    }
    ss << " " << binOpToString(op) << " ";
    
    // Generate right operand
    ss << rightTensor->name;
    for (int i = 0; i < rightTensor->num_dims; i++) {
      char loopVar;
      if (i < resultTensor->dim_sources.size() && resultTensor->dim_sources[i].loop_variable != '\0') {
        loopVar = resultTensor->dim_sources[i].loop_variable;
    } else {
      loopVar = 'i' + i;
    }
      ss << "[" << loopVar << "]";
    }
    
    ss << ";";
    return ss.str();
}

// Function to generate element-wise operation loops
shared_ptr<IRNode> generateElementwiseLoops(shared_ptr<Tensor> resultTensor, shared_ptr<Tensor> leftTensor,shared_ptr<Tensor> rightTensor, BinOpKind op) {
  
    // Handle scalar case (0 dimensions)
    if (resultTensor->num_dims == 0) {
      return make_shared<CalcNode>(generateElementwiseCalc(resultTensor, leftTensor, rightTensor, op));
    }
    
    // Generate innermost calculation
    shared_ptr<IRNode> innermost = make_shared<CalcNode>(generateElementwiseCalc(resultTensor, leftTensor, rightTensor, op)
    );
    
    // Build nested loops from inside out (reverse order)
    shared_ptr<IRNode> current = innermost;
    
    for (int dim = resultTensor->num_dims - 1; dim >= 0; dim--) {

      char loopVar;
      if (dim < resultTensor->dim_sources.size() && resultTensor->dim_sources[dim].loop_variable != '\0') {
        loopVar = resultTensor->dim_sources[dim].loop_variable;
      }
      else {
        loopVar = 'i' + dim;  // Fallback to default
      }
      
      // Create shape variable using dimension source tracking
      shared_ptr<Tensor> boundsTensor;
      int boundsPosition;
      
      if (dim < resultTensor->dim_sources.size() && resultTensor->dim_sources[dim].source_tensor) {
        // Use the tracked dimension source
        boundsTensor = resultTensor->dim_sources[dim].source_tensor;
        boundsPosition = resultTensor->dim_sources[dim].source_dim_index;
      } else {
        // Fallback: use left operand (they should have same dimensions)
        boundsTensor = leftTensor;
        boundsPosition = dim;
      }
      
      shape_variable shapeVar(boundsTensor, boundsPosition);
      current = make_shared<LoopNode>(loopVar, shapeVar, current);
    }
    
    return current;
}

string generateTensorDeclaration(shared_ptr<Tensor> tensor) {

  if (tensor->num_dims == 0) {
      // Scalar case (total contraction)
      return "double " + tensor->name + ";";
  }
  
  stringstream ss;
  ss << "double " << tensor->name;
  
  for (int i = 0; i < tensor->num_dims; i++) {
      ss << "[" << tensor->getDimensionRef(i) << "]";
  }
  
  ss << ";";
  return ss.str();
}

shared_ptr<Tensor> createTempForEinsum(const EinsumNode* einsum, IRGenerationContext& ctx) {

  const string& outputPattern = einsum->indexes.back();
  
  // Handle scalar case (total contraction)
  if (outputPattern.empty()) {
      // For now, create a 0-dimensional tensor - actually seems to be a fair workaround, might not need furtherchanges
      auto temp = ctx.generateTempTensor(0);
      // No dimension sources needed for scalar
      return temp;
  }
  
  // Create temp tensor with proper dimensions
  int outputDims = getResultDimensions(make_shared<EinsumNode>(*einsum));
  auto temp = ctx.generateTempTensor(outputDims);
  
  // Map each output dimension to its source
  for (int out_dim = 0; out_dim < outputDims; out_dim++) {
      char outputChar = outputPattern[out_dim];
      
      // Find which input tensor and dimension this character comes from
      for (size_t inp_idx = 0; inp_idx < einsum->inputs.size(); inp_idx++) {
          const string& inputPattern = einsum->indexes[inp_idx];
          
          for (size_t char_pos = 0; char_pos < inputPattern.size(); char_pos++) {
              if (inputPattern[char_pos] == outputChar) {
                  // Found the source: input tensor inp_idx, dimension char_pos
                  temp->setDimensionSource(out_dim, einsum->inputs[inp_idx], char_pos, outputChar);
                  goto next_output_dim;  // Break out of nested loops
              }
          }
      }
      next_output_dim:;
  }
  return temp;
}

shared_ptr<Tensor> processEinsum(const EinsumNode* einsum, shared_ptr<SequenceNode> sequence,  IRGenerationContext& ctx) {

    // Create temp tensor with dimension sources
    auto tempTensor = createTempForEinsum(einsum, ctx);

    // Generate temp declaration using dimension sources
    string decl = generateTensorDeclaration(tempTensor);
    sequence->addTempDeclaration(tempTensor->name, decl);

    // Generate loops (can use dimension sources for bounds)
    auto loops = convertEinsumToLoops(einsum, tempTensor->name, ctx);
    sequence->addOperation(loops);

    return tempTensor;
}


// Create temp tensor for BinOp result with dimension source tracking  
shared_ptr<Tensor> createTempForBinOp(shared_ptr<Tensor> leftTensor, shared_ptr<Tensor> rightTensor, IRGenerationContext& ctx) {  
      
    int dims = leftTensor->num_dims;  // Should match rightTensor->num_dims: users responsibilty
    auto temp = ctx.generateTempTensor(dims);

    // For element-wise operations, each dimension comes from the same dimension of operands
    // We can choose either left or right as source (they should have same dimensions)
    for (int i = 0; i < dims; i++) {
        char inferredLoopVar = '\0';
        // Use left operand as source - will have same dimension as output
        if (i < leftTensor->dim_sources.size() && leftTensor->dim_sources[i].source_tensor && leftTensor->dim_sources[i].loop_variable != '\0') {
        // Propagate from left operand's source
          temp->setDimensionSource(i, 
          leftTensor->dim_sources[i].source_tensor,
          leftTensor->dim_sources[i].source_dim_index,
          inferredLoopVar);
        }
        else if (i < rightTensor->dim_sources.size() && rightTensor->dim_sources[i].source_tensor && rightTensor->dim_sources[i].loop_variable != '\0') {
            // Right operand has loop variable from previous Einsum/BinOp
            inferredLoopVar = rightTensor->dim_sources[i].loop_variable;
            temp->setDimensionSource(i,rightTensor->dim_sources[i].source_tensor,rightTensor->dim_sources[i].source_dim_index,inferredLoopVar);
        }
        // Fallback: both operands are original tensors - use default mapping
        else {
            // Default loop variable assignment: i, j, k, l, ...
            inferredLoopVar = 'i' + i;
            if (i >= 26) inferredLoopVar = 'z'; // Fallback for many dimensions
            //TODO: remove this hacky logic and add something more robust
            // Use left operand as source tensor
            temp->setDimensionSource(i, leftTensor, i, inferredLoopVar);
        }
    }

    return temp;
}

shared_ptr<Tensor> processNode(shared_ptr<expr> node, shared_ptr<SequenceNode> sequence, IRGenerationContext& ctx);

shared_ptr<Tensor> processBinOp(const BinOpNode* binop, shared_ptr<SequenceNode> sequence, IRGenerationContext& ctx) {

  // Process operands recursively
  auto leftTensor = processNode(binop->inp1, sequence, ctx);
  auto rightTensor = processNode(binop->inp2, sequence, ctx);
  
  // Create result temp with dimension tracking
  auto resultTensor = createTempForBinOp(leftTensor, rightTensor, ctx);
  
  // Generate temp declaration
  string decl = generateTensorDeclaration(resultTensor);
  sequence->addTempDeclaration(resultTensor->name, decl);
  
  // Generate element-wise operation loops
  auto loops = generateElementwiseLoops(resultTensor, leftTensor, rightTensor, binop->opn);
  sequence->addOperation(loops);
  
  return resultTensor;
}


shared_ptr<Tensor> processNode(shared_ptr<expr> node, shared_ptr<SequenceNode> sequence, IRGenerationContext& ctx) {

    if (auto tensor = dynamic_pointer_cast<Tensor>(node)) {
        return tensor;  // Return original tensor directly
    }
    
    if (auto einsum = dynamic_pointer_cast<EinsumNode>(node)) {
        return processEinsum(einsum.get(), sequence, ctx);  // Return temp tensor
    }
    
    if (auto binop = dynamic_pointer_cast<BinOpNode>(node)) {
        return processBinOp(binop.get(), sequence, ctx);   // Return temp tensor
    }
    
    throw runtime_error("Unknown AST node type");
} 

shared_ptr<SequenceNode> convertASTtoIR(shared_ptr<expr> AST, IRGenerationContext& ctx) {
  /*
  Sets up overarching sequence node and calls processnode, the main function traversing AST
  */
  auto sequence = make_shared<SequenceNode>();
  auto resultTensor = processNode(AST, sequence, ctx);
  sequence->setResultTemp(resultTensor->name);  // Extract name here
  return sequence;
}


shared_ptr<expr> parseExprFromJson(const json &j) {
  // Handle array format: ["Type", ...args]
  if (j.is_array() && !j.empty()) {
    string nodeType = j[0].get<string>();
    
    if (nodeType == "Tensor") {
      if (j.size() != 2) {
        throw runtime_error("Invalid Tensor format");
      }
      string name = j[1].get<string>();
      return make_shared<Tensor>(name);
    }
    
    if (nodeType == "BinOp") {
      if (j.size() != 4) {
        throw runtime_error("Invalid BinOp format");
      }
      
      string opStr = j[1][0].get<string>(); // j[1] is ["Add"], so j[1][0] is "Add"
      auto lhs = parseExprFromJson(j[2]);
      auto rhs = parseExprFromJson(j[3]);
      BinOpKind op = parseBinOpKind(opStr);
      return make_shared<BinOpNode>(op, lhs, rhs);
    }
    
    if (nodeType == "Einsum") {
      if (j.size() != 2) {
        throw runtime_error("Invalid Einsum format");
      }
      
      const auto &einsum = j[1]; // This should be the object with notation and tensors
      
      string notation = einsum.at("notation").get<string>();
      vector<string> indexes = splitEinsumNotation(notation);
      
      vector<shared_ptr<Tensor>> tensorList;
      for (const auto &tensorName : einsum.at("tensors")) {
        string name = tensorName.get<string>();
        tensorList.push_back(make_shared<Tensor>(name));
      }
      
      return make_shared<EinsumNode>(indexes, tensorList);
    }
    
    throw runtime_error("Unknown AST node type: " + nodeType);
  }
  
  // Handle old object format (keeping for backward compatibility and test cases)
  if (j.is_object()) {
    if (j.contains("Tensor")) {
      string name = j.at("Tensor").get<string>();
      return make_shared<Tensor>(name);
    }

    if (j.contains("BinOp")) {
      const auto &binop = j.at("BinOp");
      if (!binop.is_array() || binop.size() != 3) {
        throw runtime_error("Invalid BinOp format");
      }

      string opStr = binop[0].get<string>();
      auto lhs = parseExprFromJson(binop[1]);
      auto rhs = parseExprFromJson(binop[2]);
      BinOpKind op = parseBinOpKind(opStr);
      return make_shared<BinOpNode>(op, lhs, rhs);
    }

    if (j.contains("Einsum")) {
      const auto &einsum = j.at("Einsum");

      string notation = einsum.at("notation").get<string>();
      vector<string> indexes = splitEinsumNotation(notation);

      vector<shared_ptr<Tensor>> tensorList;
      for (const auto &tensorExpr : einsum.at("tensors")) {
        auto tensorPtr = dynamic_pointer_cast<Tensor>(parseExprFromJson(tensorExpr));
        if (!tensorPtr) {
          throw runtime_error("Einsum tensor list must contain only tensors.");
        }
        tensorList.push_back(tensorPtr);
      }

      return make_shared<EinsumNode>(indexes, tensorList);
    }
  }
  
  throw runtime_error("Expected JSON array or object for expr node.");
}

// =============================================================================
// OPTIMISATIONS
// =============================================================================

// Loop Optimization Infrastructure
struct Dependencies {
  set<string> reads;   // Variables/tensors read (e.g., "A.data", "temp1")
  set<string> writes;  // Variables/tensors written (e.g., "temp1", "output.data")
};

// Helper function to detect if a LoopNode has been unrolled
bool isUnrolledLoop(LoopNode* loop) {
  // Check if the body is a SequenceNode with multiple similar operations
  if (auto bodySeq = dynamic_pointer_cast<SequenceNode>(loop->body)) {
      return bodySeq->operations.size() > 1;
  }
  return false;
}

// Modified code generation for unrolled loops
void generateUnrolledLoopCode(LoopNode* loop, int indent = 0) {
  string indentStr(indent * 2, ' ');
  
  if (auto bodySeq = dynamic_pointer_cast<SequenceNode>(loop->body)) {
      int unrollFactor = bodySeq->operations.size();
      
      cout << indentStr << "// Unrolled loop (factor: " << unrollFactor << ")" << endl;
      cout << indentStr << "for (int " << loop->loop_variable << " = 0; "
           << loop->loop_variable << " < " 
           << loop->shape_var.tensor_name->name << ".shape["
           << loop->shape_var.idx_posn << "] - " << (unrollFactor - 1) << "; "
           << loop->loop_variable << " += " << unrollFactor << ") {" << endl;
      
      // Generate unrolled body
      for (auto& operation : bodySeq->operations) {
          generateCode(operation, "", indent + 1);
      }
      
      cout << indentStr << "}" << endl;
      
      // Generate remainder loop for leftover iterations
      cout << indentStr << "// Remainder loop" << endl;
      cout << indentStr << "for (int " << loop->loop_variable << " = " 
           << loop->shape_var.tensor_name->name << ".shape["
           << loop->shape_var.idx_posn << "] - (" 
           << loop->shape_var.tensor_name->name << ".shape["
           << loop->shape_var.idx_posn << "] % " << unrollFactor << "); "
           << loop->loop_variable << " < "
           << loop->shape_var.tensor_name->name << ".shape["
           << loop->shape_var.idx_posn << "]; "
           << loop->loop_variable << "++) {" << endl;
      
      // Generate original body for remainder
      if (!bodySeq->operations.empty()) {
          // Use first operation as template, but remove the offset
          auto firstOp = bodySeq->operations[0];
          if (auto calcNode = dynamic_pointer_cast<CalcNode>(firstOp)) {
              string originalCode = calcNode->code_line;
              // Remove any "+ 0" patterns that might exist
              size_t pos = originalCode.find(" + 0");
              while (pos != string::npos) {
                  originalCode.erase(pos, 4);
                  pos = originalCode.find(" + 0", pos);
              }
              // Remove parentheses around single variables
              pos = originalCode.find("(" + string(1, loop->loop_variable) + ")");
              while (pos != string::npos) {
                  originalCode.replace(pos, 3, string(1, loop->loop_variable));
                  pos = originalCode.find("(" + string(1, loop->loop_variable) + ")", pos);
              }
              
              cout << indentStr << "  " << originalCode << endl;
          }
      }
      
      cout << indentStr << "}" << endl;
  }
};

// Helper to extract base variable name (e.g., "temp1[i][j]" -> "temp1")
string extractBaseVariable(const string& expr) {
  string trimmed = expr;
  
  // Remove whitespace
  trimmed.erase(remove_if(trimmed.begin(), trimmed.end(), ::isspace), trimmed.end());
  
  // Find first '[' or end of string
  size_t bracketPos = trimmed.find('[');
  if (bracketPos != string::npos) {
      return trimmed.substr(0, bracketPos);
  }
  
  return trimmed;
}

// Helper to extract all variable names from an expression
void extractVariablesFromExpression(const string& expr, set<string>& variables) {
  // Simple approach: look for patterns like "word.word" or "word"
  // followed by optional array indices [...]
  
  size_t pos = 0;
  while (pos < expr.length()) {
      // Skip non-alphabetic characters
      while (pos < expr.length() && !isalpha(expr[pos])) {
          pos++;
      }
      
      if (pos >= expr.length()) break;
      
      // Extract variable name (letters, digits, dots, underscores)
      string varName;
      while (pos < expr.length() && 
             (isalnum(expr[pos]) || expr[pos] == '.' || expr[pos] == '_')) {
          varName += expr[pos];
          pos++;
      }
      
      // Skip array indices [...]
      while (pos < expr.length() && expr[pos] == '[') {
          int bracketCount = 1;
          pos++; // Skip opening '['
          while (pos < expr.length() && bracketCount > 0) {
              if (expr[pos] == '[') bracketCount++;
              if (expr[pos] == ']') bracketCount--;
              pos++;
          }
      }
      
      if (!varName.empty()) {
          variables.insert(varName);
      }
  }
}

// Parse CalcNode code to extract variable dependencies
Dependencies analyzeDependencies(CalcNode* calc) {
  Dependencies deps;
  string code = calc->code_line;
  
  // Check for compound assignment operators (+=, -=, *=, /=)
  bool isCompoundAssignment = false;
  size_t assignPos = code.find("+=");
  if (assignPos == string::npos) {
      assignPos = code.find("-=");
  }
  if (assignPos == string::npos) {
      assignPos = code.find("*=");
  }
  if (assignPos == string::npos) {
      assignPos = code.find("/=");
  }
  if (assignPos != string::npos) {
      isCompoundAssignment = true;
  } else {
      // Look for regular assignment
      assignPos = code.find('=');
      if (assignPos == string::npos) {
          return deps; // No assignment found
      }
  }
  
  string leftSide = code.substr(0, assignPos);
  string rightSide = code.substr(assignPos + (isCompoundAssignment ? 2 : 1));
  
  // Extract write target from left side
  string writeVar = extractBaseVariable(leftSide);
  if (!writeVar.empty()) {
      deps.writes.insert(writeVar);
      
      // For compound assignments, left side is also read
      if (isCompoundAssignment) {
          deps.reads.insert(writeVar);
      }
  }
  
  // Extract read variables from right side
  extractVariablesFromExpression(rightSide, deps.reads);
  
  return deps;
}

// Check if two CalcNodes have data dependencies
bool hasDependency(CalcNode* calc1, CalcNode* calc2) {
  auto deps1 = analyzeDependencies(calc1);
  auto deps2 = analyzeDependencies(calc2);
  
  // Check for any intersection between reads and writes
  for (const string& write1 : deps1.writes) {
      if (deps2.reads.find(write1) != deps2.reads.end()) {
          return true; // RAW dependency: calc2 reads what calc1 writes
      }
  }
  
  for (const string& read1 : deps1.reads) {
      if (deps2.writes.find(read1) != deps2.writes.end()) {
          return true; // WAR dependency: calc2 writes what calc1 reads
      }
  }
  
  for (const string& write1 : deps1.writes) {
      if (deps2.writes.find(write1) != deps2.writes.end()) {
          return true; // WAW dependency: both write to same variable
      }
  }
  
  return false; // No dependencies found
}

// Helper to check if a node contains any child loops
bool hasChildLoops(shared_ptr<IRNode> node) {
  if (!node) return false;
  
  if (dynamic_pointer_cast<LoopNode>(node)) {
      return true; // Found a child loop
  }
  
  if (auto seqNode = dynamic_pointer_cast<SequenceNode>(node)) {
      // Check all operations in sequence
      for (auto& operation : seqNode->operations) {
          if (hasChildLoops(operation)) {
              return true;
          }
      }
  }
  
  return false; // No child loops found
}

void findInnermostLoopsRecursive(shared_ptr<IRNode> node, vector<LoopNode*>& innermostLoops) {
  if (!node) return;
  
  if (auto loopNode = dynamic_pointer_cast<LoopNode>(node)) {
      // Check if this loop has any child loops
      if (!hasChildLoops(loopNode->body)) {
          innermostLoops.push_back(loopNode.get());
      } else {
          // Recurse into body to find deeper innermost loops
          findInnermostLoopsRecursive(loopNode->body, innermostLoops);
      }
  } else if (auto seqNode = dynamic_pointer_cast<SequenceNode>(node)) {
      // Recurse into all operations in sequence
      for (auto& operation : seqNode->operations) {
          findInnermostLoopsRecursive(operation, innermostLoops);
      }
  }
  // CalcNode has no children, so nothing to do
}

// Find all innermost loops in the IR (loops with no child loops)
vector<LoopNode*> findInnermostLoops(shared_ptr<IRNode> root) {
  vector<LoopNode*> innermostLoops;
  findInnermostLoopsRecursive(root, innermostLoops);
  return innermostLoops;
}

// Find adjacent loops in a SequenceNode that might be candidates for fusion
vector<pair<LoopNode*, LoopNode*>> findAdjacentLoopsInSequence(SequenceNode* seq) {
  vector<pair<LoopNode*, LoopNode*>> adjacentPairs;
  
  for (size_t i = 0; i < seq->operations.size() - 1; i++) {
      auto loop1 = dynamic_pointer_cast<LoopNode>(seq->operations[i]);
      auto loop2 = dynamic_pointer_cast<LoopNode>(seq->operations[i + 1]);
      
      if (loop1 && loop2) {
          adjacentPairs.push_back({loop1.get(), loop2.get()});
      }
  }
  
  return adjacentPairs;
}

// Loop Unrolling Optimizer
class LoopUnrollOptimizer {
public:
  shared_ptr<IRNode> optimize(shared_ptr<IRNode> ir, int unrollFactor) {
      if (unrollFactor <= 1) {
          return ir; // No unrolling needed - Code should not reach here. should have been filtered away by main.
      }
      
      return unrollNode(ir, unrollFactor);
  }
  
private:

  shared_ptr<IRNode> unrollNode(shared_ptr<IRNode> node, int unrollFactor) {

      if (!node) return node; //defense in case some nullptr reaches here. Something very seriously wrong then.
      
      if (auto loopNode = dynamic_pointer_cast<LoopNode>(node)) {
          // Check if this is an innermost loop
          if (!hasChildLoops(loopNode->body)) {
              return unrollLoop(loopNode.get(), unrollFactor);
          } else {
              // Recurse into body for nested loops
              auto newBody = unrollNode(loopNode->body, unrollFactor);
              return make_shared<LoopNode>(loopNode->loop_variable, loopNode->shape_var, newBody);
          }
      } else if (auto seqNode = dynamic_pointer_cast<SequenceNode>(node)) {
          // Create new sequence with unrolled operations
          auto newSeq = make_shared<SequenceNode>();
          newSeq->temp_declarations = seqNode->temp_declarations;
          newSeq->result_temp = seqNode->result_temp;
          
          for (auto& operation : seqNode->operations) {
              newSeq->addOperation(unrollNode(operation, unrollFactor));
          }
          
          return newSeq;
      } else if (auto calcNode = dynamic_pointer_cast<CalcNode>(node)) {
          // CalcNode doesn't need unrolling modification at this level
          return node;
      }
      
      return node;
  }
  
  shared_ptr<IRNode> unrollLoop(LoopNode* loop, int unrollFactor) {
      // Simple unrolling: replicate the loop body unrollFactor times
      // and adjust the loop bound accordingly
      
      char loopVar = loop->loop_variable;

      auto seq = make_shared<SequenceNode>(); // to hold unrolled loop and reminder loop

      auto unrolledBody = createUnrolledBody(loop->body, loopVar, unrollFactor);
      auto mainLoop = make_shared<LoopNode>(loopVar, loop->shape_var, unrolledBody, 0, unrollFactor, unrollFactor - 1);
      seq->addOperation(mainLoop);

      auto remainderBody = loop->body;// body of original loop for reminder loop
      auto remainderLoop = make_shared<LoopNode>(loopVar, loop->shape_var, remainderBody, -1, 1, 0); // "-1" for loop start variable as a placeholder, will be handled at codegen as a special case for reminder loops.
      seq->addOperation(remainderLoop);
      return seq;

  }
  
  shared_ptr<IRNode> createUnrolledBody(shared_ptr<IRNode> originalBody, char loopVar, int unrollFactor) {
      auto bodySeq = make_shared<SequenceNode>();
      
      // Generate unrollFactor copies of the body with substituted loop variables
      for (int i = 0; i < unrollFactor; i++) {
          auto unrolledIteration = substituteLoopVariable(originalBody, loopVar, i);
          bodySeq->addOperation(unrolledIteration);
      }
      
      return bodySeq;
  }
  
  shared_ptr<IRNode> substituteLoopVariable(shared_ptr<IRNode> node, char loopVar, int offset) {
      if (!node) return node;
      
      if (auto calcNode = dynamic_pointer_cast<CalcNode>(node)) {
          // Substitute loop variable in the calculation
          string newCode = calcNode->code_line;
          string oldVar = string(1, loopVar);
          string newVar = oldVar + " + " + to_string(offset);
          
          // Replace all occurrences of the loop variable
          newCode = substituteLoopVarInString(newCode, oldVar, newVar);
          
          return make_shared<CalcNode>(newCode);
      } else if (auto seqNode = dynamic_pointer_cast<SequenceNode>(node)) {
          auto newSeq = make_shared<SequenceNode>();
          newSeq->temp_declarations = seqNode->temp_declarations;
          newSeq->result_temp = seqNode->result_temp;
          
          for (auto& operation : seqNode->operations) {
              newSeq->addOperation(substituteLoopVariable(operation, loopVar, offset));
          }
          
          return newSeq;
      } else if (auto loopNode = dynamic_pointer_cast<LoopNode>(node)) {
          // This shouldn't happen for innermost loops, but handle it
          auto newBody = substituteLoopVariable(loopNode->body, loopVar, offset);
          return make_shared<LoopNode>(loopNode->loop_variable, loopNode->shape_var, newBody);
      }
      
      return node;
  }
  
  string substituteLoopVarInString(const string& code, const string& oldVar, const string& newVar) {
      string result = code;
      size_t pos = 0;
      
      while ((pos = result.find(oldVar, pos)) != string::npos) {
          // Check if this is a complete variable name (not part of a larger identifier)
          bool isCompleteVar = true;
          
          // Check character before
          if (pos > 0 && (isalnum(result[pos - 1]) || result[pos - 1] == '_')) {
              isCompleteVar = false;
          }
          
          // Check character after
          if (pos + oldVar.length() < result.length() && 
              (isalnum(result[pos + oldVar.length()]) || result[pos + oldVar.length()] == '_')) {
              isCompleteVar = false;
          }
          
          if (isCompleteVar) {
              result.replace(pos, oldVar.length(), "(" + newVar + ")");
              pos += newVar.length() + 2; // Account for added parentheses
          } else {
              pos += oldVar.length();
          }
      }
      
      return result;
  }
  
  shared_ptr<IRNode> cloneIRNode(shared_ptr<IRNode> node) {
      if (!node) return node;
      
      if (auto calcNode = dynamic_pointer_cast<CalcNode>(node)) {
          return make_shared<CalcNode>(calcNode->code_line);
      } else if (auto loopNode = dynamic_pointer_cast<LoopNode>(node)) {
          return make_shared<LoopNode>(loopNode->loop_variable, loopNode->shape_var, 
                                     cloneIRNode(loopNode->body));
      } else if (auto seqNode = dynamic_pointer_cast<SequenceNode>(node)) {
          auto newSeq = make_shared<SequenceNode>();
          newSeq->temp_declarations = seqNode->temp_declarations;
          newSeq->result_temp = seqNode->result_temp;
          
          for (auto& operation : seqNode->operations) {
              newSeq->addOperation(cloneIRNode(operation));
          }
          
          return newSeq;
      }
      
      return node;
  }
};


// Loop Fusion Optimizer
class LoopFusionOptimizer {
  public:
      shared_ptr<IRNode> optimize(shared_ptr<IRNode> ir) {
          return fuseLoopsRecursive(ir);
      }
  
  private:
      shared_ptr<IRNode> fuseLoopsRecursive(shared_ptr<IRNode> node) {
          if (!node) return node;
  
          if (auto seqNode = dynamic_pointer_cast<SequenceNode>(node)) {
              return fuseLoopsInSequence(seqNode.get());
          } 
          else if (auto loopNode = dynamic_pointer_cast<LoopNode>(node)) {
              // Recursively process the body first
              auto newBody = fuseLoopsRecursive(loopNode->body);
              return make_shared<LoopNode>(loopNode->loop_variable, loopNode->shape_var, 
                                         newBody, loopNode->start, loopNode->increment, 
                                         loopNode->end_offset);
          }
          
          return node; // CalcNode or other types don't need fusion
      }
  
      shared_ptr<IRNode> fuseLoopsInSequence(SequenceNode* seq) {
          auto newSeq = make_shared<SequenceNode>();
          newSeq->temp_declarations = seq->temp_declarations;
          newSeq->result_temp = seq->result_temp;
  
          vector<shared_ptr<IRNode>> processedOps;
          
          for (size_t i = 0; i < seq->operations.size(); ++i) {
              auto currentOp = fuseLoopsRecursive(seq->operations[i]);
              
              // Try to fuse with previous operation
              if (!processedOps.empty()) {
                  auto prevOp = processedOps.back();
                  auto fused = tryFuseLoops(prevOp, currentOp);
                  
                  if (fused) {
                      processedOps.back() = fused; // Replace previous with fused version
                      continue; // Skip adding current operation separately
                  }
              }
              
              processedOps.push_back(currentOp);
          }
          
          // Add all processed operations to new sequence
          for (auto& op : processedOps) {
              newSeq->addOperation(op);
          }
          
          return newSeq;
      }
  
      shared_ptr<IRNode> tryFuseLoops(shared_ptr<IRNode> loop1, shared_ptr<IRNode> loop2) {
          auto loopNode1 = dynamic_pointer_cast<LoopNode>(loop1);
          auto loopNode2 = dynamic_pointer_cast<LoopNode>(loop2);
          
          if (!loopNode1 || !loopNode2) {
              return nullptr; // Can only fuse LoopNodes
          }
          
          // Check if loops can be fused
          if (!canFuseLoops(loopNode1.get(), loopNode2.get())) {
              return nullptr;
          }
          
          return performLoopFusion(loopNode1.get(), loopNode2.get());
      }
  
      bool canFuseLoops(LoopNode* loop1, LoopNode* loop2) {
          // Check 1: Same loop variable
          if (loop1->loop_variable != loop2->loop_variable) {
              return false;
          }
          
          // Check 2: Trust user for bounds compatibility - skip bounds check
          // User is responsible for ensuring dimension compatibility
          
          // Check 3: Same loop parameters (start, increment, end_offset)
          if (loop1->start != loop2->start || 
              loop1->increment != loop2->increment || 
              loop1->end_offset != loop2->end_offset) {
              return false;
          }
          
          // Check 4: No data dependencies that would be violated
          if (hasViolatingDependencies(loop1, loop2)) {
              return false;
          }
          
          return true;
      }
  
      // Removed haveSameBounds() - trusting user for dimension compatibility
  
      bool hasViolatingDependencies(LoopNode* loop1, LoopNode* loop2) {
          // Get all CalcNodes from both loop bodies
          vector<CalcNode*> body1Calcs = extractCalcNodes(loop1->body);
          vector<CalcNode*> body2Calcs = extractCalcNodes(loop2->body);
          
          // Check for dependencies between operations in different loops
          for (CalcNode* calc1 : body1Calcs) {
              for (CalcNode* calc2 : body2Calcs) {
                  if (hasDependency(calc1, calc2)) {
                      // Found a dependency - check if fusion would violate it
                      if (wouldViolateDependency(calc1, calc2)) {
                          return true;
                      }
                  }
              }
          }
          
          return false;
      }
  
      bool wouldViolateDependency(CalcNode* calc1, CalcNode* calc2) {
          // For loop fusion, we need to ensure that if calc1 writes to a variable
          // that calc2 reads, and they access the same array elements in the same
          // iteration, then fusion is safe. If they access different elements
          // based on the loop variable, fusion might change the order of operations.
          
          auto deps1 = analyzeDependencies(calc1);
          auto deps2 = analyzeDependencies(calc2);
          
          // Check for RAW (Read After Write) dependencies
          for (const string& write1 : deps1.writes) {
              if (deps2.reads.find(write1) != deps2.reads.end()) {
                  // calc2 reads what calc1 writes
                  // This is safe for fusion if they're accessing the same elements
                  // in the same iteration (which is typically the case for element-wise ops)
                  // For now, we'll be conservative and allow fusion for most cases
                  return false; // Allow fusion for same-iteration dependencies
              }
          }
          
          // WAR and WAW dependencies are generally safe for fusion if they
          // occur within the same iteration
          return false; // Conservative: allow fusion
      }
  
      vector<CalcNode*> extractCalcNodes(shared_ptr<IRNode> node) {
          vector<CalcNode*> calcNodes;
          extractCalcNodesRecursive(node, calcNodes);
          return calcNodes;
      }
  
      void extractCalcNodesRecursive(shared_ptr<IRNode> node, vector<CalcNode*>& calcNodes) {
          if (!node) return;
          
          if (auto calcNode = dynamic_pointer_cast<CalcNode>(node)) {
              calcNodes.push_back(calcNode.get());
          }
          else if (auto loopNode = dynamic_pointer_cast<LoopNode>(node)) {
              extractCalcNodesRecursive(loopNode->body, calcNodes);
          }
          else if (auto seqNode = dynamic_pointer_cast<SequenceNode>(node)) {
              for (auto& operation : seqNode->operations) {
                  extractCalcNodesRecursive(operation, calcNodes);
              }
          }
      }
  
      shared_ptr<IRNode> performLoopFusion(LoopNode* loop1, LoopNode* loop2) {
          // Create a new fused loop with deeply merged body
          auto fusedBody = createDeeplyFusedBody(loop1->body, loop2->body);
          
          // Use the parameters from loop1 (they should be the same anyway)
          return make_shared<LoopNode>(
              loop1->loop_variable,
              loop1->shape_var,
              fusedBody,
              loop1->start,
              loop1->increment,
              loop1->end_offset
          );
      }
  
      shared_ptr<IRNode> createDeeplyFusedBody(shared_ptr<IRNode> body1, shared_ptr<IRNode> body2) {
          // Check if both bodies are LoopNodes with the same loop variable
          auto loop1 = dynamic_pointer_cast<LoopNode>(body1);
          auto loop2 = dynamic_pointer_cast<LoopNode>(body2);
          
          if (loop1 && loop2 && loop1->loop_variable == loop2->loop_variable) {
              // Both are loops with same variable - recursively fuse them
              auto deeplyFusedInner = createDeeplyFusedBody(loop1->body, loop2->body);
              
              // Use loop1's parameters (assuming they match)
              return make_shared<LoopNode>(
                  loop1->loop_variable,
                  loop1->shape_var,  // Trust user that bounds are compatible
                  deeplyFusedInner,
                  loop1->start,
                  loop1->increment,
                  loop1->end_offset
              );
          } else {
              // At least one is not a compatible loop - create sequence
              return createFusedBody(body1, body2);
          }
      }
  
      shared_ptr<IRNode> createFusedBody(shared_ptr<IRNode> body1, shared_ptr<IRNode> body2) {
          // Create a sequence node that contains operations from both bodies
          auto fusedSeq = make_shared<SequenceNode>();
          
          // Add operations from first body
          addOperationsToSequence(body1, fusedSeq.get());
          
          // Add operations from second body
          addOperationsToSequence(body2, fusedSeq.get());
          
          return fusedSeq;
      }
  
      void addOperationsToSequence(shared_ptr<IRNode> source, SequenceNode* target) {
          if (!source) return;
          
          if (auto sourceSeq = dynamic_pointer_cast<SequenceNode>(source)) {
              // Copy temp declarations (merge them)
              for (const auto& decl : sourceSeq->temp_declarations) {
                  target->addTempDeclaration(decl.first, decl.second);
              }
              
              // Add all operations
              for (auto& operation : sourceSeq->operations) {
                  target->addOperation(operation);
              }
              
              // Handle result temp (keep the last one, or merge logic as needed)
              if (!sourceSeq->result_temp.empty()) {
                  target->setResultTemp(sourceSeq->result_temp);
              }
          }
          else {
              // Single operation (CalcNode, LoopNode, etc.)
              target->addOperation(source);
          }
      }
  };

// =============================================================================
// MAIN FUNCTION
// =============================================================================

// Modified main function (replace the code generation part)
int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " ast.json [-debug] [-unroll N]" << endl;
    return 1;
  }

  string filename = argv[1];
  bool debugMode = false, fusionEnabled= false;
  int unrollFactor = 1;

  // Parse optional arguments
  for (int i = 2; i < argc; ++i) {
      if (string(argv[i]) == "-debug") debugMode = true;
      if (string(argv[i]) == "-unroll" && i + 1 < argc) {
          unrollFactor = stoi(argv[i + 1]);
          ++i;
      }
      if (string(argv[i]) == "-fusion") fusionEnabled = true;
  }

  ifstream inFile(filename);
  if (!inFile) {
    cout << "Error: Cannot open file " << filename << endl;
    return 1;
  }

  try {
    json astJson;
    inFile >> astJson;

    auto ast = parseExprFromJson(astJson);

    // Extract input tensor names for function signature
    vector<string> inputTensorNames = extractInputTensorNames(ast);

    DimensionInferenceEngine dimEngine;
    dimEngine.inferAllDimensions(ast);
    
    if (debugMode) {
        cout << "=== Dimension inferred ===" << endl;
        dimEngine.printAllDimensions(ast);
        cout << endl;
    }

    IRGenerationContext ctx;
    auto IR = convertASTtoIR(ast, ctx);

    if (debugMode) {
      cout << "=== IR Structure ===" << endl;
      printIR(IR);
      cout << endl;
    }

    if (fusionEnabled){
      LoopFusionOptimizer loopfuser;
      IR = dynamic_pointer_cast<SequenceNode>(loopfuser.optimize(IR));
      if (debugMode) {
          cout << "=== IR after Loop Fusion ===" << endl;
          printIR(IR);
          cout << endl;
      }
    }

    if (unrollFactor >1){
        LoopUnrollOptimizer loopunroller;
        IR = dynamic_pointer_cast<SequenceNode>(loopunroller.optimize(IR, unrollFactor));
        if (debugMode){
          cout << "=== IR after Loop Unrool ===" << endl;
          printIR(IR);
          cout<<endl;
        }
    }

    if (debugMode){
        cout << "=== Generated Kernel ===" << endl;
    }
    generateKernelCode(IR, inputTensorNames);
    cout << endl;

  } catch (const exception &e) {
    cout << "Error: " << e.what() << endl;
    return 1;
  }

  return 0;
}