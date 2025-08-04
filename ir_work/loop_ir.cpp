/*
Program that takes in AST and builds loops
*/
#include <iostream>
#include <memory>
#include <set>
#include <map>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;

#include <fstream>
#include "json.hpp"
using json = nlohmann::json;


enum BinOpKind { Add, Subtract, Multiply, Divide };


//forr reading from json
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
  int num_dims = -1;  // -1 means unknown/uninitialized
  bool dims_inferred = false;  // track if dimensions were inferred
  
  Tensor(const string &n) : name(n) {}
  Tensor(const string &n, int dims, bool inferred = false) 
      : name(n), num_dims(dims), dims_inferred(inferred) {}
};

struct EinsumNode : expr {
  /*
  indexes; list of einsum indices: Eg: einsum("ij,jk,kl -> il" , A,B,C) would
  return an EinsumNode object with indices string "ij","jk","ik","il" inputs:
  list of tensor objects to be operated on.
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
  BinOpNode(BinOpKind operation, shared_ptr<expr> input1,
            shared_ptr<expr> input2)
      : opn(operation), inp1(input1), inp2(input2) {}
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
  shape_variable(const shared_ptr<Tensor> t_name, int pos)
      : tensor_name(t_name), idx_posn(pos) {}
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
  EinsumNode("ij","ji",A) would have LoopNode(i, A[0], LoopNode(j,A[1],
  Calcnode(temp1[j][i] = A[i][j]) )) BinOpNode(Multiply, A,B) would have
  LoopNode()
  */
  char loop_variable;
  shape_variable shape_var;
  shared_ptr<IRNode> body;
  LoopNode(char var, const shape_variable &shape, shared_ptr<IRNode> b)
      : loop_variable(var), shape_var(shape), body(b) {}
};

struct SequenceNode : IRNode {
  /*
  Essentialy queue of loop nests. BinOpNode(Add, EinsumNode("ij,jk->ik",A,B), C)
  */
  vector<shared_ptr<IRNode>> operations;
  
  SequenceNode(const vector<shared_ptr<IRNode>>& ops) : operations(ops) {}
  
  // Helper to add operations
  void addOperation(shared_ptr<IRNode> op) {
      operations.push_back(op);
  }
};

// Global counter for temporary variables
static int temp_counter = 1;

// Print IR structure (abstract representation)
void printIR(shared_ptr<IRNode> node, int indent = 0) {
  if (!node)
    return;

  string indentStr(indent * 2, ' ');

  if (auto loopNode = dynamic_pointer_cast<LoopNode>(node)) {
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
void generateCode(shared_ptr<IRNode> node, int indent = 0) {
  if (!node)
    return;

  string indentStr(indent * 2, ' ');

  if (auto loopNode = dynamic_pointer_cast<LoopNode>(node)) {
    cout << indentStr << "for (int " << loopNode->loop_variable << " = 0; "
         << loopNode->loop_variable << " < "
         << loopNode->shape_var.tensor_name->name << ".shape["
         << loopNode->shape_var.idx_posn << "]; " << loopNode->loop_variable
         << "++) {" << endl;
    generateCode(loopNode->body, indent + 1);
    cout << indentStr << "}" << endl;
  } else if (auto calcNode = dynamic_pointer_cast<CalcNode>(node)) {
    cout << indentStr << calcNode->code_line << endl;
  }
}

string generateTempName() { return "temp" + to_string(temp_counter++); }

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
string generateTensorAccess(const string &tensorName,
                            const string &indexPattern,
                            const vector<char> &loopVars) {
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

// Helper function to convert einsum to nested loops
shared_ptr<IRNode> convertEinsumToLoops(const EinsumNode *einsum, const string &tempName) {

  set<char> loopVars = getLoopVariables(einsum->indexes);
  vector<char> orderedVars(loopVars.begin(), loopVars.end());

  if (orderedVars.empty()) {
    // No loops needed, just direct assignment
    return make_shared<CalcNode>(generateEinsumCalc(einsum, tempName));
  }

  // Generate innermost calculation
  shared_ptr<IRNode> innermost = make_shared<CalcNode>(generateEinsumCalc(einsum, tempName));

  // Build nested loops from inside out
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
        throw runtime_error("Could not find bounds for loop variable: " +
                            string(1, loopVar));
      }

      shape_variable shapeVar(boundsTensor, boundsPosition);
      current = make_shared<LoopNode>(loopVar, shapeVar, current);
  }

  return current;
}

shared_ptr<IRNode> convertASTtoIR(shared_ptr<expr> AST) {
  // Handle EinsumNode
  if (auto einsumPtr = dynamic_pointer_cast<EinsumNode>(AST)) {
    string tempName = generateTempName();
    return convertEinsumToLoops(einsumPtr.get(), tempName);
  }

  // Handle BinOpNode
  if (auto binOpPtr = dynamic_pointer_cast<BinOpNode>(AST)) {
    // Check if both operands are tensors (unimplemented case)
    bool leftIsTensor = dynamic_pointer_cast<Tensor>(binOpPtr->inp1) != nullptr;
    bool rightIsTensor =
        dynamic_pointer_cast<Tensor>(binOpPtr->inp2) != nullptr;

    if (leftIsTensor && rightIsTensor) {
      throw runtime_error(
          "Binary operations between two tensors not yet supported");
    }

    // Convert left operand
    shared_ptr<IRNode> leftIR = convertASTtoIR(binOpPtr->inp1);

    // Handle case where right operand is a tensor
    if (rightIsTensor) {
      auto rightTensor = dynamic_pointer_cast<Tensor>(binOpPtr->inp2);
      string tempName = generateTempName();

      // Generate operation code
      string opStr;
      switch (binOpPtr->opn) {
      case Add:
        opStr = " + ";
        break;
      case Subtract:
        opStr = " - ";
        break;
      case Multiply:
        opStr = " * ";
        break;
      case Divide:
        opStr = " / ";
        break;
      }

      // For now, create a simple calculation node
      // This is a simplified implementation - in practice you'd need to handle
      // the tensor dimensions properly
      string calcCode =
          tempName + " = prev_result" + opStr + rightTensor->name + ";";
      shared_ptr<IRNode> calcNode = make_shared<CalcNode>(calcCode);

      return calcNode;
    }

    // If right operand is not a tensor, convert it recursively
    shared_ptr<IRNode> rightIR = convertASTtoIR(binOpPtr->inp2);

    // For complex binary operations, you would need to combine the IRs
    // This is a simplified version
    return leftIR;
  }

  // Handle Tensor (leaf node)
  if (auto tensorPtr = dynamic_pointer_cast<Tensor>(AST)) {
    // A single tensor doesn't generate loops, just return identity
    return make_shared<CalcNode>("// Tensor: " + tensorPtr->name);
  }

  throw runtime_error("Unknown AST node type");
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
// MAIN FUNCTION
// =============================================================================

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " ast.json [-debug]" << endl;
    return 1;
  }

  string filename = argv[1];
  bool debugMode = (argc >= 3 && string(argv[2]) == "-debug");

  ifstream inFile(filename);
  if (!inFile) {
    cout << "Error: Cannot open file " << filename << endl;
    return 1;
  }

  try {
    json astJson;
    inFile >> astJson;

    auto ast = parseExprFromJson(astJson);

    DimensionInferenceEngine dimEngine;
    dimEngine.inferAllDimensions(ast);
    
    if (debugMode) {
        cout<<"=== Dimension inferred ==="<<endl;
        dimEngine.printAllDimensions(ast);
        cout << endl;
    }

    auto IR = convertASTtoIR(ast);

    if (debugMode) {
      cout << "=== IR Structure ===" << endl;
      printIR(IR);
      cout << endl;
    }

    cout << "=== Generated Code ===" << endl;
    generateCode(IR);
    cout << endl;

  } catch (const exception &e) {
    cout << "Error: " << e.what() << endl;
    return 1;
  }

  return 0;
}
