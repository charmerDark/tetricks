/*
Program that takes in AST and builds preliminary loops
*/
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;

enum BinOpKind { Add, Subtract, Multiply, Divide };

struct expr {
  /*
  AST nodes representation
  */
  virtual ~expr() = default;
};

struct Tensor : expr {
  string name;
  Tensor(const string &n) : name(n) {}
};

struct EinsumNode : expr {
  /*
  indexes; list of einsum indices: Eg: einsum("ij,jk,kl -> il" , A,B,C) would
  return an EinsumNode object with indices string "ij","jk","ik","il" inputs:
  list of tensor objects to be operated on.
  */
  vector<string> indexes;
  vector<Tensor> inputs;
  EinsumNode(const vector<string> &idx,
             const std::vector<Tensor> &inp_tensors) {
    if (idx.size() == inp_tensors.size() + 1) {
      indexes = idx;
      inputs = inp_tensors;
    } else {
      throw std::invalid_argument(
          "Input indices don't match the number of inputs provided.");
    }
  }
};

struct BinOpNode : expr {
  BinOpKind opn;
  shared_ptr<expr> inp1;
  shared_ptr<expr> inp2;
  BinOpNode(BinOpKind operation, shared_ptr<expr> input1,
            shared_ptr<expr> input2)
      : opn(operation), inp1(input1), inp2(input2) {}
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
  EisnumNode("ij","ji",A) would have LoopNode(i, A[0], LoopNode(j,A[1],
  Calcnode(temp1[j][i] = A[i][j]) )) BinOpNode(Multiply, A,B) would have
  LoopNode()
  */
  char loop_variable;
  shape_variable shape_var;
  shared_ptr<IRNode> body;
  LoopNode(char var, const shape_variable &shape, shared_ptr<IRNode> b)
      : loop_variable(var), shape_var(shape), body(b) {}
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
  for (char c : outputIndex) {
    ss << "[" << c << "]";
  }

  ss << " += ";

  // Generate right side (multiplication of all input tensors)
  for (size_t i = 0; i < einsum->inputs.size(); i++) {
    if (i > 0)
      ss << " * ";
    ss << generateTensorAccess(einsum->inputs[i].name, einsum->indexes[i], {});
  }

  ss << ";";
  return ss.str();
}

// Helper function to convert einsum to nested loops
shared_ptr<IRNode> convertEinsumToLoops(const EinsumNode *einsum,
                                        const string &tempName) {
  set<char> loopVars = getLoopVariables(einsum->indexes);
  vector<char> orderedVars(loopVars.begin(), loopVars.end());

  if (orderedVars.empty()) {
    // No loops needed, just direct assignment
    return make_shared<CalcNode>(generateEinsumCalc(einsum, tempName));
  }

  // Generate innermost calculation
  shared_ptr<IRNode> innermost =
      make_shared<CalcNode>(generateEinsumCalc(einsum, tempName));

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
        boundsTensor = make_shared<Tensor>(einsum->inputs[j].name);
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
          "Binary operations between two tensors not yet implemented");
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

int main() {
  cout << "=== Testing n-tensor einsum expressions ===" << endl << endl;

  // Test 2-tensor einsum
  cout << "1. Two-tensor einsum: einsum(\"ij,jk -> ik\",A,B)" << endl;
  Tensor A("A");
  Tensor B("B");
  Tensor C("C");
  std::vector<Tensor> tensors2 = {A, B};
  std::vector<string> indexes2 = {"ij", "jk", "ik"};
  auto einsumNode2 = std::make_shared<EinsumNode>(indexes2, tensors2);
  auto IR2 = convertASTtoIR(einsumNode2);

  cout << "IR Structure:" << endl;
  printIR(IR2);
  cout << "\nGenerated Code:" << endl;
  generateCode(IR2);
  cout << endl;

  // Test 3-tensor einsum
  cout << "2. Three-tensor einsum: eisnum(\"ij,jk,kl->il\")" << endl;
  Tensor D("D");
  std::vector<Tensor> tensors3 = {A, B, D};
  std::vector<string> indexes3 = {"ij", "jk", "kl", "il"};
  auto einsumNode3 = std::make_shared<EinsumNode>(indexes3, tensors3);
  auto IR3 = convertASTtoIR(einsumNode3);

  cout << "IR Structure:" << endl;
  printIR(IR3);
  cout << "\nGenerated Code:" << endl;
  generateCode(IR3);
  cout << endl;

  // Test 4-tensor einsum:
  cout << "3. Four-tensor einsum: eisnum(\"ij,jk,kl,lm->im\",A,B,D,E)" << endl;
  Tensor E("E");
  std::vector<Tensor> tensors4 = {A, B, D, E};
  std::vector<string> indexes4 = {"ij", "jk", "kl", "lm", "im"};
  auto einsumNode4 = std::make_shared<EinsumNode>(indexes4, tensors4);
  auto IR4 = convertASTtoIR(einsumNode4);

  cout << "IR Structure:" << endl;
  printIR(IR4);
  cout << "\nGenerated Code:" << endl;
  generateCode(IR4);
  cout << endl;

  // Test case with BinOp and one einsum
  cout << "4. einsum(\"ij,jk,kl->il\",A,B,D) + C" << endl;
  auto prog_AST = std::make_shared<BinOpNode>(Add, einsumNode3,
                                              std::make_shared<Tensor>("C"));
  auto IR_complex = convertASTtoIR(prog_AST);

  cout << "IR Structure:" << endl;
  printIR(IR_complex);
  cout << "\nGenerated Code:" << endl;
  generateCode(IR_complex);
  cout << endl;

  // Test matrix-vector multiplication:
  cout << "5. Matrix-vector multiplication: eisnum(\"ij,j -> i\",A,v)" << endl;
  Tensor v("v");
  std::vector<Tensor> tensors_mv = {A, v};
  std::vector<string> indexes_mv = {"ij", "j", "i"};
  auto einsumNode_mv = std::make_shared<EinsumNode>(indexes_mv, tensors_mv);
  auto IR_mv = convertASTtoIR(einsumNode_mv);

  cout << "IR Structure:" << endl;
  printIR(IR_mv);
  cout << "\nGenerated Code:" << endl;
  generateCode(IR_mv);
  cout << endl;

  return 0;
}