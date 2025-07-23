#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"

#include <iostream>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>
#include <queue>
#include <iomanip>

using namespace llvm;

// =============================================================================
// DATA FLOW GRAPH (DFG) DATA STRUCTURES
// =============================================================================

/**
 * Represents a single node in the Data Flow Graph
 * Each node corresponds to an LLVM instruction that we want to map to the CGRA
 */
struct DFGNode {
    int id;
    std::string label;  // String representation of the LLVM instruction
    std::string opcode; // Operation type (add, mul, phi, etc.)
};

/**
 * Represents an edge in the Data Flow Graph
 * Captures data dependencies between instructions
 */
struct DFGEdge {
    int src;  // Source instruction ID
    int dst;  // Destination instruction ID
    bool isLoopCarried = false;  // True for dependencies that span loop iterations
};

/**
 * Complete Data Flow Graph representation
 * Contains all computational nodes and their data dependencies
 */
struct DFG {
    std::vector<DFGNode> nodes;
    std::vector<DFGEdge> edges;
    std::unordered_map<const llvm::Instruction*, int> instToId;
    
    // Helper to find all predecessors of a node
    std::vector<int> getPredecessors(int nodeId) const {
        std::vector<int> preds;
        for (const auto& edge : edges) {
            if (edge.dst == nodeId) {
                preds.push_back(edge.src);
            }
        }
        return preds;
    }
    
    // Helper to find all successors of a node
    std::vector<int> getSuccessors(int nodeId) const {
        std::vector<int> succs;
        for (const auto& edge : edges) {
            if (edge.src == nodeId) {
                succs.push_back(edge.dst);
            }
        }
        return succs;
    }
};

// =============================================================================
// CGRA ARCHITECTURE DATA STRUCTURES
// =============================================================================

/**
 * Represents a single processing element (PE) in the CGRA
 * Design Decision: Homogeneous architecture - all PEs are identical ALUs
 * Rationale: Simplifies initial mapping algorithm, suitable for tensor operations
 */
struct CGRANode {
    int row, col;  // Position in the 2D grid
    int id;        // Unique identifier
    std::set<std::string> supportedOps;  // Operations this PE can perform
    
    CGRANode(int r, int c, int identifier) : row(r), col(c), id(identifier) {
        // All nodes support basic arithmetic operations needed for tensor ops
        supportedOps = {"add", "sub", "mul", "fadd", "fsub", "fmul", "phi"};
    }
    
    bool canExecute(const std::string& opcode) const {
        return supportedOps.count(opcode) > 0;
    }
};

/**
 * Represents the complete CGRA architecture
 * Design Decision: 2D mesh with nearest-neighbor connectivity. TODO: make this variable
 */
struct CGRAArchitecture {
    int rows, cols;
    std::vector<std::vector<CGRANode>> grid;  // 2D grid of processing elements
    
    CGRAArchitecture(int r, int c) : rows(r), cols(c) {
        grid.resize(rows);
        int nodeId = 0;
        for (int i = 0; i < rows; ++i) {
            grid[i].reserve(cols);
            for (int j = 0; j < cols; ++j) {
                grid[i].emplace_back(i, j, nodeId++);
            }
        }
    }
    
    // Get valid neighbors for routing (4-connected mesh)
    std::vector<std::pair<int,int>> getNeighbors(int row, int col) const {
        std::vector<std::pair<int,int>> neighbors;
        // Up, Down, Left, Right
        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};
        
        for (int i = 0; i < 4; ++i) {
            int nr = row + dr[i];
            int nc = col + dc[i];
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                neighbors.emplace_back(nr, nc);
            }
        }
        return neighbors;
    }
    
    bool isValidPosition(int row, int col) const {
        return row >= 0 && row < rows && col >= 0 && col < cols;
    }
    
    CGRANode& getNode(int row, int col) {
        return grid[row][col];
    }
    
    const CGRANode& getNode(int row, int col) const {
        return grid[row][col];
    }
};

// =============================================================================
// MAPPING DATA STRUCTURES
// =============================================================================

/**
 * Represents a routing path between two CGRA nodes
 * Design Decision: Explicit path storage with delay calculation
 * Rationale: Allows precise timing analysis and routing conflict detection
 */
struct Route {
    std::vector<std::pair<int,int>> path;  // Sequence of (row,col) coordinates
    
    Route() = default;
    Route(const std::vector<std::pair<int,int>>& p) : path(p) {}
    
    // Calculate routing delay (number of hops)
    int delay() const { 
        return std::max(0, static_cast<int>(path.size()) - 1); 
    }
    
    bool isEmpty() const { return path.empty(); }
    
    std::pair<int,int> source() const { 
        return path.empty() ? std::make_pair(-1,-1) : path.front(); 
    }
    
    std::pair<int,int> destination() const { 
        return path.empty() ? std::make_pair(-1,-1) : path.back(); 
    }
};

struct PairHash {
    std::size_t operator()(const std::pair<int,int>& p) const {
        return std::hash<int>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
    }
};

/**
 * Complete mapping result containing spatial placement and temporal scheduling
 * Design Decision: Separate placement and scheduling for modularity
 * Rationale: Allows independent optimization of space and time dimensions
 */
struct MappingResult {
    // Spatial placement: which CGRA node each DFG operation is mapped to
    std::unordered_map<int, std::pair<int,int>> placement; // inst_id -> (row, col)
    
    // Temporal scheduling: when each operation executes
    std::unordered_map<int, int> schedule; // inst_id -> time_step
    
    // Routing information for data movement between non-adjacent nodes
    std::unordered_map<std::pair<int,int>, Route, PairHash> routing; // (src_inst, dst_inst) -> route
    
    // Derived information for analysis
    std::map<int, std::vector<int>> timeToInstructions; // time_step -> [inst_ids]
    int maxTimeStep = 0;
    
    // Custom hash function for instruction pairs
    struct PairHash {
        std::size_t operator()(const std::pair<int,int>& p) const {
            return std::hash<int>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
        }
    };
    
    // Add a mapping entry
    void addMapping(int instId, int row, int col, int timeStep) {
        placement[instId] = std::make_pair(row, col);
        schedule[instId] = timeStep;
        timeToInstructions[timeStep].push_back(instId);
        maxTimeStep = std::max(maxTimeStep, timeStep);
    }
    
    // Add routing information
    void addRoute(int srcInst, int dstInst, const Route& route) {
        routing[std::make_pair(srcInst, dstInst)] = route;
    }
    
    // Check if mapping satisfies all constraints
    bool isValidMapping(const DFG& dfg, const CGRAArchitecture& cgra) const;
    
    // Get the route between two instructions
    Route getRoute(int srcInst, int dstInst) const {
        auto key = std::make_pair(srcInst, dstInst);
        auto it = routing.find(key);
        return (it != routing.end()) ? it->second : Route();
    }
    
    void clear() {
        placement.clear();
        schedule.clear();
        routing.clear();
        timeToInstructions.clear();
        maxTimeStep = 0;
    }
};

// =============================================================================
// ROUTING ALGORITHMS
// =============================================================================

/**
 * Find shortest path between two CGRA nodes using BFS
 * Design Decision: Simple BFS for shortest path
 * Rationale: Minimizes routing delay, suitable for mesh topology
 */
Route findRoute(std::pair<int,int> src, std::pair<int,int> dst, 
                const CGRAArchitecture& cgra) {
    if (src == dst) {
        return Route({src});  // Same node, no routing needed
    }
    
    std::queue<std::pair<int,int>> queue;
    std::unordered_map<std::pair<int,int>, std::pair<int,int>, 
                      PairHash> parent;
    std::set<std::pair<int,int>> visited;
    
    queue.push(src);
    visited.insert(src);
    parent[src] = std::make_pair(-1, -1);  // Root has no parent
    
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop();
        
        if (current == dst) {
            // Reconstruct path
            std::vector<std::pair<int,int>> path;
            auto node = dst;
            while (node != std::make_pair(-1, -1)) {
                path.push_back(node);
                node = parent[node];
            }
            std::reverse(path.begin(), path.end());
            return Route(path);
        }
        
        // Explore neighbors
        for (const auto& neighbor : cgra.getNeighbors(current.first, current.second)) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                parent[neighbor] = current;
                queue.push(neighbor);
            }
        }
    }
    
    // No path found (shouldn't happen in connected mesh)
    return Route();
}

/**
 * Calculate Manhattan distance between two positions
 * Used for heuristic routing decisions
 */
int manhattanDistance(std::pair<int,int> a, std::pair<int,int> b) {
    return std::abs(a.first - b.first) + std::abs(a.second - b.second);
}

// =============================================================================
// VALIDATION FUNCTIONS
// =============================================================================

/**
 * Validate that the mapping satisfies all CGRA constraints
 * Design Decision: Comprehensive constraint checking
 * Rationale: Early detection of invalid mappings prevents runtime errors
 */
bool MappingResult::isValidMapping(const DFG& dfg, const CGRAArchitecture& cgra) const {
    // Check 1: Resource constraint - no two operations on same node at same time
    std::map<std::pair<std::pair<int,int>, int>, int> resourceUsage; // ((row,col), time) -> count
    
    for (const auto& [instId, pos] : placement) {
        auto timeIt = schedule.find(instId);
        if (timeIt == schedule.end()) {
            std::cerr << "ERROR: Instruction " << instId << " has placement but no schedule\n";
            return false;
        }
        
        auto key = std::make_pair(pos, timeIt->second);
        resourceUsage[key]++;
        
        if (resourceUsage[key] > 1) {
            std::cerr << "ERROR: Resource conflict at (" << pos.first << "," << pos.second 
                     << ") at time " << timeIt->second << std::endl;
            return false;
        }
        
        // Check if position is valid
        if (!cgra.isValidPosition(pos.first, pos.second)) {
            std::cerr << "ERROR: Invalid position (" << pos.first << "," << pos.second 
                     << ") for instruction " << instId << std::endl;
            return false;
        }
    }
    
    // Check 2: Dependency constraint - schedule[dst] >= schedule[src] + routing_delay + 1
    for (const auto& edge : dfg.edges) {
        auto srcSchedIt = schedule.find(edge.src);
        auto dstSchedIt = schedule.find(edge.dst);
        
        if (srcSchedIt == schedule.end() || dstSchedIt == schedule.end()) {
            continue; // Skip unmapped instructions
        }
        
        int srcTime = srcSchedIt->second;
        int dstTime = dstSchedIt->second;
        
        // Get routing delay
        Route route = getRoute(edge.src, edge.dst);
        int routingDelay = route.delay();
        
        int requiredTime = srcTime + routingDelay + 1;
        if (dstTime < requiredTime) {
            std::cerr << "ERROR: Timing violation - instruction " << edge.dst 
                     << " scheduled at " << dstTime << " but requires time >= " << requiredTime
                     << " (dependency from " << edge.src << " with routing delay " << routingDelay << ")"
                     << std::endl;
            return false;
        }
    }
    
    return true;
}

// =============================================================================
// DFG GENERATION (Enhanced from original)
// =============================================================================

// Helper: Find the innermost loop in a function
Loop* findInnermostLoop(LoopInfo &LI) {
    for (auto *L : LI) {
        std::vector<Loop*> stack;
        stack.push_back(L);
        while (!stack.empty()) {
            Loop *cur = stack.back();
            stack.pop_back();
            if (cur->getSubLoops().empty()) {
                return cur;
            }
            for (auto *sub : cur->getSubLoops())
                stack.push_back(sub);
        }
    }
    return nullptr;
}

// Helper: Extract opcode from LLVM instruction
std::string getOpcodeName(const Instruction* I) {
    if (isa<PHINode>(I)) return "phi";
    if (const BinaryOperator* binOp = dyn_cast<BinaryOperator>(I)) {
        return binOp->getOpcodeName();
    }
    if (const CallInst* call = dyn_cast<CallInst>(I)) {
        if (const Function* calledFunc = call->getCalledFunction()) {
            if (calledFunc->getIntrinsicID() == Intrinsic::fmuladd) {
                return "fmuladd";
            }
        }
    }
    return "unknown";
}

// Helper: Is this instruction relevant for CGRA mapping?
bool isArithmeticOrPHI(const Instruction* I) {
    if (isa<PHINode>(I)) return true;
    if (isa<BinaryOperator>(I)) return true;
    
    // Add support for LLVM intrinsics
    if (const CallInst* call = dyn_cast<CallInst>(I)) {
        if (const Function* calledFunc = call->getCalledFunction()) {
            if (calledFunc->isIntrinsic()) {
                // Accept common arithmetic intrinsics
                switch (calledFunc->getIntrinsicID()) {
                    case Intrinsic::fmuladd:
                    case Intrinsic::fma:
                    case Intrinsic::sqrt:
                    case Intrinsic::sin:
                    case Intrinsic::cos:
                        return true;
                    default:
                        break;
                }
            }
        }
    }
    return false;
}

/**
 * Generate DFG from LLVM loop, enhanced with opcode information
 */
DFG generateDFGForArithmeticAndPHI(llvm::Loop *loop) {
    DFG dfg;
    int id = 0;

    // 1. Collect nodes (instructions)
    std::set<const llvm::Instruction*> loopInsts;
    for (auto *BB : loop->blocks()) {
        for (const llvm::Instruction &I : *BB) {
            if (isArithmeticOrPHI(&I)) {
                loopInsts.insert(&I);
            }
        }
    }
    
    for (const llvm::Instruction* I : loopInsts) {
        std::string instrStr;
        llvm::raw_string_ostream rso(instrStr);
        I->print(rso);
        
        DFGNode node;
        node.id = id;
        node.label = rso.str();
        node.opcode = getOpcodeName(I);
        
        dfg.nodes.push_back(node);
        dfg.instToId[I] = id++;
    }

    // 2. Collect edges (data dependencies)
    for (const llvm::Instruction* I : loopInsts) {
        int srcID = dfg.instToId[I];

        // PHI node incoming values (loop-carried dependencies)
        if (const llvm::PHINode *phi = llvm::dyn_cast<llvm::PHINode>(I)) {
            for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
                llvm::Value *incoming = phi->getIncomingValue(i);
                if (const llvm::Instruction *inInst = llvm::dyn_cast<llvm::Instruction>(incoming)) {
                    if (loopInsts.count(inInst)) {
                        int inID = dfg.instToId[inInst];
                        dfg.edges.push_back({inID, srcID, true}); // Loop-carried
                    }
                }
            }
        }

        // Data dependencies to users
        for (const llvm::User *U : I->users()) {
            if (const llvm::Instruction *userInst = llvm::dyn_cast<llvm::Instruction>(U)) {
                if (loopInsts.count(userInst)) {
                    int dstID = dfg.instToId[userInst];
                    dfg.edges.push_back({srcID, dstID, false}); // Regular dependency
                }
            }
        }
    }

    return dfg;
}

// =============================================================================
// PRINTING AND VISUALIZATION FUNCTIONS
// =============================================================================

void printDFGasDot(const DFG& dfg) {
    std::cout << "digraph DFG {\n";
    std::cout << "  node [shape=box, fontname=\"monospace\"];\n";
    for (const auto& node : dfg.nodes) {
        std::string label = node.label;
        size_t pos = 0;
        while ((pos = label.find("\"", pos)) != std::string::npos) {
            label.replace(pos, 1, "\\\"");
            pos += 2;
        }
        std::cout << "  " << node.id << " [label=\"ID:" << node.id << "\\n" 
                  << "Op:" << node.opcode << "\\n" << label << "\"];\n";
    }
    for (const auto& edge : dfg.edges) {
        std::cout << "  " << edge.src << " -> " << edge.dst;
        if (edge.isLoopCarried) std::cout << " [color=blue, label=\"LC\"]";
        std::cout << ";\n";
    }
    std::cout << "}\n";
}

void printCGRAArchitecture(const CGRAArchitecture& cgra) {
    std::cout << "\n=== CGRA Architecture ===\n";
    std::cout << "Dimensions: " << cgra.rows << "x" << cgra.cols << " (rows x cols)\n";
    std::cout << "Total Processing Elements: " << (cgra.rows * cgra.cols) << "\n";
    std::cout << "Connectivity: Nearest-neighbor mesh (4-connected)\n";
    std::cout << "Node capabilities: Homogeneous ALUs supporting {add, sub, mul, fadd, fsub, fmul, phi}\n\n";
    
    std::cout << "Grid layout:\n";
    for (int i = 0; i < cgra.rows; ++i) {
        for (int j = 0; j < cgra.cols; ++j) {
            std::cout << std::setw(4) << "(" << i << "," << j << ")";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void printMappingResult(const MappingResult& result, const DFG& dfg, const CGRAArchitecture& cgra) {
    std::cout << "\n=== Mapping Result ===\n";
    
    if (result.placement.empty()) {
        std::cout << "No mapping performed yet.\n\n";
        return;
    }
    
    std::cout << "Total instructions mapped: " << result.placement.size() << "\n";
    std::cout << "Schedule length: " << (result.maxTimeStep + 1) << " time steps\n\n";
    
    // Print instruction mappings
    std::cout << "Instruction Mappings:\n";
    std::cout << std::setw(6) << "ID" << std::setw(12) << "Operation" 
              << std::setw(12) << "Position" << std::setw(8) << "Time" << "\n";
    std::cout << std::string(38, '-') << "\n";
    
    for (const auto& [instId, pos] : result.placement) {
        auto schedIt = result.schedule.find(instId);
        int timeStep = (schedIt != result.schedule.end()) ? schedIt->second : -1;
        
        std::string opcode = "unknown";
        for (const auto& node : dfg.nodes) {
            if (node.id == instId) {
                opcode = node.opcode;
                break;
            }
        }
        
        std::cout << std::setw(6) << instId 
                  << std::setw(12) << opcode
                  << std::setw(6) << "(" << pos.first << "," << pos.second << ")"
                  << std::setw(8) << timeStep << "\n";
    }
    
    // Print temporal schedule
    std::cout << "\nTemporal Schedule:\n";
    for (const auto& [time, instructions] : result.timeToInstructions) {
        std::cout << "t=" << time << ": ";
        for (size_t i = 0; i < instructions.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "inst" << instructions[i];
        }
        std::cout << "\n";
    }
    
    // Print routing information
    if (!result.routing.empty()) {
        std::cout << "\nRouting Information:\n";
        std::cout << std::setw(12) << "Edge" << std::setw(8) << "Delay" << std::setw(20) << "Path" << "\n";
        std::cout << std::string(40, '-') << "\n";
        
        for (const auto& [edge, route] : result.routing) {
            std::cout << std::setw(6) << edge.first << "->" << std::setw(4) << edge.second
                      << std::setw(8) << route.delay();
            
            std::cout << "    ";
            for (size_t i = 0; i < route.path.size(); ++i) {
                if (i > 0) std::cout << " -> ";
                std::cout << "(" << route.path[i].first << "," << route.path[i].second << ")";
            }
            std::cout << "\n";
        }
    }
    
    // Validation check
    std::cout << "\nMapping Validation: " 
              << (result.isValidMapping(dfg, cgra) ? "VALID" : "INVALID") << "\n\n";
}

void printMappingVisualization(const MappingResult& result, const CGRAArchitecture& cgra, int timeStep = -1) {
    std::cout << "\n=== CGRA Mapping Visualization";
    if (timeStep >= 0) {
        std::cout << " (Time Step " << timeStep << ")";
    }
    std::cout << " ===\n\n";
    
    // Create a grid showing which instructions are mapped where
    std::vector<std::vector<std::string>> grid(cgra.rows, std::vector<std::string>(cgra.cols, "."));
    
    for (const auto& [instId, pos] : result.placement) {
        if (timeStep >= 0) {
            // Show only instructions active at this time step
            auto schedIt = result.schedule.find(instId);
            if (schedIt != result.schedule.end() && schedIt->second == timeStep) {
                grid[pos.first][pos.second] = std::to_string(instId);
            }
        } else {
            // Show all mapped instructions
            grid[pos.first][pos.second] = std::to_string(instId);
        }
    }
    
    // Print the grid
    for (int i = 0; i < cgra.rows; ++i) {
        for (int j = 0; j < cgra.cols; ++j) {
            std::cout << std::setw(4) << grid[i][j];
        }
        std::cout << "\n";
    }
    
    std::cout << "\nLegend: Numbers = Instruction IDs, '.' = Unused PE\n\n";
}

// =============================================================================
// MAPPING ALGORITHM (PLACEHOLDER)
// =============================================================================

/**
 * Main mapping function - maps a DFG to a CGRA architecture
 * 
 * This function will implement the core mapping algorithm that:
 * 1. Performs spatial placement of operations onto CGRA nodes
 * 2. Schedules operations in time respecting dependencies
 * 3. Handles routing between non-adjacent nodes
 * 4. Optimizes for metrics like throughput, latency, or energy
 * 
 * @param dfg The data flow graph to map
 * @param cgra The target CGRA architecture
 * @param result Output parameter containing the mapping solution
 * @return True if mapping successful, false otherwise
 * 
 * Design Decisions for Future Implementation:
 * - Consider list scheduling with priority functions
 * - Use simulated annealing or genetic algorithms for placement optimization
 * - Implement resource-aware scheduling to avoid conflicts
 * - Add support for different optimization objectives
 */
 bool mapDFGToCGRA(const DFG& dfg, const CGRAArchitecture& cgra, MappingResult& result) {
    std::cout << "\n=== Starting DFG to CGRA Mapping ===\n";
    std::cout << "DFG: " << dfg.nodes.size() << " nodes, " << dfg.edges.size() << " edges\n";
    std::cout << "CGRA: " << cgra.rows << "x" << cgra.cols << " grid\n";
    result.clear();

    // 1. Compute in-degree for each node (for scheduling)
    std::unordered_map<int, int> inDegree;
    for (const auto& node : dfg.nodes) inDegree[node.id] = 0;
    for (const auto& edge : dfg.edges) inDegree[edge.dst]++;

    // 2. List of ready nodes (in-degree 0)
    std::queue<int> ready;
    for (const auto& [id, deg] : inDegree)
        if (deg == 0) ready.push(id);

    // 3. Keep track of PE usage: (row,col,time) -> instId
    std::map<std::tuple<int,int,int>, int> peUsage;

    // 4. For each node, store when it is ready
    std::unordered_map<int, int> nodeReadyTime;

    // 5. Map from DFG node id to (row,col)
    std::unordered_map<int, std::pair<int,int>> nodePlacement;

    // 6. For placement, keep a list of available PEs
    std::vector<std::pair<int,int>> availablePEs;
    for (int i = 0; i < cgra.rows; ++i)
        for (int j = 0; j < cgra.cols; ++j)
            availablePEs.emplace_back(i, j);

    // 7. Main scheduling loop
    int scheduledCount = 0;
    while (!ready.empty()) {
        int nodeId = ready.front(); ready.pop();
        const DFGNode& node = dfg.nodes[nodeId];

        // Find earliest time when all dependencies are satisfied
        int earliest = 0;
        for (int pred : dfg.getPredecessors(nodeId)) {
            // Routing delay from pred to this node
            auto predPos = nodePlacement[pred];
            auto thisPos = availablePEs[scheduledCount % availablePEs.size()];
            Route route = findRoute(predPos, thisPos, cgra);
            int predReady = nodeReadyTime[pred] + route.delay() + 1;
            earliest = std::max(earliest, predReady);
        }

        // Find a PE that is free at 'earliest' and supports the opcode
        bool placed = false;
        for (size_t idx = 0; idx < availablePEs.size(); ++idx) {
            auto [row, col] = availablePEs[idx];
            auto& pe = cgra.grid[row][col];
            if (!pe.canExecute(node.opcode)) continue;
            if (peUsage.count({row, col, earliest}) == 0) {
                // Place here
                result.addMapping(nodeId, row, col, earliest);
                peUsage[{row, col, earliest}] = nodeId;
                nodePlacement[nodeId] = {row, col};
                nodeReadyTime[nodeId] = earliest;
                placed = true;
                break;
            }
        }
        if (!placed) {
            // If no PE is free at 'earliest', try later time steps
            int t = earliest + 1;
            bool found = false;
            while (!found && t < earliest + 100) { // avoid infinite loop
                for (size_t idx = 0; idx < availablePEs.size(); ++idx) {
                    auto [row, col] = availablePEs[idx];
                    auto& pe = cgra.grid[row][col];
                    if (!pe.canExecute(node.opcode)) continue;
                    if (peUsage.count({row, col, t}) == 0) {
                        result.addMapping(nodeId, row, col, t);
                        peUsage[{row, col, t}] = nodeId;
                        nodePlacement[nodeId] = {row, col};
                        nodeReadyTime[nodeId] = t;
                        found = true;
                        break;
                    }
                }
                if (found) break;
                ++t;
            }
            if (!found) {
                std::cerr << "ERROR: Could not place node " << nodeId << " due to resource constraints.\n";
                return false;
            }
        }

        // Routing: for each predecessor, store the route
        for (int pred : dfg.getPredecessors(nodeId)) {
            auto srcPos = nodePlacement[pred];
            auto dstPos = nodePlacement[nodeId];
            Route route = findRoute(srcPos, dstPos, cgra);
            result.addRoute(pred, nodeId, route);
        }

        // Update successors' in-degree
        for (int succ : dfg.getSuccessors(nodeId)) {
            inDegree[succ]--;
            if (inDegree[succ] == 0) ready.push(succ);
        }
        scheduledCount++;
    }

    // Check if all nodes were scheduled
    if (scheduledCount != dfg.nodes.size()) {
        std::cerr << "ERROR: Not all nodes were scheduled (" << scheduledCount << "/" << dfg.nodes.size() << ")\n";
        return false;
    }

    // Validate mapping
    if (!result.isValidMapping(dfg, cgra)) {
        std::cerr << "ERROR: Mapping is invalid.\n";
        return false;
    }

    std::cout << "INFO: Mapping completed successfully.\n";
    return true;
}
// bool mapDFGToCGRA(const DFG& dfg, const CGRAArchitecture& cgra, MappingResult& result) {
//     std::cout << "\n=== Starting DFG to CGRA Mapping (Two-Phase Approach) ===\n";
//     std::cout << "DFG: " << dfg.nodes.size() << " nodes, " << dfg.edges.size() << " edges\n";
//     std::cout << "CGRA: " << cgra.rows << "x" << cgra.cols << " grid\n";
    
//     result.clear();
    
//     // Check if CGRA has enough resources
//     if (dfg.nodes.empty()) {
//         std::cout << "INFO: Empty DFG, nothing to map.\n";
//         return true;
//     }
    
//     // Create node ID to index mapping for easier access
//     std::unordered_map<int, size_t> nodeIdToIndex;
//     for (size_t i = 0; i < dfg.nodes.size(); ++i) {
//         nodeIdToIndex[dfg.nodes[i].id] = i;
//     }
    
//     // =================================================================
//     // PHASE 1: TEMPORAL SCHEDULING
//     // =================================================================
//     std::cout << "\n--- Phase 1: Temporal Scheduling ---\n";
    
//     // Compute in-degree for topological sorting
//     std::unordered_map<int, int> inDegree;
//     for (const auto& node : dfg.nodes) {
//         inDegree[node.id] = 0;
//     }
//     for (const auto& edge : dfg.edges) {
//         inDegree[edge.dst]++;
//     }
    
//     // Initialize ready queue with nodes that have no dependencies
//     std::queue<int> ready;
//     for (const auto& [nodeId, degree] : inDegree) {
//         if (degree == 0) {
//             ready.push(nodeId);
//         }
//     }
    
//     // Schedule each node at the earliest possible time
//     std::unordered_map<int, int> nodeSchedule; // nodeId -> time step
//     std::unordered_map<int, int> nodeFinishTime; // nodeId -> finish time
//     const int CONSERVATIVE_ROUTING_DELAY = 1; // Conservative estimate for any routing
//     const int EXECUTION_TIME = 1; // Assume all operations take 1 cycle
    
//     int scheduledNodes = 0;
//     while (!ready.empty()) {
//         int nodeId = ready.front();
//         ready.pop();
        
//         // Find earliest start time based on dependencies
//         int earliestStart = 0;
//         for (const auto& edge : dfg.edges) {
//             if (edge.dst == nodeId) {
//                 // This node depends on edge.src
//                 auto finishIt = nodeFinishTime.find(edge.src);
//                 if (finishIt != nodeFinishTime.end()) {
//                     int depFinishTime = finishIt->second;
//                     int routingDelay = edge.isLoopCarried ? 0 : CONSERVATIVE_ROUTING_DELAY;
//                     earliestStart = std::max(earliestStart, depFinishTime + routingDelay);
//                 }
//             }
//         }
        
//         // Schedule this node
//         nodeSchedule[nodeId] = earliestStart;
//         nodeFinishTime[nodeId] = earliestStart + EXECUTION_TIME;
//         scheduledNodes++;
        
//         // Update successors
//         for (const auto& edge : dfg.edges) {
//             if (edge.src == nodeId) {
//                 inDegree[edge.dst]--;
//                 if (inDegree[edge.dst] == 0) {
//                     ready.push(edge.dst);
//                 }
//             }
//         }
//     }
    
//     if (scheduledNodes != dfg.nodes.size()) {
//         std::cerr << "ERROR: Phase 1 failed - could not schedule all nodes (cyclic dependencies?)\n";
//         std::cerr << "       Scheduled: " << scheduledNodes << "/" << dfg.nodes.size() << " nodes\n";
//         return false;
//     }
    
//     std::cout << "Phase 1 complete: Scheduled " << scheduledNodes << " nodes\n";
    
//     // =================================================================
//     // PHASE 2: SPATIAL PLACEMENT
//     // =================================================================
//     std::cout << "\n--- Phase 2: Spatial Placement ---\n";
    
//     // Group nodes by their scheduled time
//     std::map<int, std::vector<int>> timeToNodes;
//     for (const auto& [nodeId, time] : nodeSchedule) {
//         timeToNodes[time].push_back(nodeId);
//     }
    
//     // Track PE usage: (row, col, time) -> nodeId
//     std::map<std::tuple<int, int, int>, int> peUsage;
//     std::unordered_map<int, std::pair<int, int>> nodePlacement; // nodeId -> (row, col)
    
//     // Process each time step in order
//     for (const auto& [timeStep, nodesAtTime] : timeToNodes) {
//         std::cout << "Placing " << nodesAtTime.size() << " nodes at time " << timeStep << std::endl;
        
//         for (int nodeId : nodesAtTime) {
//             const DFGNode& node = dfg.nodes[nodeIdToIndex[nodeId]];
            
//             // Find the best position for this node
//             std::pair<int, int> bestPosition{-1, -1};
//             int bestCost = INT_MAX;
            
//             // Try all available PEs
//             for (int row = 0; row < cgra.rows; ++row) {
//                 for (int col = 0; col < cgra.cols; ++col) {
//                     const CGRANode& pe = cgra.getNode(row, col);
                    
//                     // Check if PE can execute this operation
//                     if (!pe.canExecute(node.opcode)) {
//                         continue;
//                     }
                    
//                     // Check if PE is free at this time
//                     if (peUsage.count({row, col, timeStep}) > 0) {
//                         continue;
//                     }
                    
//                     // Calculate placement cost (routing cost to predecessors)
//                     int placementCost = 0;
//                     for (const auto& edge : dfg.edges) {
//                         if (edge.dst == nodeId) {
//                             auto predPosIt = nodePlacement.find(edge.src);
//                             if (predPosIt != nodePlacement.end()) {
//                                 auto predPos = predPosIt->second;
//                                 int routingCost = manhattanDistance(predPos, {row, col});
//                                 placementCost += routingCost;
//                             }
//                         }
//                     }
                    
//                     // Select position with minimum cost
//                     if (placementCost < bestCost) {
//                         bestCost = placementCost;
//                         bestPosition = {row, col};
//                     }
//                 }
//             }
            
//             // Check if we found a valid position
//             if (bestPosition.first == -1) {
//                 std::cerr << "ERROR: Phase 2 failed - cannot place node " << nodeId 
//                          << " (opcode: " << node.opcode << ") at time " << timeStep << std::endl;
//                 std::cerr << "       No compatible or available PE found\n";
//                 return false;
//             }
            
//             // Place the node
//             peUsage[{bestPosition.first, bestPosition.second, timeStep}] = nodeId;
//             nodePlacement[nodeId] = bestPosition;
//             result.addMapping(nodeId, bestPosition.first, bestPosition.second, timeStep);
            
//             std::cout << "  Node " << nodeId << " (" << node.opcode << ") -> (" 
//                      << bestPosition.first << "," << bestPosition.second << ") cost=" << bestCost << std::endl;
//         }
//     }
    
//     std::cout << "Phase 2 complete: Placed " << nodePlacement.size() << " nodes\n";
    
//     // =================================================================
//     // PHASE 3: ROUTING AND VALIDATION
//     // =================================================================
//     std::cout << "\n--- Phase 3: Routing and Validation ---\n";
    
//     // Calculate actual routes and check timing constraints
//     bool needRescheduling = false;
//     for (const auto& edge : dfg.edges) {
//         auto srcPosIt = nodePlacement.find(edge.src);
//         auto dstPosIt = nodePlacement.find(edge.dst);
        
//         if (srcPosIt != nodePlacement.end() && dstPosIt != nodePlacement.end()) {
//             auto srcPos = srcPosIt->second;
//             auto dstPos = dstPosIt->second;
            
//             // Calculate actual route
//             Route route = findRoute(srcPos, dstPos, cgra);
//             result.addRoute(edge.src, edge.dst, route);
            
//             // Check if actual routing delay violates timing
//             int actualRoutingDelay = route.delay();
//             int srcFinishTime = nodeSchedule[edge.src] + EXECUTION_TIME;
//             int dstStartTime = nodeSchedule[edge.dst];
//             int requiredDelay = edge.isLoopCarried ? 0 : actualRoutingDelay;
            
//             if (dstStartTime < srcFinishTime + requiredDelay) {
//                 std::cerr << "WARNING: Timing violation on edge " << edge.src << " -> " << edge.dst
//                          << " (actual routing delay: " << actualRoutingDelay 
//                          << ", conservative estimate: " << CONSERVATIVE_ROUTING_DELAY << ")\n";
//                 needRescheduling = true;
//             }
//         }
//     }
    
//     if (needRescheduling) {
//         std::cerr << "ERROR: Routing delays exceeded conservative estimates - rescheduling needed\n";
//         std::cerr << "       Consider increasing CONSERVATIVE_ROUTING_DELAY or implementing rescheduling\n";
//         return false;
//     }
    
//     // Final validation
//     if (!result.isValidMapping(dfg, cgra)) {
//         std::cerr << "ERROR: Final mapping validation failed\n";
//         return false;
//     }
    
//     std::cout << "\n=== Mapping Successful ===\n";
//     std::cout << "Total schedule length: " << (result.maxTimeStep + 1) << " cycles\n";
//     std::cout << "Average routing delay: ";
    
//     if (!result.routing.empty()) {
//         int totalDelay = 0;
//         for (const auto& [edge, route] : result.routing) {
//             totalDelay += route.delay();
//         }
//         std::cout << (totalDelay / static_cast<double>(result.routing.size())) << std::endl;
//     } else {
//         std::cout << "0 (all operations adjacent)\n";
//     }
    
//     return true;
// }

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main(int argc, char** argv) {
    std::string filename;
    std::string funcname;
    int nrow = 3, ncol = 3;  // Default CGRA dimensions

    // Command line argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            filename = argv[++i];
        } else if (arg == "-fn" && i + 1 < argc) {
            funcname = argv[++i];
        } else if (arg == "--nrow" && i + 1 < argc) {
            nrow = std::atoi(argv[++i]);
        } else if (arg == "--ncol" && i + 1 < argc) {
            ncol = std::atoi(argv[++i]);
        }
    }

    if (filename.empty() || funcname.empty()) {
        std::cerr << "Usage: " << argv[0] << " -f <filename> -fn <function_name> [--nrow N] [--ncol M]\n";
        std::cerr << "  -f <filename>     : LLVM IR file to parse\n";
        std::cerr << "  -fn <function>    : Function name to analyze\n";
        std::cerr << "  --nrow N          : Number of CGRA rows (default: 3)\n";
        std::cerr << "  --ncol M          : Number of CGRA columns (default: 3)\n";
        return 1;
    }

    if (nrow <= 0 || ncol <= 0) {
        std::cerr << "ERROR: CGRA dimensions must be positive (nrow=" << nrow << ", ncol=" << ncol << ")\n";
        return 1;
    }

    // Parse LLVM IR
    LLVMContext context;
    SMDiagnostic err;
    std::unique_ptr<Module> module = parseAssemblyFile(filename, err, context);
    if (!module) {
        errs() << "Failed to parse IR file\n";
        err.print(argv[0], errs());
        return 1;
    }

    Function *func = module->getFunction(funcname);
    if (!func) {
        errs() << "Function '" << funcname << "' not found in module.\n";
        return 1;
    }

    // Build analysis passes
    DominatorTree DT(*func);
    LoopInfo LI(DT);

    // Find the innermost loop
    Loop *innerLoop = findInnermostLoop(LI);
    if (!innerLoop) {
        std::cerr << "No inner loop found in function '" << funcname << "'.\n";
        return 1;
    }

    // Generate DFG from the loop
    std::cout << "=== Extracting DFG from innermost loop ===\n";
    DFG dfg = generateDFGForArithmeticAndPHI(innerLoop);
    std::cout << "Extracted " << dfg.nodes.size() << " nodes and " << dfg.edges.size() << " edges\n\n";

    // Create CGRA architecture
    std::cout << "=== Creating CGRA Architecture ===\n";
    CGRAArchitecture cgra(nrow, ncol);
    printCGRAArchitecture(cgra);

    // Print DFG in DOT format
    std::cout << "=== Data Flow Graph (DOT format) ===\n";
    printDFGasDot(dfg);

    // Attempt to map DFG to CGRA
    MappingResult mappingResult;
    bool mappingSuccess = mapDFGToCGRA(dfg, cgra, mappingResult);
    
    if (mappingSuccess) {
        std::cout << "=== Mapping Successful ===\n";
        printMappingResult(mappingResult, dfg, cgra);
        printMappingVisualization(mappingResult, cgra);
        
        // Print visualization for each time step
        for (int t = 0; t <= mappingResult.maxTimeStep; ++t) {
            printMappingVisualization(mappingResult, cgra, t);
        }
    } else {
        std::cout << "=== Mapping Failed or Not Yet Implemented ===\n";
        printMappingResult(mappingResult, dfg, cgra);
    }

    return 0;
}