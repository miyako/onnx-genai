import sys
import onnx

# -------------------------------------------------------------
# Configuration: The Kill List
# -------------------------------------------------------------
# We add both nodes to ensure they are both gone.
TARGET_NODES = ["node_slice_1", "node_slice_2"]

def bypass_node_recursive(graph, target_name):
    """
    Recursively searches for a node by name in the graph and all subgraphs.
    If found, removes the node and rewires its input to its output consumers.
    Returns True if the node was found and removed.
    """
    node_found = False
    nodes_to_remove = []
    
    # Map to track rewiring: Old_Output -> New_Input
    replacement_map = {}

    # 1. Iterate over nodes
    for node in graph.node:
        # A. Recurse into Subgraphs (If, Loop, Scan)
        for attr in node.attribute:
            if attr.type == 5: # GRAPH
                if bypass_node_recursive(attr.g, target_name):
                    node_found = True
            elif attr.type == 9: # GRAPHS
                for g in attr.graphs:
                    if bypass_node_recursive(g, target_name):
                        node_found = True

        # B. Check if this is the target node
        if node.name == target_name:
            print(f"🔪 FOUND TARGET in graph '{graph.name}': '{node.name}' ({node.op_type})")
            
            # Identify Input (Data) and Output (Data)
            # Slice inputs: data(0), starts(1), ends(2)...
            if len(node.input) > 0 and len(node.output) > 0:
                source_data = node.input[0]
                output_data = node.output[0]
                
                print(f"   -> Wiring '{output_data}' to come directly from '{source_data}'")
                replacement_map[output_data] = source_data
                
                nodes_to_remove.append(node)
                node_found = True
            else:
                print(f"   ⚠️ Skipping '{node.name}' - Strange Input/Output count.")

    # 2. Apply Rewiring
    if replacement_map:
        # A. Update Graph Outputs
        for output in graph.output:
            if output.name in replacement_map:
                print(f"   -> Redirecting Graph Output '{output.name}'")
                output.name = replacement_map[output.name]

        # B. Update Node Inputs (Consumers)
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp in replacement_map:
                    node.input[i] = replacement_map[inp]
    
    # 3. Remove the node
    for n in nodes_to_remove:
        graph.node.remove(n)
        
    return node_found

# -------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python fix_all_slices.py <path_to_model.onnx>")
    sys.exit(1)

model_path = sys.argv[1]
print(f"Opening: {model_path}")
model = onnx.load(model_path)

print(f"Scanning for targets: {TARGET_NODES}...")

any_fixed = False
for target in TARGET_NODES:
    print(f"--- Hunting '{target}' ---")
    if bypass_node_recursive(model.graph, target):
        print(f"✅ Removed '{target}'")
        any_fixed = True
    else:
        print(f"ℹ️  '{target}' not found (already removed?)")

if any_fixed:
    onnx.save(model, model_path)
    print(f"💾 Saved fixed model to: {model_path}")
    print("🚀 Try running your C++ code now!")
else:
    print("⚠️ No nodes were removed. Check if the model is already fixed.")
