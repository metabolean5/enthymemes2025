import json
import pickle
import re
from collections import defaultdict
import pprint

def load_dms(file_path):
    """Load discourse markers from dms.txt."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read DMs, strip whitespace, and convert to lowercase for case-insensitive matching
            return [line.strip().lower() for line in file if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"DM file not found at: {file_path}")

def get_all_relations(json_data):
    """Collect all unique relations across all documents."""
    all_relations = set()
    for doc_id, data in json_data.items():
        tree_topdown = data.get('tree_topdown', [])
        for tree_str in tree_topdown:
            matches = re.findall(r'\((\d+):(\w+)=([\w-]+):([\d,]+)(?:,(\d+):(\w+)=([\w-]+):([\d,]+))?\)', tree_str)
            for match in matches:
                left_relation = match[2]
                right_relation = match[6] if len(match) > 6 else None
                for rel in [left_relation, right_relation]:
                    if rel and rel.lower() != 'span':
                        all_relations.add(rel.lower())
    # Sort relations and take first 18
    sorted_relations = sorted(all_relations)[:18]
    # Create relation-to-index mapping
    relation_to_index = {rel: idx for idx, rel in enumerate(sorted_relations)}
    print(relation_to_index)

    return relation_to_index

def parse_rst_tree(tree_topdown, relation_to_index):
    # Initialize tree structure
    nodes = {}
    relations = []
    depths = defaultdict(list)
    nucleus_counts = defaultdict(int)
    satellite_counts = defaultdict(int)
    node_roles = {}
    max_depth = 0
    internal_nodes = 0
    leaf_nodes = 0

    # Parse each tree string
    for tree_str in tree_topdown:
        matches = re.findall(r'\((\d+):(\w+)=([\w-]+):([\d,]+)(?:,(\d+):(\w+)=([\w-]+):([\d,]+))?\)', tree_str)
        for match in matches:
            left_id, left_role, left_relation, left_span = match[0:4]
            right_id, right_role, right_relation, right_span = match[4:8] if len(match) > 4 else (None, None, None, None)

            # Store node roles
            node_roles[left_id] = left_role
            if right_id:
                node_roles[right_id] = right_role

            # Parse spans
            left_spans = [int(s) for s in left_span.split(',')]
            right_spans = [int(s) for s in right_span.split(',')] if right_span else []

            # Determine node type
            is_leaf = len(left_spans) == 1 and left_relation.lower() == 'span'
            if is_leaf:
                leaf_nodes += 1
            else:
                internal_nodes += 1

            # Store node info
            nodes[left_id] = {'role': left_role, 'relation': left_relation, 'spans': left_spans}
            if right_id:
                nodes[right_id] = {'role': right_role, 'relation': right_relation, 'spans': right_spans}
                relations.append({
                    'left_id': left_id, 'left_role': left_role, 'left_relation': left_relation,
                    'right_id': right_id, 'right_role': right_role, 'right_relation': right_relation
                })

    # Compute depths using a recursive function
    def compute_depth(node_id, current_depth=0, visited=None):
        if visited is None:
            visited = set()
        if node_id in visited:
            return current_depth
        visited.add(node_id)

        max_child_depth = current_depth
        for rel in relations:
            if rel['left_id'] == node_id:
                child_depth = compute_depth(rel['right_id'], current_depth + 1, visited)
                max_child_depth = max(max_child_depth, child_depth)
            elif rel['right_id'] == node_id:
                child_depth = compute_depth(rel['left_id'], current_depth + 1, visited)
                max_child_depth = max(max_child_depth, child_depth)

        # Assign depth to relation
        node = nodes.get(node_id, {})
        relation = node.get('relation', '').lower()
        if relation in relation_to_index:
            relation_index = relation_to_index[relation]
            depths[relation_index].append(max_child_depth)

        return max_child_depth

    # Compute depths for all root nodes
    for node_id in nodes:
        if nodes[node_id].get('relation').lower() != 'span':
            max_depth = max(max_depth, compute_depth(node_id))

    # Count nucleus and satellite roles
    for rel in relations:
        left_relation = rel['left_relation'].lower()
        right_relation = rel['right_relation'].lower()
        left_role = rel['left_role']
        right_role = rel['right_role']
        if left_relation in relation_to_index:
            relation_index = relation_to_index[left_relation]
            if left_role == 'Satellite':
                satellite_counts[relation_index] += 1
            elif left_role == 'Nucleus':
                nucleus_counts[relation_index] += 1
        if right_relation in relation_to_index:
            relation_index = relation_to_index[right_relation]
            if right_role == 'Satellite':
                satellite_counts[relation_index] += 1
            elif right_role == 'Nucleus':
                nucleus_counts[relation_index] += 1

    return relations, depths, nucleus_counts, satellite_counts, max_depth, internal_nodes, leaf_nodes, node_roles, nodes

def get_rst_feature_vector(data, relation_to_index, dms):
    tree_topdown = data['tree_topdown']
    segments = data['segments']
    tokens = data['tokens']

    # Parse tree
    relations, depths, nucleus_counts, satellite_counts, max_depth, internal_nodes, leaf_nodes, node_roles, nodes = parse_rst_tree(tree_topdown, relation_to_index)

    # Initialize 31-dimensional vector (30 original + 1 for DM presence)
    vector = [0] * 31

    # 1. Sum of depths for each relation (18 dims)
    for relation_index, depth_list in depths.items():
        vector[relation_index] = sum(depth_list)

    

    # 2. Structural and segment features (12 dims)
    num_segments = len(segments)
    print(relations)
    
    num_relations = len([r for r in relations if r['left_relation'].lower() != 'span' or r['right_relation'].lower() != 'span'])
    print(num_relations)
    #input()
    # Compute segment lengths in tokens
    segment_lengths = []
    for i in range(len(segments)):
        start = segments[i-1] if i > 0 else 0
        end = segments[i]
        segment_lengths.append(end - start)
        print(segment_lengths)
    avg_segment_length = sum(segment_lengths) / len(segment_lengths) if segment_lengths else 0

    print(avg_segment_length)

    # Count nucleus and satellite spans
    nucleus_spans = sum(1 for node in nodes.values() if node['role'] == 'Nucleus')
    satellite_spans = sum(1 for node in nodes.values() if node['role'] == 'Satellite')

    # Count relation types (nucleus-satellite, nucleus-nucleus, satellite-satellite)
    ns_relations = 0
    nn_relations = 0
    ss_relations = 0
    for rel in relations:
        left_role = rel['left_role']
        right_role = rel['right_role']
        if (left_role == 'Nucleus' and right_role == 'Satellite') or (left_role == 'Satellite' and right_role == 'Nucleus'):
            ns_relations += 1
        elif left_role == 'Nucleus' and right_role == 'Nucleus':
            nn_relations += 1
        elif left_role == 'Satellite' and right_role == 'Satellite':
            ss_relations += 1

    # Ratio of nucleus to satellite nodes
    nucleus_to_satellite_ratio = nucleus_spans / satellite_spans if satellite_spans > 0 else 0

    # Assign structural features
    vector[18] = num_segments
    vector[19] = num_relations
    vector[20] = avg_segment_length
    vector[21] = max_depth
    vector[22] = nucleus_spans
    vector[23] = satellite_spans
    vector[24] = ns_relations
    vector[25] = nn_relations
    vector[26] = ss_relations
    vector[27] = leaf_nodes
    vector[28] = internal_nodes
    vector[29] = nucleus_to_satellite_ratio

    # 3. DM presence (1 dim)
    tokens_lower = [token.lower() for token in tokens]
    dm_present = any(dm in tokens_lower for dm in dms)
    vector[30] = 1 if dm_present else 0

    return vector

def process_json_data(json_data, relation_to_index, dms):
    rst_feature_dic = {}
    for doc_id, data in json_data.items():
        vector = get_rst_feature_vector(data, relation_to_index, dms)
        rst_feature_dic[doc_id] = vector
    return rst_feature_dic

def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found at: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {file_path}")

def main():
    json_file_path = 'dmrst_tropes_multifull-JJE.json'  # Path to the JSON file
    dms_file_path = 'dms.txt'  # Path to the DMs file
    json_data = load_json_data(json_file_path)
    dms = load_dms(dms_file_path)
    
    # Get global relation-to-index mapping
    relation_to_index = get_all_relations(json_data)
    print("Relation to Index Mapping:")
    pprint.pprint(relation_to_index)
    
    rst_feature_dic = process_json_data(json_data, relation_to_index, dms)
    
    # Pretty print the result
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(rst_feature_dic)
    
    # Save to pickle file
    with open('rst_features4.pkl', 'wb') as file:
        pickle.dump(rst_feature_dic, file)

if __name__ == '__main__':
    main()