import sys
from typing import Any, List
from posixpath import relpath
import pprint as pp
import glob
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




def compute_evaluation_metrics(gold_labels, pred_labels):
    accuracy = accuracy_score(gold_labels, pred_labels)
    precision = precision_score(gold_labels, pred_labels, average='weighted')
    recall = recall_score(gold_labels, pred_labels, average='weighted')
    f1 = f1_score(gold_labels, pred_labels, average='weighted')
    
    return accuracy, precision, recall, f1

# Parse input string into a list of all parentheses and atoms (int or str),
# exclude whitespaces.

def get_aggregated_label(long_label):

    long_label = long_label.lower().replace(' ','')
    to_replace = ['-s','-e','-n']

    for ss in to_replace:
        if ss in long_label[-4:]:
            long_label = long_label.replace(ss,'')

    label_mappings ={
            "ATTRIBUTION": ["attribution", "attribution-negative"],
            "BACKGROUND": ["background", "circumstance",   "preparation"],
            "CAUSE": ["cause", "cause-result", "consequence", "non-volitional-cause", "non-volitional-result",  "result",  "volitional-cause", "volitional-result"],
            "COMPARISON": ["analogy", "comparison", "preference", "proportion"],
            "CONDITION": ["condition", "contingency", "hypothetical", "otherwise", "unconditional", "unless"],
            "CONTRAST": ["antithesis", "antitesis", "concesion", "concession", "contrast"],
            "ELABORATION": ["definition", "elaboration-e", "elaboration", "elaboration-additional","elaboration-argumentative", "elaboration-general-specific", "elaboration-object-attribute", "elaboration-part-whole", "elaboration-process-step", "elaboration-set-member", "example", "parenthetical"],
            "ENABLEMENT": ["enablement", "proposito", "purpose"],
            "EVALUATION": ["comment", "conclusion","evaluation", "interpretation"],
            "EXPLANATION": [ "evidence", "explanation", "explanation-argumentative", "justificacion", "justify",  "motivation", "reason"],
            "JOINT":  ["conjunction", "disjunction",  "joint", "list", "union"],
            "MANNER-MEANS": ["manner", "means"],
            "SAME-UNIT": ["same-unit"],
            "SUMMARY": [ "restatement", "summary"],
            "TEMPORAL": ["inverted-sequence",   "sequence", "temporal-after", "temporal-before", "temporal-same-time"],
            "TEXTUALORGANIZATION": ["textualorganization"],
            "TOPIC-CHANGE": ["topic-drift", "topic-shift"],
            "TOPIC-COMMENT": ["comment-topic", "problem-solution", "question-answer", "rhetorical-question", "solutionhood", "statement-response", "topic-comment"]
            }
    
    for h_label in label_mappings:
        if long_label in label_mappings[h_label]:
            return str(h_label).lower()


def normalize_str(string: str) -> List[str]:
    str_norm = []
    last_c = None
    for c in string:
        if c.isalnum():
            if last_c.isalnum():
                str_norm[-1] += c
            else:
                str_norm.append(c)
        elif not c.isspace():
            str_norm.append(c)
        last_c = c
    return str_norm

# Generate abstract syntax tree from normalized input.
def get_ast(input_norm: List[str]) -> List[Any]:
    ast = []

    i = 0
    while i < len(input_norm):
        symbol = input_norm[i]
        if symbol == '(':
            list_content = []
            match_ctr = 1 # If 0, parenthesis has been matched.
            while match_ctr != 0:
                i += 1
                if i >= len(input_norm):
                    raise ValueError("Invalid input: Unmatched open parenthesis.")
                symbol = input_norm[i]
                if symbol == '(':
                    match_ctr += 1
                elif symbol == ')':
                    match_ctr -= 1
                if match_ctr != 0:
                    list_content.append(symbol)             
            ast.append(get_ast(list_content))
        elif symbol == ')':
                raise ValueError("Invalid input: Unmatched close parenthesis.")
        else:
            try:
                ast.append(int(symbol))
            except ValueError:
                ast.append(symbol)
        i += 1
    return ast

def access_element_by_indices(my_tuple, indices):
    try:
        for index in indices:
            my_tuple = my_tuple[index]
        return my_tuple
    except (IndexError, TypeError):
        return None

def getRelation(seg, ast, index):
  relation =  {'seg1span':None,'seg2span':None,'seg1type':None,'seg2type':None,'relation':None, }
  if seg == 'Nucleus':
    ns_order = 'N-S'
    relation['seg1type'] = 'Nucleus'
    relation['seg2type'] = 'Satellite'
    seg1span_index = index
    seg1span_index[-1] = seg1span_index[-1] + 1
    relation['seg1span'] = access_element_by_indices(ast,seg1span_index)#get neighboor
    sat_span_index = index[:-1]#get parent
    sat_span_index[-1] = sat_span_index[-1] +  1#get neighboor
    sat_span_index += [1]#get 2nd child
    relation['seg2span'] = access_element_by_indices(ast,sat_span_index)
    sat_type_index = sat_span_index
    sat_type_index[-1] = sat_span_index[-1] - 1 #get previous neighboor
    if  access_element_by_indices(ast,sat_type_index) == 'Nucleus':  
        relation['seg2type'] = 'Nucleus'
        ns_order = 'N-N'
    relation_index = sat_span_index
    relation_index[-1] = relation_index[-1] + 2
    relation['relation'] = access_element_by_indices(ast,relation_index)
  if seg == 'Satellite':
    ns_order = 'S-N'
    relation['seg2type'] = 'Nucleus'
    relation['seg1type'] = 'Satellite'
    mutable_index = index
    mutable_index[-1] = mutable_index[-1] + 1
    relation['seg1span'] = access_element_by_indices(ast,mutable_index)
    mutable_index[-1] = mutable_index[-1] + 1 #get next neighboor 
    relation['relation'] = access_element_by_indices(ast,mutable_index)
    mutable_index[-1] = mutable_index[-1] + 2 #get next neighboor 
    mutable_index += [1] #get first child
    relation['seg2span'] = access_element_by_indices(ast,mutable_index)
  return relation,ns_order

def is_nested_list(arr):
  for el in arr:
    if type(el) == tuple:
      return True
def list_to_tuple(input_list):
    if not isinstance(input_list, list):
        return input_list  #nase case
    return tuple(list_to_tuple(item) for item in input_list)

def get_rst_tree_relations(arr):
    doc_relations = {}
    def _iterate_recursive(nested_arr,index):
        if not is_nested_list(nested_arr):
            if type(arr) !=tuple:
              pass
            else:
              if nested_arr in ["Nucleus", "Satellite"]: 
                relation,ns_order = getRelation(nested_arr,arr,index)
                doc_relations.setdefault(str(relation['seg1span']) + str(relation["seg2span"]),[relation,ns_order])
        else:
            #if we haven't reached the leaf element, iterate over the current dimension
            for el in nested_arr:
              _iterate_recursive(el,index + [nested_arr.index(el)])

    if len(arr) == 0:
        #if the array has no dimensions, it's a scalar, just return it
        print('scalar')
    else:
        _iterate_recursive(arr,[])
    

    return doc_relations

def is_integer(string):
    try:
        int(string)
        return True
    except ValueError:
        return False
  
def clean_discourse_relation(doc_relations,file):
  discourse_relations = {}
  abnormal = []
  for rel in doc_relations:

    '''
    print(rel)
    print(doc_relations[rel])
    if "wsj_0616" in file:
        input()
    '''


    if doc_relations[rel][0]['relation'] == None:continue
    if 'leaf' in rel or "span" in rel:

        if 'leaf' in rel.split(',')[0]:
            edu1 = (rel.split(')')[0].split(',')[1])
        else:
            edu1 = (rel.split(')')[0].split(',')[2])
        edu2 = int(edu1) + 1
        edus = str(edu1)+'_'+str(edu2)
        abnormal.append([edus,doc_relations[rel]])
        continue
    
    if "None" in rel :continue
    
    if "," not in rel.split(')')[1]:continue
    if 'leaf' in rel.split(')')[0]:
      edu1 = (rel.split(')')[0].split(',')[1])
    else:
      edu1 = rel.split(')')[0].split(',')[2]

    edu2 = rel.split(')')[1].split(',')[1]
    if not is_integer(edu2) and not is_integer(edu2):continue
    if int(edu2) - int(edu1) != 1: continue
    discourse_relations.setdefault(str(edu1)+'_'+str(edu2),doc_relations[rel])

  for a in abnormal:
      if a[0] not in discourse_relations:
          if a[1][0]['relation'] == None:continue
          discourse_relations.setdefault(a[0],a[1])

  return discourse_relations

def get_gold_trees_dic():
    gold_tree_docs = {}
    for tree_file in glob.glob('gold_RST_trees/*.txt'):
        print(tree_file)
        lisp_tree = open(tree_file).read()
        input_norm = normalize_str(lisp_tree)
        ast = get_ast(input_norm)    
        nd_list = list_to_tuple(ast)
        doc_relations = get_rst_tree_relations(nd_list)         
        discourse_relations_gold =  clean_discourse_relation(doc_relations,tree_file)
        gold_tree_docs.setdefault(tree_file,discourse_relations_gold)
    
    with open("gold_trees_dic.json", "w") as file:
        json.dump(gold_tree_docs, file, indent=4)

    return gold_tree_docs

def compute_distance(relation_string): #compute the ditance by looking at the spans of the 1st occuring discourse tree
    relation_string = relation_string.replace('(',",").replace(')',",")
    digits1 = [char for char in relation_string.split('=')[0] if char.isnumeric()]
    digits2 = [char for char in relation_string.split('=')[1].split(',')[0] if char.isnumeric()]
    span1 = int(''.join(digits1))
    span2 = int(''.join(digits2))
    return span2 - span1

def get_aligned_pred():
    #for raw text dmrst_predictions.json
    #for gold edus break dmrst_predictions_gold_edus
    #for french edus breaks
    with open('dmrst_predictions_french.json', 'r') as json_file:
        dmrst_predictions = json.load(json_file)

    aligned_pred = {}
    for doc in dmrst_predictions:
   
        relations = dmrst_predictions[doc]['tree_topdown'][0].split('(')

        edus_text = []
        tokens = dmrst_predictions[doc]['tokens']
        segments =[0] + dmrst_predictions[doc]['segments']
        for i,index in enumerate(segments):
            if i+1 == len(segments): break
            sent = ' '.join(tokens[index:segments[i+1]])
            sent = " ".join( [token.replace(' ','') for token in sent.split('▁')])
            edus_text.append(sent.replace('▁', ' ')[1:])
        
        for r in relations[1:]:
            edu1 = r.split(',')[0].split(':')[2]
            edu2 = r.split(',')[1].split(':')[0]

            if r.split('=')[1].split(':')[0] == 'span': #identify relation
                relation = r.split('=')[2].split(':')[0].lower()
            else:
                relation = r.split('=')[1].split(':')[0].lower()
            
            if r.split('=')[0].split(':')[1] == 'Nucleus': #identify NS order
                if r.split('=')[1].split(':')[2] == 'Nucleus':
                    ns_order = 'N-N'
                else:
                    ns_order = 'N-S'
            else: 
                ns_order = 'S-N'

            edus = edu1+'_'+edu2
            aligned_pred.setdefault(doc, {})
            aligned_pred[doc].setdefault(edus,{})
            aligned_pred[doc][edus].setdefault('relation',relation)
            aligned_pred[doc][edus].setdefault('edu1_text',edus_text[int(edu1)-1])
            aligned_pred[doc][edus].setdefault('edu2_text',edus_text[int(edu2)-1])
            aligned_pred[doc][edus].setdefault('N-S_pred', ns_order)
            aligned_pred[doc][edus].setdefault('N_distance', compute_distance(r))
    return aligned_pred


def get_edus_text(fname):
    edus = []
    path = "gold_RST_trees/data_edus_text/RSTtrees-WSJ-main-1.0/TEST/"+str(fname).replace('.txt','')+'.out.edus'
    f = open(path).readlines()
    for edu in f:
        edus.append(edu.strip())
    return edus


def get_aligned_gold():
    gold_trees_dic = get_gold_trees_dic()


    relations_dic = {}
    aligned_gold = {}
    for tree in gold_trees_dic:
        print(tree)

        edus_text = get_edus_text(tree.split('/')[1])
        for r in gold_trees_dic[tree]:
            edus = r
            ns_order = gold_trees_dic[tree][r][1]
            r = gold_trees_dic[tree][r][0]
            rel_label = ' '.join(r["relation"]).replace('rel2par','').lower()[1:]
            doc = tree.split("/")[1].replace('.txt','')
            aligned_gold.setdefault(doc,{})
            edus = edus.replace(' ', '')
            aligned_gold[doc].setdefault(edus, {})
            aligned_gold[doc][edus].setdefault('relation',rel_label)
            aligned_gold[doc][edus].setdefault('edu1_text',edus_text[int(edus.split('_')[0])-1])
            try:
                aligned_gold[doc][edus].setdefault('edu2_text',edus_text[int(edus.split('_')[1])-1])
            except:
                print('je suis une merde')
            aligned_gold[doc][edus].setdefault('N-S_gold',ns_order)
            #aligned_gold[].setdefault(rel_label,0)
    return aligned_gold

def gold_relations_stats():
    gold_trees = get_gold_trees_dic()
    relations_full= {}
    for doc in gold_trees:
            
            for r in gold_trees[doc]:
                relation = " ".join(gold_trees[doc][r]["relation"][1:]).lower()
    
                relations_full.setdefault(relation,0)
                relations_full[relation]+=1
    relations_full = sorted(relations_full.items(), key=lambda x:x[1])

    relations_nb = 0
    for r in relations_full:
        relations_nb += r[1]

    return relations_nb

def substring_similarity(string, substring):
    common_substring = ""
    max_common_length = 0
    
    for i in range(len(substring)):
        for j in range(i + 1, len(substring) + 1):
            sub = substring[i:j]
            if sub in string and len(sub) > max_common_length:
                max_common_length = len(sub)
                common_substring = sub

    if len(substring) == 0: substring = "0"
    similarity_ratio = len(common_substring) / len(substring)
    
    return similarity_ratio

def get_relation_distribution():
    aligned_relations_full = {}
    for doc in aligned_pred:
        for r in aligned_pred[doc]:
            aligned_relations_full.setdefault(aligned_pred[doc][r],0)
            aligned_relations_full[aligned_pred[doc][r]]+=1

    aligned_relations_full = sorted(aligned_relations_full.items(), key=lambda x:x[1], reverse=True)


def run_base_eval(aligned_pred,aligned_gold):
    correct = 0
    wrong = 0
    rst_relations = {}

    gold = []
    y_pred = []
    for doc in aligned_gold:



        for r in aligned_gold[doc]:
            if r not in aligned_pred[doc]:continue



            original_label = aligned_gold[doc][r]["relation"]
            aligned_gold_relation = get_aggregated_label(aligned_gold[doc][r]["relation"])
            if aligned_pred[doc][r]['relation'] == aligned_gold_relation:
                file_path = "correct.txt"
                
                if "edu2_text" not in aligned_gold[doc][r]: continue
                edus_text = '['+aligned_pred[doc][r]["edu1_text"] + ']' +'['+ aligned_pred[doc][r]["edu2_text"] + ']'
                rst_relations.setdefault('correct', {})
                with open(file_path, 'a') as file:
                    file.write("\n")
                    file.write(doc+" "+ r+ "\n")
                   # file.write('Prediction for ' + r+' : ' + aligned_pred[doc][r]['relation']+"\n")
                   # file.write('Gold Annotation for ' + r+' : ' + aligned_gold_relation+"\n")
                   # file.write('gold EDUS:'+  edus_text+"\n")
                    file.write('Original label for ' + r+' : ' +original_label+"\n")
                    file.write('N-S order gold' + r+' : ' +aligned_gold[doc][r]["N-S_gold"]+"\n")
                    
                correct+=1
                rst_relations.setdefault('correct', {})
                rst_relations['correct'].setdefault(doc+'_'+r,{'relation':aligned_pred[doc][r]['relation'], 'edus_text':edus_text, 'N-distance':aligned_pred[doc][r]['N_distance']})
                # Open the file in write mode ('w')
                gold.append(aligned_gold_relation)
                y_pred.append(aligned_pred[doc][r]['relation'])


            
            else:
                file_path = "errors.txt"
                with open(file_path, 'a') as file:
                    # Write content to the file
                    if "edu2_text" not in aligned_gold[doc][r]: continue
                    edus_text = '['+ aligned_pred[doc][r]["edu1_text"] + ']' +'['+ aligned_pred[doc][r]["edu2_text"] + ']'
                    file.write("\n")
                    file.write(doc+" "+ r+ "\n")
                   # file.write('Prediction for ' + r+' : ' + aligned_pred[doc][r]['relation']+"\n")
                   # file.write('Gold Annotation for ' + r+' : ' +aligned_gold_relation+"\n")
                   # file.write('gold EDUS:'+  edus_text+"\n")
                    file.write('Original label for ' + r+' : ' +original_label+"\n")
                    file.write('N-S order gold' + r+' : ' +aligned_gold[doc][r]["N-S_gold"]+"\n")
                    file.write('N-S order predicted' + r+' : ' +aligned_pred[doc][r]["N-S_pred"]+"\n")

                wrong+=1
                rst_relations.setdefault('error', {})
                rst_relations['error'].setdefault(doc+'_'+r,{'relation_pred':aligned_pred[doc][r]['relation'], 'relation_gold': aligned_gold_relation, 'edus_text':edus_text, 'N-distance':aligned_pred[doc][r]['N_distance']})
                
                gold.append(aligned_gold_relation)
                y_pred.append(aligned_pred[doc][r]['relation'])


    
    accuracy, precision, recall, f1 = compute_evaluation_metrics(gold, y_pred)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    return rst_relations
 


aligned_gold = get_aligned_gold()
aligned_pred = get_aligned_pred()



'''

for file in aligned_gold:
    prev = 0

    sorted_dict = dict(sorted(aligned_gold[file].items(), key=lambda item: int(item[0].split('_')[0])))

    print(file)
    for rel in sorted_dict:
        print(rel)
        actual = int(rel.split('_')[0])
        if actual - prev != 1 :
            input()
        prev = actual
'''






rst_relations = run_base_eval(aligned_pred,aligned_gold)

with open("rst_relations_french.json", "w") as file:
    json.dump(rst_relations, file)

