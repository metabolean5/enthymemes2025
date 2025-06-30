import sys
from typing import Any, List
from posixpath import relpath
import pprint as pp
import glob
import json
import csv
from collections import Counter
from itertools import tee, islice
import os
import random

from rst_tree import * #also inports the build_from_constituency function()

import pandas as pd

import spacy
nlp = spacy.load("en_core_web_sm")


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
    
    print(long_label)


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
    # Go through each element in the input:
    # - if it is an open parenthesis, find matching parenthesis and make recursive
    #   call for content in-between. Add the result as an element to the current list.
    # - if it is an atom, just add it to the current list.
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
    with open('dmrst_predictions_news.json', 'r') as json_file:
        dmrst_predictions = json.load(json_file)

    aligned_pred = {}
    for doc in dmrst_predictions:
   
        relations = dmrst_predictions[doc]['tree_topdown'][0].split('(')

        edus_text = []
        tokens = dmrst_predictions[doc]['tokens']
        segments =[0] + dmrst_predictions[doc]['segments']
        for i,index in enumerate(segments):
            if i+1 == len(segments): break
            if index == 0:
                sent = ''.join(tokens[index:segments[i+1]+1])
            else:
                sent = ''.join(tokens[index+1:segments[i+1]+1])

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


def get_relations_ngrams(lst, n):
    # Create n-grams using itertools
    ngrams = zip(*(islice(seq, index, None) for index, seq in enumerate(tee(lst, n))))

    # Count the occurrences of each n-gram
    ngram_counts = Counter(ngrams)

    return ngram_counts

'''

count = 0
for doc in dmrst_predictions:
    print(doc)
    
    tree_topdown = dmrst_predictions[doc]['tree_topdown'][0]
    tokens = dmrst_predictions[doc]['tokens']
    segments = dmrst_predictions[doc]['segments']

    tree_inorder, joint_relations = build_tree_from_constituency(tree_topdown,tokens,segments)


    for node in joint_relations:
        count+=1
        print(node.tree_code)
        input(count)

'''


#test = {"source1_24565583_1_1_1": {"tree_topdown": ["(1:Satellite=Background:1,2:Nucleus=span:5) (2:Nucleus=span:4,5:Satellite=Evaluation:5) (2:Nucleus=Joint:2,3:Nucleus=Joint:4) (3:Nucleus=Joint:3,4:Nucleus=Joint:4)"], "segments": [3, 25, 33, 38, 52], "tokens": ["\u2581Za", "nda", "li", ",", "\u2581The", "ir", "\u2581product", "ivit", "y", "\u2581and", "\u2581relative", "\u2581se", "cular", "ism", "\u2581didn", "'", "t", "\u2581stop", "\u2581them", "\u2581from", "\u2581being", "\u2581sla", "ught", "er", "ed", ".", "\u2581The", "\u2581ones", "\u2581who", "\u2581survive", "d", "\u2581moved", "\u2581to", "\u2581Israel", "\u2581and", "\u2581became", "\u2581more", "\u2581religious", ".", "\u2581No", "\u2581wonder", "\u2581new", "er", "\u2581immigrant", "s", "\u2581hold", "\u2581on", "\u2581tight", "er", "\u2581to", "\u2581their", "\u2581culture", "."]}}

def get_text(tree_code):
    splits1 = tree_code.split('[')

    text = ''

    for s in splits1:
        if ']' in s:
            text = text+ ' ' + s.split(']')[0]
    
    text = text.replace('  ', ' ')
    return text[1:]


def get_all_analysis(doc):
    analysis = {
        "Tokens": [],
        "Named_Entities": [],
        "Noun_Chunks": [],
        "Sentences": []
    }
    
    # Extract tokens information
    for token in doc:
        token_info = {
            "Text": token.text,
            "Lemma": token.lemma_,
            "POS": token.pos_,
            "Tag": token.tag_,
            "Dep": token.dep_,
            "Shape": token.shape_,
            "is_alpha": token.is_alpha,
            "is_stop": token.is_stop
        }
        analysis["Tokens"].append(token_info)
    
    # Extract named entities
    for ent in doc.ents:
        entity_info = {
            "Text": ent.text,
            "Start": ent.start_char,
            "End": ent.end_char,
            "Label": ent.label_
        }
        analysis["Named_Entities"].append(entity_info)
    
    # Extract noun chunks
    for chunk in doc.noun_chunks:
        chunk_info = {
            "Text": chunk.text,
            "Root Text": chunk.root.text,
            "Root Dep": chunk.root.dep_,
            "Root Head Text": chunk.root.head.text
        }
        analysis["Noun_Chunks"].append(chunk_info)
    
    return analysis

def contains_terms(input_string):
    """
    Find and return specified terms present in the input string as a comma-separated string in alphabetical order.
    
    Args:
        input_string (str): The string to check.
    
    Returns:
        str: A comma-separated string of terms found in the input string. Returns empty string if no terms are found.
    """
    # List of terms to check for
    terms = [
        "anti-black",
        "monument",
        "blacklives",
        "removed",
        "blacklivesmatter",
        "removal",
        "police brutality",
        "statue",
        "police violence",
        "tribute",
        "abuse of authority",
        "memorial",
        "racism",
        "plaque",
        "racial bias",
        "bust",
        "anti-racism",
        "take down",
        "george floyd",
        "beheaded",
        "slavery",
        "desecrated",
        "slave",
        "vandalized",
        "vandalism",
        "vandals",
        "protests",
        "protesters"
    ]
    
    # Convert input string to lowercase for case-insensitive matching
    input_string = input_string.lower()
    
    # Find all terms present in the input string
    found_terms = [term for term in terms if term.lower() in input_string]

    if found_terms == [] : return None
    
    # Return the found terms as a comma-separated string in alphabetical order
    return ",".join(sorted(found_terms))
    


def reconstruct_text(tokens):
    text = ""
    for token in tokens:
        # Remove the special character (▁) used for spacing
        if token.startswith("\u2581"):
            # Add a space before the token (except at the start) and remove ▁
            if text:  # Only add space if text is not empty
                text += " "
            text += token[1:]  # Append the token without ▁
        else:
            # Append token directly without adding a space
            text += token
    return text


def blm_is_relavant(form_type):

    relavant_rels = ['cause', 'explanation', 'evaluation','enablement']


    for rel in relavant_rels:
        if rel in form_type.lower():
            return True

    return False

def has_causal(text, entities):
    causal_connectives = [
        "because", "due to", "owing to", "on account of", "given that",
        "in view of the fact that", "therefore", "thus", "hence", "consequently",
        "as a result", "accordingly", "for that reason", "thereby", "in consequence",
        "in order that", "in order to", "with the aim of", "with the purpose of"
    ]
    tokens = text.lower().split()
    entity_list = [e.strip().lower() for e in entities.split(',')]
    
    for connective in causal_connectives:
        for i, token in enumerate(tokens):
            if token == connective:
                start = max(0, i - 10)
                end = min(len(tokens), i + 11)
                for entity in entity_list:
                    if entity in tokens[start:end]:
                        return True
    return False




def process_dmrst_output(unary_forms_dic, dmrst_predictions, stats_dic,comment_count,file,biforms_local_dic, seen_connective):

    i = 0
    
    for doc in dmrst_predictions:
        #print(doc)
        #print(doc)
        #i+=1
        #if i > 28000 : break #for auspol v2

        
        tree_topdown = dmrst_predictions[doc]['tree_topdown'][0]
        tokens = dmrst_predictions[doc]['tokens']
        segments = dmrst_predictions[doc]['segments']


        full_text = reconstruct_text(tokens)
        entities = contains_terms(full_text)
        #if entities == None: continue

        tree_inorder, biforms_local, unary_forms = build_tree_from_constituency(tree_topdown,tokens,segments)
 


        tree_types = ""

        for p in biforms_local:

        
            doc_type = file
            comment_count+=1
            text = get_text(biforms_local[p]['tree_code'])

            if contains_terms(text) == False : continue

            #if not blm_is_relavant(biforms_local[p]["form_type"]) :
            #    continue
            #else:
            form_type = biforms_local[p]["form_type"]


            stats_dic['bi']['individual'].setdefault(form_type,0)
            stats_dic['bi']['individual'][form_type] += 1 
            
            '''
            for e in entities.split(','):
                stats_dic['bi']['per_entity'].setdefault(form_type,{})
                stats_dic['bi']['per_entity'][form_type].setdefault(e.strip(),0)
                stats_dic['bi']['per_entity'][form_type][e.strip()] +=1
            '''

            
            

        

            #new_row = {'Document': doc, 'id': p, "form_type" :form_type ,"Text" :text, 'RST_code' :tree_inorder, 'entities' : entities}
            #biforms_local_dic = biforms_local_dic._append(new_row, ignore_index=True)


        for p in unary_forms:
            #if triforms_local[p]['triform_type'] not in ['Joint Joint Evaluation']:continue
            #if triforms_local[p]['tree_code'].count('(') > 10 :
            #    edulen = ">10"
        
            doc_type = file
            comment_count+=1
            #comments_ng.setdefault(triforms_local[p]['triform_type'],0)
            #comments_ng[triforms_local[p]['triform_type']]+=1

            #edulen = unary_forms[p]['tree_code'].count('(')

            #comments_ng.setdefault(edulen,{})
            #comments_ng[edulen].setdefault(unary_forms[p]['form_type'],0)
            #comments_ng[edulen][unary_forms[p]['form_type']] += 1 


            #comments_ng.setdefault(file,0)
            #comments_ng[file]+=1

            text = get_text(unary_forms[p]['tree_code'])

         

            #if contains_terms(text) == False : continue

            #if not blm_is_relavant(unary_forms[p]["form_type"]) :
            '''
            if has_causal(text, entities):
                form_type = "connective"
                if doc in seen_connective:continue
                    seen_connective.setdefault(doc)
                else: 
                    continue
            '''
            #else:
            form_type = unary_forms[p]["form_type"]

            
            stats_dic['uni']['individual'].setdefault(form_type,0)
            stats_dic['uni']['individual'][form_type] += 1 
            
            '''
            for e in entities.split(','):
                stats_dic['uni']['per_entity'].setdefault(form_type,{})
                stats_dic['uni']['per_entity'][form_type].setdefault(e.strip(),0)
                stats_dic['uni']['per_entity'][form_type][e.strip()] +=1
            '''
    

            #new_row = {'Document': doc, 'id': p, "form_type" :form_type ,"Text" :text, 'RST_code' :tree_inorder,'entities' : entities}
            #unary_forms_dic = unary_forms_dic._append(new_row, ignore_index=True)

           
    return stats_dic, comment_count, unary_forms_dic,biforms_local_dic, seen_connective



''' SOCC




            if '_' in doc:
                doc_type = 'comment'
                comment_count+=1
                comments_ng.setdefault(triforms_local[p]['triform_type'],0)
                comments_ng[triforms_local[p]['triform_type']]+=1
            else:
                article_count+=1
                doc_type = 'article'
                articles_ng.setdefault(triforms_local[p]['triform_type'],0)
                articles_ng[triforms_local[p]['triform_type']]+=1
        


comments_ng = {}
comment_count = 0


#with open('dmrst_predictions_news_add.json', 'r') as json_file:
#    dmrst_predictions_add = json.load(json_file)


with open('dmrst_tropes_multifull.json', 'r') as json_file:
    dmrst_predictions = json.load(json_file)

columns = ['Document', 'id', "Tree_type","Text", "RST_code"]
df_triplets = pd.DataFrame(columns=columns)

id2tree = {}
id2treetypes = {}

comments_ng, comment_count, df_triplets, id2tree,id2treetypes  = process_dmrst_output(df_triplets,dmrst_predictions,comments_ng,comment_count, "tropes_synth",id2tree,id2treetypes)
#comments_ng, comment_count, df_triplets = process_dmrst_output(df_triplets,dmrst_predictions_v2,comments_ng,comment_count)


docs = pd.read_csv('tropes_data.csv')
docs = pd.read_csv('tropes_rst_immigration_mp_no_1.csv')

# Add a new column called 'new_col' (you can rename it)
docs['tree_types'] = None
docs['clean_tree_code'] = None

rows_to_drop = []

for index, row in docs.iterrows():

    if row['source'] != "immigration" or row["id"] not in id2tree:
        rows_to_drop.append(index)
        continue
    
    

    #docs.at[index,'rst_tree'] = id2tree[row['id']]
    docs.at[index,'Tree_type'] = id2treetypes[row["id"]]["tree_types"]
    docs.at[index,'clean_tree_code'] = id2treetypes[row["id"]]["tree_code"]



docs.drop(index=rows_to_drop, inplace=True)

docs.to_csv('tropes_CLEANrst_immigration.csv', sep='|', index=False)


 code for jje and tropes stats


tropes_jje = {}
for index, row in df_triplets.iterrows():
    print(row["Document"])
    id = row["Document"]
    tropes_jje.setdefault(id,{"trope_presence" : 0, "jje_presence":0})

    if row["Tree_type"] == "Joint Joint Evaluation":
        tropes_jje[id]["jje_presence"] = 1
    
    if row["Trope"] == 0:
        tropes_jje[id]["trope_presence"] = 1

pp.pprint(tropes_jje)


nb_tropes_jje = 0
nb_tropes_nojje = 0
nb_notropes_jje = 0
nb_notropes_nojje =  0


for t in tropes_jje:
    if tropes_jje[t]['trope_presence'] == 1 and tropes_jje[t]['jje_presence'] ==1:
        nb_tropes_jje +=1

    if tropes_jje[t]['trope_presence'] == 1 and tropes_jje[t]['jje_presence'] == 0:
        nb_tropes_nojje +=1
    
    if tropes_jje[t]['trope_presence'] == 0 and tropes_jje[t]['jje_presence'] == 0:
        nb_notropes_nojje +=1
    
    if tropes_jje[t]['trope_presence'] == 0 and tropes_jje[t]['jje_presence'] == 1:
        nb_notropes_jje +=1

print("number tropes with jje : " + str(nb_tropes_jje))
print("number tropes without  jje : " + str(nb_tropes_nojje))
print("number no tropes with jje : " + str(nb_notropes_jje))
print("number no tropes without  jje : " + str(nb_notropes_nojje))

print(len(df_triplets))
df_triplets.to_csv('tropes_RST.csv', index=False, sep='|')

#articles_ng = sorted(articles_ng.items(), key=lambda kv: kv[1], reverse=True)

comments_ng = sorted(comments_ng.items(), key=lambda kv: kv[1], reverse=True)

print()
print('total nb comment: ' +str(comment_count ))
pp.pprint(comments_ng[:50])





'''

#stats_dic = {"bi" : {"individual":{}, "per_entity":{} }, "uni" : {"individual":{}, "per_entity":{} }}
stats_dic = {"bi" : {"individual":{} }, "uni" : {"individual":{}}}
comment_count = 0
columns = ['Document', 'id', "Type","Text", "RST_code", "full_text",'entities' ]
unary_forms_dic = pd.DataFrame(columns=columns)
biforms_local_dic =  pd.DataFrame(columns=columns)

seen_connective = {}

i = 0



for blm_file in glob.glob('blm_rst/*.json'):


    print(blm_file)
    with open(blm_file, 'r') as json_file:
        dmrst_predictions = json.load(json_file)
    stats_dic, comment_count, unary_forms_dic,biforms_local_dic,seen_connective = process_dmrst_output(unary_forms_dic,dmrst_predictions,stats_dic,comment_count,blm_file,biforms_local_dic,seen_connective)


pp.pprint(stats_dic)

with open('blm_stats_general.json', 'w') as file:
    json.dump(stats_dic, file, indent=4)




biforms_local_dic.to_csv('blm_biforms_forms_entities.csv', sep='|', index=False)
unary_forms_dic.to_csv('bln_unary_forms_entities.csv', sep='|', index=False)







"""


    inorder_txt = str(tree_inorder).split(' ')
    ngrams = get_relations_ngrams(inorder_txt,3)

    for p in parataxes_local:
        if '_' in doc:
            doc_type = 'comment'
        else:
            doc_type = 'article'

        new_row = {'Document': doc, 'Parataxe_id': p, "Doc_type": doc_type , 'Joint1': parataxes_local[p][0] ,"Joint2" : parataxes_local[p][1],"Joint3": parataxes_local[p][2]}

        df_parataxes = df_parataxes._append(new_row, ignore_index=True)

    for ng in ngrams:
        if '_' in doc:
            comments_ng.setdefault(ng,ngrams[ng])
            comments_ng[ng]+=ngrams[ng]

        else:
            articles_ng.setdefault(ng,ngrams[ng])
            articles_ng[ng]+=ngrams[ng]


pp.pprint(df_parataxes)

"""
'''

columns = ['Document', 'Parataxe_id', 'Doc_type', "Joint1","Joint2","Joint3"]
df_parataxes = pd.DataFrame(columns=columns)

for doc in dmrst_predictions:
    
    tree_topdown = dmrst_predictions[doc]['tree_topdown'][0]
    tokens = dmrst_predictions[doc]['tokens']
    segments = dmrst_predictions[doc]['segments']

    tree_inorder, parataxes_local = build_tree_from_constituency(tree_topdown,tokens,segments)
    inorder_txt = str(tree_inorder).split(' ')
    ngrams = get_relations_ngrams(inorder_txt,3)

    for p in parataxes_local:
        if '_' in doc:
            doc_type = 'comment'
        else:
            doc_type = 'article'

        new_row = {'Document': doc, 'Parataxe_id': p, "Doc_type": doc_type , 'Joint1': parataxes_local[p][0] ,"Joint2" : parataxes_local[p][1],"Joint3": parataxes_local[p][2]}

        df_parataxes = df_parataxes._append(new_row, ignore_index=True)

    for ng in ngrams:
        if '_' in doc:
            comments_ng.setdefault(ng,ngrams[ng])
            comments_ng[ng]+=ngrams[ng]

        else:
            articles_ng.setdefault(ng,ngrams[ng])
            articles_ng[ng]+=ngrams[ng]


pp.pprint(df_parataxes)
''' #parataxe extraction








'''
aligned_pred = get_aligned_pred()
attributions = {}

for txt in aligned_pred:
    for rel in aligned_pred[txt]:
        if 'attribution' not in aligned_pred[txt][rel]['relation']: continue
        aligned_pred[txt][rel]['edu1_text'] = '['+ aligned_pred[txt][rel]['edu1_text'] +']'
        aligned_pred[txt][rel]['edu2_text'] = '['+ aligned_pred[txt][rel]['edu2_text'] +']'
        attributions.setdefault(txt+'-'+rel, aligned_pred[txt][rel])


pp.pprint(attributions)
print(len(attributions))


csv_file_path = 'attributions_news.csv'
# Write data to CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    fieldnames = ['id', 'N-S_pred', 'N_distance', 'edu1_text', 'edu2_text', 'relation']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write the header
    writer.writeheader()

    # Write the data
    for key, values in attributions.items():
        row_data = {'id': key}
        row_data.update(values)
        writer.writerow(row_data)
'''


'''
rst_relations = run_base_eval(aligned_pred,aligned_gold)

with open("rst_relations.json", "w") as file:
    json.dump(rst_relations, file)
'''
