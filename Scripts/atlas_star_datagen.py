import os, sys, re, tqdm, csv, random, glob
import javalang
import numpy as np

whitespace_re = re.compile(r'\s+')

#split_ori_re = re.compile("\"<AssertPlaceHolder>\" ;[ }]*")
split_re = re.compile("\"<AssertPlaceHolder>\" .*?;( })+")

long_re = re.compile("(assertEquals\([-]?[0-9]+)L(, long[0-9]*\))")
cast_re = re.compile("(assertEquals\()\(\w+\)(\S*, \w+\))")
paren_re = re.compile("(assertEquals\()\((\S+)\)(, \w+\))")
equals_bool_re = re.compile("assertEquals \( (false|true) , (.*) \)")

arg_re_generic = re.compile("assert\w*\s*\((.*)\)")

vocab = None

TEMPLATES = ["assertTrue", "assertFalse", "assertEquals", "assertNotNull", "assertNull"]

errs = {"non-template-assertion-type": 0,
        "assertion-not-in-templates":0, 
        "non-typeable-arg": 0,
        "cant-parse": 0,
        "no-return": 0,
        "more-than-2-args": 0,
        "misc": 0
        }

output_err = False

nontmp_assert = {'non':[]}
falseNum = 0



def clean(code):
    return whitespace_re.sub(' ', code).strip()

def parser_type_to_java_type(t):
    try:
        t = t.value if "value" in dir(t) else t.member
    except AttributeError:
        return None

    if t == "true" or t == "false":
        return bool
    try:
        t = int(t)
        return int
    except ValueError:
        try:
            t = float(t)
            return float
        except ValueError:
            return str
    except TypeError:
        return None

def get_type(assertion_type, assertion, arg, full_test):
    full_test = "public void " + full_test.replace("\"<AssertPlaceHolder>\" ;", assertion + " ; ")

    tokens = javalang.tokenizer.tokenize(full_test)
    parser = javalang.parser.Parser(tokens)
    try:
        test_obj = parser.parse_member_declaration()
    except (javalang.parser.JavaSyntaxError, TypeError, IndexError, StopIteration):
        if(output_err):
            f_cant_parse.write(assertion+'\n')
        errs["cant-parse"] += 1
        return None

    target_type = None

    if assertion_type == "assertTrue" or assertion_type == "assertFalse":
        target_type = bool

    elif assertion_type == "assertEquals":
        target_type =  parser_type_to_java_type(arg)

    if assertion_type == "assertNotNull" or assertion_type == "assertNull" and isinstance(arg, javalang.tree.MethodInvocation):
        return None

    all_var_types = []
    all_vars = []
    for p, node in test_obj:
        if isinstance(node, javalang.tree.LocalVariableDeclaration):
            name = node.declarators[0].name

            if "member" in dir(arg) and name == arg.member and not target_type:
                target_type = pretty_type(node.type)
            else:
                all_var_types += [pretty_type(node.type)]
                all_vars += [node.declarators[0].name]

        elif isinstance(node, javalang.tree.Literal):
            all_var_types += [parser_type_to_java_type(node.value)]
            all_vars += [node.value]


    if not target_type:
        if(output_err):
            f_misc.write(assertion+"\n")
        errs["misc"] += 1
        return None

    same_type_vars = []
    for var, _type in zip(all_vars, all_var_types):
        if _type == target_type:
            same_type_vars += [var]

    return target_type, same_type_vars

def pretty_type(tree, full_type=""):
    if isinstance(tree, javalang.tree.BasicType):
        return tree.name

    if tree.sub_type:
        return tree.name + "." + pretty_type(tree.sub_type, full_type)
    return tree.name
        

def get_type_info(assertion, test_method):
    global falseNum
    #print(assertion)
    start = 0

    if not test_method: 
        if(output_err):
            f_cant_parse.write(assertion+'\n')
        errs["cant-parse"] += 1
        return None

    end = assertion.find("(")

    tokens = javalang.tokenizer.tokenize(assertion)
    parser = javalang.parser.Parser(tokens)

    try:
        assertion_obj = parser.parse_primary()
    except:
    #except javalang.parser.JavaSyntaxError:
        falseNum += 1
        return False

    if(hasattr(assertion_obj, 'arguments') == False):
        return False
    
    if len(assertion_obj.arguments) > 2: 
        if(output_err):
            f_more_than_2_args.write(assertion+'\n')
        errs["more-than-2-args"] += 1
        return False

    assertion_type = assertion[start:end].strip()
    if not assertion_type in TEMPLATES:
        if(output_err):
            f_non_template_assertion_type.write(assertion+"\n")
        errs["non-template-assertion-type"] += 1
        nontmp_assert['non'] += [assertion_type]
        return False

    #IF there is only 1 arg -> then use first arg
    #OTHERWISE, find the method invocation
     
    relevant_arg = None
    other_arg = None
    total_args = len(assertion_obj.arguments)
    arg_num = -1
    if len(assertion_obj.arguments) == 1:
        relevant_arg = assertion_obj.arguments[0]
    else:
        for arg_idx, arg in enumerate(assertion_obj.arguments):
            if isinstance(arg, javalang.tree.MethodInvocation):
                relevant_arg = arg
                arg_num = arg_idx
            else:
                other_arg = arg

    if not relevant_arg:
        #print("err non-typeable-arg: ",assertion)
        if(output_err):
            f_non_typeable_arg.write(assertion+'\n')
        errs["non-typeable-arg"] += 1
        return False

    if not other_arg: other_arg = relevant_arg

    out = get_type(assertion_type, assertion, other_arg, test_method)
    if not out: return False

    _type, matching_type_vars = out

    return _type, arg_num, total_args, matching_type_vars

def assertion_to_arg(assertion, arg_num, total_args):
    m = arg_re_generic.search(assertion)

    g = m.group(1)
    args = g.split(",")
    try:
        assert len(args) == total_args and total_args <= 2 and len(args) > arg_num 
    except AssertionError as e:
        if total_args == 1:
            return g
        else:
            if(output_err):
                f_misc.write(assertion+"\n")
            errs["misc"] += 1
            raise e

    return args[arg_num]


def gen_varients(_type, arg, matching_type_vars):
    
    out = []
    values = matching_type_vars 
    arg = arg.strip()
    if _type in vocab: 
        top_values = list(vocab[_type].keys())
        if _type == int:
            top_values = [int(x.replace("(", "").replace(")","").replace(" ","")) for x in top_values]
        elif _type == float:
            top_values = [float(x.replace("(", "").replace(")","").replace(" ","").replace('Complex.', '')) for x in top_values]
        elif _type == str:
            top_values += ["'"+k+"'" for k in vocab[_type].keys()]

        values = top_values + values

    for var in values:
        out += ["assertEquals ( {} , {} )".format(var, arg)]

    if _type == bool:
        out +=  ["assertTrue ( {} )".format(arg), "assertFalse ( {} )".format(arg)]
    elif not _type == int and not _type == float:
        out += ["assertNotNull ( {} )".format(arg), "assertNull ( {} )".format(arg)]
    
    return list(set(out))


def get_data(assertion_file, method_file):
    method_test_assert_data = []

    assertions = open(assertion_file, encoding="utf-8").read().split("\n")

    print(f'processing {len(assertions)} samples')

    n_type_inferred = 0

    methods = open(method_file,encoding="utf-8").read().split("\n")

    clean_methods = []
    original = sys.stdout
    sys.stdout = open("extrainfo.txt",'w')
    for method in methods:
        match = split_re.search(method)
        
        if not match:
            clean_methods += [(None, None)]
            continue
        ori_match = split_re.search(method)
        
        if ori_match.span()[1] < match.span()[1]:
            print(method[match.span()[0]:match.span()[1]])
            print(method[ori_match.span()[0]:ori_match.span()[1]])
            print(f'ori_idx:{ori_match.span()},idx:{match.span()}')

        idx = match.span()[1]
        test_method = method[0:idx] 
        focal_method = method[idx:]

        clean_methods += [(test_method, focal_method)]
    sys.stdout.close()
    sys.stdout = original
    methods = clean_methods

    assert len(assertions) == len(methods)

    included_idxs = []
    aligned_included_idxs = []

    atlas_star = []

    for idx, (assertion, method) in tqdm.tqdm(enumerate(zip(assertions, methods))):

        test_method = str(method[0]).strip()
        focal_method = str(method[1]).replace("< focal > ","",1).strip()

        assertion = assertion.strip()
        start = len("org . junit . Assert . ")
        if assertion.startswith("org . junit . Assert . "):
            assertion = assertion[start:]
        if("Assert . " in assertion):
            assertion = assertion.replace("Assert . ", "", 1)
        if("Assertions . " in assertion):
            assertion = assertion.replace("Assertions . ", "", 1)

        m = equals_bool_re.match(assertion)
        if m:
            if m.group(1) == "true":
                assertion = "assertTrue ( {} )".format(m.group(2))
            else:
                assertion = "assertFalse ( {} )".format(m.group(2))

        
        out = get_type_info(assertion, test_method)

        if not out: continue

        _type, arg_num, total_args, matching_type_vars = out
        
        try:
            arg_txt = assertion_to_arg(assertion, arg_num, total_args)
        except AssertionError:
            continue

        focal_method_clean = clean(focal_method)
        test_method_clean = clean(test_method)
        if not focal_method_clean: 
            focal_method_clean = "()"

        n_type_inferred += 1

        # save idx, fm, test, assertion, type
        atlas_star += [(idx, focal_method_clean, test_method_clean, assertion, str(_type))]
        

        template_asserts = gen_varients(_type, arg_txt, matching_type_vars)

        # TEMPLATE BASED DATAGEN
        if assertion not in template_asserts:
            if m := paren_re.search(assertion):
                assertion = m.group(1) + m.group(2) + m.group(3)

            if m := long_re.search(assertion):
                assertion = m.group(1) + m.group(2)
            elif m := cast_re.search(assertion):
                assertion = m.group(1) + m.group(2)


        if assertion not in template_asserts:
            '''
            print("*******************************")
            print("*******************************")
            print("err assertion-not-in-templates: ")
            print("assertion = ", assertion)
            print("template_asserts = ", template_asserts)
            print("assertion not in template_asserts:", (assertion not in template_asserts))
            print("*******************************")
            print("*******************************")
            '''
            if(output_err):
                f_assertion_not_in_templates.write(assertion+'\n')
            errs["assertion-not-in-templates"] += 1
            continue

        assertion_clean = clean(assertion)


        # GEN (lbl, method, test, assert)
        pos_sample = (1, focal_method_clean, test_method_clean, assertion_clean)
        neg_samples = []
        for i in range(len(template_asserts)):
            if not assertion_clean == clean(template_asserts[i]): #and\
                #len(neg_samples) < 3: and random.random() > 0.5: # NOTE: seeded above
                neg_samples += [(0, focal_method_clean, test_method_clean, clean(template_asserts[i]))]
                if "assertTrue" in pos_sample[-1] and "assertTrue" in template_asserts[i] or \
                        "assertFalse" in pos_sample[-1] and "assertFalse" in template_asserts[i]:
                    print(pos_sample[-1])
                    print(clean(template_asserts[i]))
                    print()
        # assert len(neg_samples) > 0

        method_test_assert_data += [pos_sample] + neg_samples

        included_idxs += [idx]
        for i in range(len(neg_samples)+1):
            aligned_included_idxs += [idx]

    return method_test_assert_data, included_idxs, aligned_included_idxs, atlas_star

if __name__ == "__main__":
    random.seed(0)

    #if not "ATLAS_PATH" in os.environ:
    #    print("Set your ATLAS_PATH!")
    #    sys.exit(1)

    #ATLAS_PATH = os.environ["ATLAS_PATH"]

    Output_Dir = "..../"
    ATLAS_Raw_Dir = "..../Raw_Dataset"
    
    vocab_src = np.load("..../toga-main/data/vocab.npy", allow_pickle=True).item()

    
    method_test_assert_data = []
    
    assertion_file = "assertLines.txt"
    method_file = "testMethods.txt"
    
    if(output_err):
        f_non_template_assertion_type = open("non-template-assertion-type.txt","w", encoding="utf-8")
        f_assertion_not_in_templates = open("assertion-not-in-templates.txt","w", encoding="utf-8")
        f_non_typeable_arg = open("non-typeable-arg.txt","w", encoding="utf-8")
        f_cant_parse = open("cant-parse.txt","w", encoding="utf-8")
        f_no_return = open("no-return.txt","w", encoding="utf-8")
        f_more_than_2_args = open("more-than-2-args.txt","w", encoding="utf-8")
        f_misc = open("misc.txt","w", encoding="utf-8")

    #i=0
    #si =chr(i+48)
    #ATLAS_Raw_Dir = ATLAS_Raw_Dir + si
    for split, split_n in zip(["Testing", "Eval", "Training"], ['test', 'valid', 'train']):
        K = 8 if split_n == 'test' else 5
        vocab = {} 
        for k,v in vocab_src.items():
            vocab[k] = {k2: v2 for k2, v2 in list(reversed(sorted(v.items(), key=lambda item: item[1])))[0:K]}

        prev_err_size = sum([v for v in errs.values()])
        #f_assertion = os.path.join(ATLAS_Raw_Dir, split, assertion_file)
        #f_method = os.path.join(ATLAS_Raw_Dir, split, method_file)
        f_assertion = ATLAS_Raw_Dir +"/"+ split+"/" + assertion_file
        f_method = ATLAS_Raw_Dir +"/"+ split +"/" + method_file

        print(f'processing {split_n} split')
    
        local_method_test_assert_data, included_idxs, aligned_included_idxs, atlas_star = get_data(f_assertion, f_method)

        if not os.path.isdir(Output_Dir+"/"+'atlas_star'):
            os.makedirs(Output_Dir+"/"+'atlas_star')
            
        #kwargs = {'escapechar': '|'}

        with open(Output_Dir+"/"+f'atlas_star'+ f'/{split_n}.csv', "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, escapechar="|") 
            w.writerow(["idx","label","fm","test","assertion"])
            for idx, d in enumerate(local_method_test_assert_data):
                assert len(list(d)) == 4
                w.writerow([idx] + list(d))

        print(f'wrote to atlas_star'+f'/{split_n}.csv')

        n_pos = sum(map(lambda r: r[0]==1, local_method_test_assert_data))
        print(f"in vocab assertions in {split_n}: {n_pos}")
        out_of_vocab_assertions = errs['assertion-not-in-templates']
        print(f"out of vocab assertions in {split_n}: {out_of_vocab_assertions}")
        print(f"total samples (pos + neg): {len(local_method_test_assert_data)}")
        print("-"*100)

        with open(Output_Dir+"/"+f'atlas_star'+f'/{split_n}.oov_asserts.txt', 'w') as f:
            f.write(str(out_of_vocab_assertions))


        method_test_assert_data += local_method_test_assert_data

    n_pos = sum(map(lambda r: r[0]==1, method_test_assert_data))
    n_neg = sum(map(lambda r: r[0]==0, method_test_assert_data))

    print('ASSERT SAMPLES:', len(method_test_assert_data), 'total,', n_pos, 'positive,', n_neg, 'negative,')
    print('MISSED TEMPLATES', errs['assertion-not-in-templates'])
    print('TEMPLATE MATCH', len(method_test_assert_data),'collected, ERRS:', errs)
    
    print("JavaSyntaxError number = ", falseNum)
    
    if(output_err):
        f_non_template_assertion_type.close()
        f_assertion_not_in_templates.close()
        f_non_typeable_arg.close()
        f_cant_parse.close()
        f_no_return.close()
        f_more_than_2_args.close()
        f_misc.close()
