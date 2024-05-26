import re
from types import NoneType
import javalang
import traceback

split_re = re.compile("\"<AssertPlaceHolder>\" .*?;( })+")

Testing_Path = "..../data/Testing/"

if_write = 0
number = 0

technique_list = ["NC", "NCC", "LCS_B", "LCS_U", "Score_Edit", "LastCall", "LastCallBeforeAssert","Avg"]
weights = {"NC":1.0, "NCC":0.375, "LCS_B":0.639, "LCS_U":0.768, "Score_Edit":0.694, "LastCall":0.607, "LastCallBeforeAssert":0.435}
failed = {"NC":0, "NCC":0, "LCS_B":0, "LCS_U":0, "Score_Edit":0, "LastCall":0, "LastCallBeforeAssert":0, "Avg":0}
weight_sum = 0
for tech in weights.keys():
    weight_sum += weights[tech]

assertion_list = ["assertArrayEquals", "assertEquals", "assertFalse", "assertNotEquals", \
    "assertNotNull", "assertNotSame", "assertNull", "assertSame", "assertThat", "assertThrows", "assertTrue"]

num_focal_atlas_in = 0
num_focal_combined_in = 0

record_failed_cases = 0

if(record_failed_cases):
    javalang_failed = open("..../data/parse_failed.txt","w",encoding="utf-8")

#Naming Conventions
#testMethod: name of the test method (nt)
#method: name of the method that may be the focal method (nf)
#return: NC and NCC
#NC: if nt(after remove the prefix "test") == nf, NC = 1, else, NC = 0
#NCC: if nt(after remove the prefix "test") contains nf, NCC = 1, else, NCC = 0
def NCAndNCC(testMethod, method):
    if(testMethod[0:4] == "test" or testMethod[0:4] == "Test"):
        testMethodName = testMethod[4:]
    else:
        testMethodName = testMethod

    NC = (testMethodName==method)
    NCC = (method in testMethodName)
    return NC,NCC

#Name Similarity
#return LCS-B, LCS-U, 1-Edit-distance

#return the length of the longest common subsequence
def LCS(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    res = [ [0 for i in range(len1+1)] for j in range(len2+1)]
    for i in range(1, len2+1):
        for j in range(1, len1+1):
            if(str2[i-1] == str1[j-1]):
                res[i][j] = 1 + res[i-1][j-1]
            else:
                res[i][j] = max(res[i-1][j], res[i][j-1])
    return res[-1][-1]

#return edit distance
def EditDistance(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    res = [ [i+j for j in range(len2+1)] for i in range(len1+1)]
    for i in range(1, len1+1):
        for j in range(1, len2+1):
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1
            res[i][j] = min(res[i-1][j]+1, res[i][j-1]+1, res[i-1][j-1]+d)
    return res[len1][len2]

#calculate name similarity
def Name_Similarity(testMethod, method):
    len_nt = len(testMethod)
    len_nf = len(method)
    lcs = LCS(testMethod, method)
    LCS_B = lcs*1.0/(max(len_nf, len_nt))
    LCS_U = lcs*1.0/len_nf
    Score_Edit = 1 - EditDistance(testMethod, method)*1.0/(max(len_nf, len_nt))
    return LCS_B, LCS_U, Score_Edit

#return the list of the method calls using deep first search(DFS)
#if the para ignore_assertion=1, assert statements will be ignored.
def getExecutingMethods(method_name, test_class, ignore_assertion=0, ifdebug = 0):
    global waitlist
    ExecutingMethods = []
    typeDecl = [javalang.tree.ConstructorDeclaration, javalang.tree.MethodDeclaration]
    typeInvo = [javalang.tree.MethodInvocation, javalang.tree.SuperMethodInvocation]

    try:
        tree = javalang.parse.parse(test_class)
    except:
        ExecutingMethods.append("-1")
        return ExecutingMethods

    for path, node in tree:
        try:
            if(type(node) in typeDecl and node.name == method_name):
                #DFS
                waitlist = []
                for item in node.body:
                    waitlist.append(item)
                    if(ifdebug):
                        print(item)
                waitlist.reverse()
                while(len(waitlist)>0):
                    tmp = waitlist.pop()
                    if(ignore_assertion and type(tmp) in typeInvo and tmp.member in assertion_list):
                        continue
                    if(type(tmp) == NoneType or type(tmp) == str or type(tmp) == set or type(tmp) == bool):
                        continue
                    if(type(tmp).__name__ == "list"):
                        for item in reversed(tmp):
                            waitlist.append(item)
                        continue
                    if(type(tmp) in typeInvo):
                        str_ = str(tmp.member)
                        para = []
                        for arg in tmp.arguments:
                            para.append(type(arg).__name__)
                        str_ = str_ + "(" + ", ".join(para) + ")"
                        #ExecutingMethods.append(tmp.member)
                        ExecutingMethods.append(str_)
                    if(type(tmp) == javalang.tree.ClassCreator):
                        tmp_tmp_node = tmp.type
                        while(tmp_tmp_node.sub_type != None):
                            tmp_tmp_node = tmp_tmp_node.sub_type
                        str_ = str(tmp_tmp_node.name)
                        para = []
                        try:
                            for arg in tmp_tmp_node.arguments:
                                para.append(type(arg).__name__)
                        except:
                            para = []
                        str_ = str_ + "(" + ", ".join(para) + ")"
                        ExecutingMethods.append(str_)
                        #ExecutingMethods.append(tmp.type.name)
                    try:
                        subNodes = []
                        for attr in tmp.attrs:
                            subNode = getattr(tmp, attr, None)
                            if(type(subNode) != NoneType and type(subNode) != str and subNode != [] and subNode != set()):
                                subNodes.append(subNode)

                        for item in reversed(subNodes):
                            waitlist.append(item)
                    except :
                        print(traceback.print_exc())
                        print()
                        continue                  
        except:
            continue
    return ExecutingMethods

def identify(test_method_name, test_class, focal_method_atlas_name):
    global number
    #print((test_method_name, test_class, focal_method_atlas_name))
    Highest_score = {}
    Highest_method_idx = {}
    Highest_methods = {}
    for tech in technique_list:
        Highest_score[tech] = 0
        Highest_method_idx[tech] = -1
        Highest_methods[tech] = ""
    
    ExecutingMethodsWithAssert = getExecutingMethods(test_method_name, test_class)

    #parse failed
    if(len(ExecutingMethodsWithAssert)==1 and ExecutingMethodsWithAssert[0]=="-1"):
        return "", {}
    
    #no executing methods
    if(len(ExecutingMethodsWithAssert)==0):
        number = number + 1
        return "", Highest_methods

    method_names_with_assert = []
    total = len(ExecutingMethodsWithAssert)
    for i in range(total):
        tmp = ExecutingMethodsWithAssert[i]
        method_names_with_assert.append( tmp[0: tmp.find("(")].strip() )

    flag = 0
    if(focal_method_atlas_name==""):
        focal_method_atlas = ""
        flag = 1
    else:
        for i in range(total):
            if(focal_method_atlas_name==method_names_with_assert[i]):
                focal_method_atlas=ExecutingMethodsWithAssert[i]
                flag = 1
                break
    if(flag==0):
        focal_method_atlas = focal_method_atlas_name
    
    try:
        for i in range(total):
            if(method_names_with_assert[total-1-i] not in assertion_list):
                Highest_score["LastCall"] = 1
                Highest_method_idx["LastCall"] = total-1-i
                break
    except:
        Highest_score["LastCall"] = 0
        Highest_method_idx["LastCall"] = -1

    try:
        for i in range(total):
            if(method_names_with_assert[i] not in assertion_list):
                Highest_score["LastCallBeforeAssert"] = 1
                Highest_method_idx["LastCallBeforeAssert"] = i
            else:
                break
    except:
        Highest_score["LastCallBeforeAssert"] = 0
        Highest_method_idx["LastCallBeforeAssert"] = -1
    
    for i in range(total):
        NC, NCC = NCAndNCC(test_method_name, method_names_with_assert[i])
        LCS_B, LCS_U, Score_Edit = Name_Similarity(test_method_name, method_names_with_assert[i])
        Avg = weights["NC"]*NC + weights["NCC"]*NCC + weights["LCS_B"]*LCS_B + weights["LCS_U"]*LCS_B + weights["Score_Edit"]*Score_Edit
        if(i == Highest_method_idx["LastCall"]):
            Avg += weights["LastCall"]*1
        if(i == Highest_method_idx["LastCallBeforeAssert"]):
            Avg += weights["LastCallBeforeAssert"]*1
        Avg /= weight_sum

        #update NC & NCC
        score = Highest_score["NC"]
        if(NC>=score):
            Highest_score["NC"] = NC
            Highest_method_idx["NC"] = i

        score = Highest_score["NCC"]
        if(NCC>=score):
            Highest_score["NCC"] = NCC
            Highest_method_idx["NCC"] = i
        
        #update LCS_B, LCS_U, Score_Edit
        score = Highest_score["LCS_B"]
        if(LCS_B>=score):
            Highest_score["LCS_B"] = LCS_B
            Highest_method_idx["LCS_B"] = i

        score = Highest_score["LCS_U"]
        if(LCS_U>=score):
            Highest_score["LCS_U"] = LCS_U
            Highest_method_idx["LCS_U"] = i

        score = Highest_score["Score_Edit"]
        if(Score_Edit>=score):
            Highest_score["Score_Edit"] = Score_Edit
            Highest_method_idx["Score_Edit"] = i
        
        #update Avg
        score = Highest_score["Avg"]
        if(Avg>=score):
            Highest_score["Avg"] = Avg
            Highest_method_idx["Avg"] = i

    '''
    if(Highest_method_idx["LastCallBeforeAssert"]!=-1):
        last_call_before_assert = ExecutingMethodsWithAssert[Highest_method_idx["LastCallBeforeAssert"]]
    else:
        last_call_before_assert = ""
    focal_method = ExecutingMethodsWithAssert[Highest_method_idx["Avg"]]
    '''
    for tech in technique_list:
        if(Highest_score[tech]>0 and Highest_method_idx[tech]!=-1):
            Highest_methods[tech] = ExecutingMethodsWithAssert[Highest_method_idx[tech]]
        else:
            failed[tech] += 1
            Highest_methods[tech] = ""

    return focal_method_atlas, Highest_methods

#delete entries with syntax error, and the implementation of focal method
#benchmark-0: atlas focal method (last non-junit method call)
#benchmark-1: last call before assert
#benchmark-2: NC
#benchmark-3: NCC
#benchmark-4: LCS_B
#benchmark-5: LCS_U
#benchmark-6: ED
#benchmark-7: Combined
def improve_focal(testMethodsPath, assertPath):
    #global sum_case, lcba, our_focal
    global num_focal_atlas_in, num_focal_combined_in, number
    methods = open(testMethodsPath).read().split("\n")
    asserts = open(assertPath).read().split("\n")

    total_num = len(asserts)
    if len(asserts) != len(methods):
        print("method line and assertion line not match!")
        return

    output_dir = "..../data/benchmark_new/"
    '''
    output_m_list = []
    output_a_list = []
    for i in range(0,8):
        output_m_list.append( open(output_dir+"benchmark-"+str(i)+"/testMethods.txt", "w", encoding="utf-8") )
        output_a_list.append( open(output_dir+"benchmark-"+str(i)+"/assertLines.txt", "w", encoding="utf-8") )
    '''
    if(if_write):
        output_b0_m = open(output_dir+"benchmark-0/testMethods.txt", "w", encoding="utf-8")
        output_b0_a = open(output_dir+"benchmark-0/assertLines.txt", "w", encoding="utf-8")
        output_b1_m = open(output_dir+"benchmark-1/testMethods.txt", "w", encoding="utf-8")
        output_b1_a = open(output_dir+"benchmark-1/assertLines.txt", "w", encoding="utf-8")
        output_b2_m = open(output_dir+"benchmark-2/testMethods.txt", "w", encoding="utf-8")
        output_b2_a = open(output_dir+"benchmark-2/assertLines.txt", "w", encoding="utf-8")
        output_b3_m = open(output_dir+"benchmark-3/testMethods.txt", "w", encoding="utf-8")
        output_b3_a = open(output_dir+"benchmark-3/assertLines.txt", "w", encoding="utf-8")
        output_b4_m = open(output_dir+"benchmark-4/testMethods.txt", "w", encoding="utf-8")
        output_b4_a = open(output_dir+"benchmark-4/assertLines.txt", "w", encoding="utf-8")
        output_b5_m = open(output_dir+"benchmark-5/testMethods.txt", "w", encoding="utf-8")
        output_b5_a = open(output_dir+"benchmark-5/assertLines.txt", "w", encoding="utf-8")
        output_b6_m = open(output_dir+"benchmark-6/testMethods.txt", "w", encoding="utf-8")
        output_b6_a = open(output_dir+"benchmark-6/assertLines.txt", "w", encoding="utf-8")
        output_b7_m = open(output_dir+"benchmark-7/testMethods.txt", "w", encoding="utf-8")
        output_b7_a = open(output_dir+"benchmark-7/assertLines.txt", "w", encoding="utf-8")

    if(if_write):
        output_b8_m = open(output_dir+"benchmark-8/testMethods.txt", "w", encoding="utf-8")
        output_b8_a = open(output_dir+"benchmark-8/assertLines.txt", "w", encoding="utf-8")
        output_b9_m = open(output_dir+"benchmark-9/testMethods.txt", "w", encoding="utf-8")
        output_b9_a = open(output_dir+"benchmark-9/assertLines.txt", "w", encoding="utf-8")


    for i in range(total_num):
        method = methods[i]
        assertion = asserts[i]
        match = split_re.search(method)
   
        #focal method (in ATLAS) = ""
        if not match:
            if(method != ""):
                test_method_assert_delete = method
                test_method = method.replace("\"<AssertPlaceHolder>\"", assertion) 
                #test_method = method.replace("\"<AssertPlaceHolder>\"", "")
                focal_method_atlas_name = ""
            else:
                continue
        else:
            idx = match.span()[1]
            test_method_assert_delete = method[0:idx]
            test_method = method[0:idx]
            focal_method_atlas_name = method[idx:]
            index = focal_method_atlas_name.find("(")
            focal_method_atlas_name = focal_method_atlas_name[0:index].strip()
            test_method = test_method.replace("\"<AssertPlaceHolder>\"", assertion)
            #test_method = test_method.replace("\"<AssertPlaceHolder>\"", "") 

        idx = test_method.find("(")
        test_method_name = test_method[0:idx].strip()

        tmp_class = "public class aTest{ @Test public void " + test_method + " }"
        try:
            tree = javalang.parse.parse(tmp_class.replace(" . ","."))
        except:
            if(record_failed_cases):
                javalang_failed.write(tmp_class+"\n")
            continue

        #get test prefix
        #find ";"
        beg = tmp_class.find(assertion)
        end = tmp_class.find(";", beg+len(assertion))
        tmp_class = tmp_class[0:beg] + tmp_class[end:]

        tmp_class = tmp_class.replace(" . ",".")
        focal_method_atlas, Highest_methods = identify(test_method_name, tmp_class, focal_method_atlas_name)

        #parse failed
        if(focal_method_atlas=="" and Highest_methods=={}):
            continue
        
        if(if_write):
            #benchmark-0: atlas focal method (last non-junit method call)
            #benchmark-1: last call before assert
            #benchmark-2: NC
            #benchmark-3: NCC
            #benchmark-4: LCS_B
            #benchmark-5: LCS_U
            #benchmark-6: ED
            #benchmark-7: Combined
            #technique_list = ["NC", "NCC", "LCS_B", "LCS_U", "Score_Edit", "LastCall", "LastCallBeforeAssert","Avg"]
            output_b0_m.write(test_method_assert_delete + " " + focal_method_atlas +"\n")
            output_b1_m.write(test_method_assert_delete + " " + Highest_methods["LastCallBeforeAssert"] +"\n")
            output_b2_m.write(test_method_assert_delete + " " + Highest_methods["NC"] +"\n")
            output_b3_m.write(test_method_assert_delete + " " + Highest_methods["NCC"] +"\n")
            output_b4_m.write(test_method_assert_delete + " " + Highest_methods["LCS_B"] +"\n")
            output_b5_m.write(test_method_assert_delete + " " + Highest_methods["LCS_U"] +"\n")
            output_b6_m.write(test_method_assert_delete + " " + Highest_methods["Score_Edit"] +"\n")
            output_b7_m.write(test_method_assert_delete + " " + Highest_methods["Avg"] +"\n")

        if(Highest_methods["NC"] != ""):
            #print(focal_method_atlas, Highest_methods["NC"])
            #if(focal_method_atlas.strip() != Highest_methods["NC"].strip()):
                #number = number + 1
            if(if_write):
                output_b8_m.write(test_method_assert_delete + " " + Highest_methods["NC"] +"\n")
                output_b9_m.write(method +"\n")

        if(if_write):
            output_b0_a.write(assertion + "\n")
            output_b1_a.write(assertion + "\n")
            output_b2_a.write(assertion + "\n")
            output_b3_a.write(assertion + "\n")
            output_b4_a.write(assertion + "\n")
            output_b5_a.write(assertion + "\n")
            output_b6_a.write(assertion + "\n")
            output_b7_a.write(assertion + "\n")

        if(Highest_methods["NC"] != ""):
            if(if_write):
                output_b8_a.write(assertion + "\n")
                output_b9_a.write(assertion + "\n")

    if(if_write):
        output_b0_m.close()
        output_b0_a.close()
        output_b1_m.close()
        output_b1_a.close()
        output_b2_m.close()
        output_b2_a.close()
        output_b3_m.close()
        output_b3_a.close()
        output_b4_m.close()
        output_b4_a.close()
        output_b5_m.close()
        output_b5_a.close()
        output_b6_m.close()
        output_b6_a.close()
        output_b7_m.close()
        output_b7_a.close()

        output_b8_m.close()
        output_b8_a.close()
        output_b9_m.close()
        output_b9_a.close()

if __name__ == "__main__":

    method_path = Testing_Path + "testMethods.txt"
    assert_path = Testing_Path + "assertLines.txt"

    print(method_path, assert_path)
    improve_focal(method_path, assert_path)
    print(failed)
    print(number)

    if(record_failed_cases):
        javalang_failed.close()
