import re
from types import NoneType
import javalang
import traceback

split_re = re.compile("\"<AssertPlaceHolder>\" .*?;( })+")

ATLAS_Path = "..../Datasets"

technique_list = ["NC", "NCC", "LCS_B", "LCS_U", "Score_Edit", "LastCall", "LastCallBeforeAssert","Avg"]
weights = {"NC":1.0, "NCC":0.375, "LCS_B":0.639, "LCS_U":0.768, "Score_Edit":0.694, "LastCall":0.607, "LastCallBeforeAssert":0.435}
weight_sum = 0
for tech in weights.keys():
    weight_sum += weights[tech]

assertion_list = ["assertArrayEquals", "assertEquals", "assertFalse", "assertNotEquals", \
    "assertNotNull", "assertNotSame", "assertNull", "assertSame", "assertThat", "assertThrows", "assertTrue"]

num_focal_atlas_in = 0
num_focal_combined_in = 0

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
    #print((test_method_name, test_class, focal_method_atlas_name))
    Highest_score = {}
    Highest_method_idx = {}
    for tech in technique_list:
        Highest_score[tech] = 0
        Highest_method_idx[tech] = -1
    
    ExecutingMethodsWithAssert = getExecutingMethods(test_method_name, test_class)
    if(len(ExecutingMethodsWithAssert)==1 and ExecutingMethodsWithAssert[0]=="-1"):
        return "-1", "-1", ""

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

    if(Highest_method_idx["LastCallBeforeAssert"]!=-1):
        last_call_before_assert = ExecutingMethodsWithAssert[Highest_method_idx["LastCallBeforeAssert"]]
    else:
        last_call_before_assert = ""
    
    focal_method = ExecutingMethodsWithAssert[Highest_method_idx["Avg"]]

    return last_call_before_assert, focal_method, focal_method_atlas

#delete entries with syntax error, and the implementation of focal method
#dataset_mode = 0, (test prefix + last non-JUnit-framework-API method, assertion) (DS-atlas)
#dataset_mode = 1, (test prefix, assertion) (DS-null)
#dataset_mode = 2, (test prefix + last call before assertion, assertion) (DS-lcba)
#dataset_mode = 3, (test prefix + accuracy focal method, assertion) (DS-combined)
#dataset_mode = 4, (test prefix + last non-JUnit-framework-API method, assertion) (ATLAS, removing syntax error entries)
def improve_focal(testMethodsPath, assertPath, outputMethodPath, outputAssertPath, dataset_mode):
    #global sum_case, lcba, our_focal
    global num_focal_atlas_in, num_focal_combined_in
    methods = open(testMethodsPath).read().split("\n")
    asserts = open(assertPath).read().split("\n")

    total_num = len(asserts)
    if len(asserts) != len(methods):
        print("method line and assertion line not match!")
        return

    if(dataset_mode==0 or dataset_mode==1 or dataset_mode==2 or dataset_mode==3 or dataset_mode==4):
        output_method = open(outputMethodPath, "w", encoding="utf-8")
        output_assert = open(outputAssertPath, "w", encoding="utf-8")

    for i in range(total_num):
        method = methods[i]
        assertion = asserts[i]
        match = split_re.search(method)
   
        if not match:
            if(method != ""):
                test_method_assert_delete = method
                test_method = method.replace("\"<AssertPlaceHolder>\"", assertion) 
                focal_method_atlas_name = ""
                focal_method_atlas_2 = ""
            else:
                continue
        else:
            idx = match.span()[1]
            test_method_assert_delete = method[0:idx]
            test_method = method[0:idx]
            focal_method_atlas_name = method[idx:]
            focal_method_atlas_2 = method[idx:]
            index = focal_method_atlas_name.find("(")
            focal_method_atlas_name = focal_method_atlas_name[0:index].strip()

            test_method = test_method.replace("\"<AssertPlaceHolder>\"", assertion) 

        idx = test_method.find("(")
        test_method_name = test_method[0:idx].strip()

        tmp_class = "public class aTest{ @Test public void " + test_method + " }"
        tmp_class = tmp_class.replace(" . ",".")

        last_call_before_assert, focal_method, focal_method_atlas = identify(test_method_name, tmp_class, focal_method_atlas_name)
        #print(last_call_before_assert, focal_method, focal_method_atlas)

        if(last_call_before_assert=="-1" and focal_method == "-1"):
            '''
            if(dataset_mode==-1):
                print(testMethodsPath,i)
                print(method)
                print(assertion)
                print()
                print()
            '''
            continue
        
        if(dataset_mode==0):
            output_method.write(test_method_assert_delete + " " + focal_method_atlas +"\n")
            output_assert.write(assertion + "\n")
        elif(dataset_mode==1):
            output_method.write(test_method_assert_delete + "\n")
            output_assert.write(assertion + "\n")
        elif(dataset_mode==2):
            output_method.write(test_method_assert_delete + " " + last_call_before_assert +"\n")
            output_assert.write(assertion + "\n")
        elif(dataset_mode==3):
            output_method.write(test_method_assert_delete + " " + focal_method +"\n")
            output_assert.write(assertion + "\n")
        elif(dataset_mode==4):
            output_method.write(test_method_assert_delete + " " + focal_method_atlas_2 +"\n")
            output_assert.write(assertion + "\n")
        else:
            if(focal_method_atlas_name in assertion):
                num_focal_atlas_in+=1
            if(focal_method[0: focal_method.find("(") ].strip() in assertion):
                num_focal_combined_in+=1
            continue

    if(dataset_mode==0 or dataset_mode==1 or dataset_mode==2 or dataset_mode==3 or dataset_mode==4):
        output_method.close()
        output_assert.close()

if __name__ == "__main__":

    '''
    input_list = ["./Datasets/Raw_Dataset/Eval/", 
    "./Datasets/Raw_Dataset/Testing/", 
    "./Datasets/Raw_Dataset/Training/", 
    "./Datasets/Abstract_Dataset/Eval/", 
    "./Datasets/Abstract_Dataset/Testing/", 
    "./Datasets/Abstract_Dataset/Training/"]
    '''

    input_list = [ATLAS_Path + "/Raw_Dataset/Eval/", 
    ATLAS_Path + "/Raw_Dataset/Testing/", 
    ATLAS_Path + "/Raw_Dataset/Training/"]

    for i in range(0,5):
        for input_path in input_list:
            output_path = input_path.replace("/Datasets", "/VariantDataset/Datasets"+str(i))
            method_path = input_path + "testMethods.txt"
            assert_path = input_path + "assertLines.txt"
            output_method = output_path + "testMethods.txt"
            output_assert = output_path + "assertLines.txt"
            print(method_path, assert_path, output_method, output_assert, i)
            improve_focal(method_path, assert_path, output_method, output_assert, i)

            #print(num_focal_atlas_in)
            #print(num_focal_combined_in)
