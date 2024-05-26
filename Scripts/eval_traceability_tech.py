from types import NoneType
import javalang
import math
import traceback
import os
import pandas as pd
import csv

#Evaluate approaches to identify focal methods:
    #LC (prefix + assertion)
    #LCBA (prefix)
    #NC NCC
    #Name Similarity(LCS-B LCS-U Edit-distance)
    #Combined

#please set tctracer_path and csv_dir before running
tctracer_path = "..../ICSE24-Artifact-FocalMethodStudy/Datasets/TCTracer/"
csv_dir = tctracer_path + "traceability-ground-truth/"
if_print = True

data = {"tm":[], "ground truth fm":[], "NC":[],"NCC":[], "LCS_B":[], "LCS_U":[], "ED":[], "LC":[], "LCBA":[],"Combined":[]}
df = pd.DataFrame(data)

subject_dir = "..../subjects/"
project_list = ["commons-io", "commons-lang", "gson", "jfreechart"]
test_dir_path = {"commons-io":subject_dir+"commons-io-commons-io-2.5/src/test/java/", \
"commons-lang":subject_dir+"commons-lang-LANG_3_7/src/test/java/", \
"gson":subject_dir+"gson-gson-parent-2.8.0/gson/src/test/java/", \
"jfreechart":subject_dir+"jfreechart-1.0.19/tests/" \
}

class_dir_path = {"commons-io":subject_dir+"commons-io-commons-io-2.5/src/main/java/", \
"commons-lang":subject_dir+"commons-lang-LANG_3_7/src/main/java/", \
"gson":subject_dir+"gson-gson-parent-2.8.0/gson/src/main/java/", \
"jfreechart":subject_dir+"jfreechart-1.0.19/source/" \
}

sum = 0
technique_list = ["NC", "NCC", "LCS_B", "LCS_U", "Score_Edit", "LastCall", "LastCallBeforeAssert","Avg"]
TP = {} #return true, positive ground truth
FP = {} #return false, positive ground truth
Failed = {} #fail to find an focal method, which means all of the alternative methods get score 0
Recall_high = {} #number of the identified focal methods 
visited = {} #list of the identified focal methods
Recall_low = 0 # total number of methods labeled as focal methods
for tech in technique_list:
    TP[tech] = 0
    FP[tech] = 0
    Failed[tech] = 0
    Recall_high[tech] = 0
    visited[tech] = []

weights = {"NC":1.0, "NCC":0.375, "LCS_B":0.639, "LCS_U":0.768, "Score_Edit":0.694, "LastCall":0.607, "LastCallBeforeAssert":0.435}
#weights = {"NC":1.0, "NCC":1.0, "LCS_B":1.0, "LCS_U":1.0, "Score_Edit":1.0, "LastCall":1.0, "LastCallBeforeAssert":1.0}
weight_sum = 0
for tech in weights.keys():
    weight_sum += weights[tech]

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

#Name Similarity
#return LCS-B, LCS-U, 1-Edit-distance
def Name_Similarity(testMethod, method):
    len_nt = len(testMethod)
    len_nf = len(method)
    lcs = LCS(testMethod, method)
    LCS_B = lcs*1.0/(max(len_nf, len_nt))
    LCS_U = lcs*1.0/len_nf
    Score_Edit = 1 - EditDistance(testMethod, method)*1.0/(max(len_nf, len_nt))
    return LCS_B, LCS_U, Score_Edit

#input path of the test class
#get the list of test method names
def getTestMethodNames(test_file_path):
    test_methods = []
    with open(test_file_path, "r", encoding="utf-8") as f:
        tree = javalang.parse.parse(f.read())
        for path, node in tree:
            try:
                if(type(node) == javalang.tree.MethodDeclaration and node.annotations[0].name == "Test"):
                    test_methods.append(node.name)
            except:
                continue
    return test_methods

#input path of the tested class
#get the list of public method names
def getTestedPublicMethodNames(tested_file_path):
    typeWeWant = [javalang.tree.ConstructorDeclaration, javalang.tree.MethodDeclaration]
    public_methods = []
    with open(tested_file_path, "r", encoding="utf-8") as f:
        tree = javalang.parse.parse(f.read())
        for path, node in tree:
            try:
                if(type(node) in typeWeWant and "public" in node.modifiers):
                    public_methods.append(node.name)
            except:
                continue

    return public_methods

#return the list of the method calls using deep first search(DFS)
#if the para ignore_assertion=1, assert statements will be ignored.
def getExecutingMethods(method_name, file_path, ignore_assertion=0, ifdebug=0):
    global waitlist
    ExecutingMethods = []
    typeDecl = [javalang.tree.ConstructorDeclaration, javalang.tree.MethodDeclaration]
    typeInvo = [javalang.tree.MethodInvocation, javalang.tree.SuperMethodInvocation]
    assertion_list = ["assertArrayEquals", "assertEquals", "assertFalse", "assertNotEquals", \
    "assertNotNull", "assertNotSame", "assertNull", "assertSame", "assertThat", "assertThrows", "assertTrue"]
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        try:
            tree = javalang.parse.parse(content)
        except:
            try:
                tree = javalang.parse.parse("public class aTest{\n"+content+"}\n")
            except:
                if(ifdebug):
                    print("syntax error!")
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
                            ExecutingMethods.append(tmp.member)
                        if(type(tmp) == javalang.tree.ClassCreator):
                            #ExecutingMethods.append(tmp.type.name)
                            tmp_tmp_node = tmp.type
                            while(tmp_tmp_node.sub_type != None):
                                tmp_tmp_node = tmp_tmp_node.sub_type
                            str_ = str(tmp_tmp_node.name)
                            ExecutingMethods.append(str_)
                        try:
                            subNodes = []
                            for attr in tmp.attrs:
                                subNode = getattr(tmp, attr, None)
                                #print(attr, type(subNode),subNode)
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

#get last method call
#return last method call before both test prefix + assertion and test prefix
def getLastMethodCall(test_method, test_method_path):
    ExecutingMethodsWithAssert = getExecutingMethods(test_method, test_method_path)
    ExecutingMethodsWithoutAssert = getExecutingMethods(test_method, test_method_path, 1)
    return ExecutingMethodsWithAssert[-1], ExecutingMethodsWithoutAssert[-1]

'''
#TF-IDF
#version 1
def TF_IDF(testMethod, method, test_file_path, tested_file_path):
    T = getTestMethodNames(test_file_path) #T is the set of all tests in the test suite
    F = getTestedPublicMethodNames(tested_file_path) #F is the set of all public methods in the tested class
    test2Exec = {}
    for test in T:
        ExecutingMethods = getExecutingMethods(test, test_file_path)
        test2Exec[test] = ExecutingMethods
    
    #calculate tf
    publicMethodExec = 0
    for f in F:
        if f in test2Exec[testMethod]:
            publicMethodExec += 1
    try:
        tf = math.log( 1.0+1.0/publicMethodExec) #the more public methods this test tests, the lower this value.
    except ZeroDivisionError:
        print("No public method in ExecutingMethods list!")
        print("test method:", testMethod)
        print("test file path: ", test_file_path)
        print("tested file path: ", tested_file_path)
        print()
        tf = 0
    
    #calculate idf
    testmethodContainMethod = 0
    for t in T:
        if method in test2Exec[t]:
            testmethodContainMethod += 1
    try:
        idf = math.log( 1.0+1.0*len(T)/testmethodContainMethod) #the more test methods contain this "focal method", the lower this value.
    except ZeroDivisionError:
        print("No test method contain focal method!")
        print("public method:", method)
        print("test file path: ", test_file_path)
        print("tested file path: ", tested_file_path)
        print()
        idf = 0

    #calculate tf-idf
    tf_idf = tf * idf
    return tf_idf

#TF-IDF
#version2: Increase execution speed
def TF_IDF_2(tf, method, T, F, test2Exec):    
  
    #calculate idf
    testmethodContainMethod = 0
    for t in T:
        if method in test2Exec[t]:
            testmethodContainMethod += 1
    try:
        idf = math.log( 1.0+1.0*len(T)/testmethodContainMethod) #the more test methods contain this "focal method", the lower this value.
    except ZeroDivisionError:
        idf = 0

    #calculate tf-idf
    tf_idf = tf * idf
    return tf_idf
'''

def run_method_level_link(test_method, test_method_class, focal_method_list, tested_class_list = ""):

    assertion_list = ["assertArrayEquals", "assertEquals", "assertFalse", "assertNotEquals", \
    "assertNotNull", "assertNotSame", "assertNull", "assertSame", "assertThat", "assertThrows", "assertTrue"]

    Highest_score = {}
    Highest_method_name = {}
    for tech in technique_list:
        Highest_score[tech] = 0
        Highest_method_name[tech] = ""
    
    ExecutingMethodsWithAssert = getExecutingMethods(test_method, test_method_class)
    ExecutingMethodsWithoutAssert = getExecutingMethods(test_method, test_method_class, 1)

    #syntax error
    if(len(ExecutingMethodsWithAssert)==1 and ExecutingMethodsWithAssert[0]=="-1"):
        return -1

    try:
        for m in reversed(ExecutingMethodsWithAssert):
            if(m not in assertion_list):
                Highest_score["LastCall"] = 1
                Highest_method_name["LastCall"] = m
                break
    except:
        Highest_score["LastCall"] = 0
        Highest_method_name["LastCall"] = ""

    try:
        Highest_score["LastCallBeforeAssert"] = 1
        Highest_method_name["LastCallBeforeAssert"] = ExecutingMethodsWithoutAssert[-1]
    except:
        Highest_score["LastCallBeforeAssert"] = 0
        Highest_method_name["LastCallBeforeAssert"] = ""
    
    #for public_method in F:
    for public_method in ExecutingMethodsWithAssert:
        NC, NCC = NCAndNCC(test_method, public_method)
        LCS_B, LCS_U, Score_Edit = Name_Similarity(test_method, public_method)
        Avg = weights["NC"]*NC + weights["NCC"]*NCC + weights["LCS_B"]*LCS_B + weights["LCS_U"]*LCS_B + weights["Score_Edit"]*Score_Edit
        if(public_method == Highest_method_name["LastCall"]):
            Avg += weights["LastCall"]*1
        if(public_method == Highest_method_name["LastCallBeforeAssert"]):
            Avg += weights["LastCallBeforeAssert"]*1
        Avg /= weight_sum

        #update NC & NCC
        score = Highest_score["NC"]
        if(NC>=score):
            Highest_score["NC"] = NC
            Highest_method_name["NC"] = public_method

        score = Highest_score["NCC"]
        if(NCC>=score):
            Highest_score["NCC"] = NCC
            Highest_method_name["NCC"] = public_method
        
        #update LCS_B, LCS_U, Score_Edit
        score = Highest_score["LCS_B"]
        if(LCS_B>=score):
            Highest_score["LCS_B"] = LCS_B
            Highest_method_name["LCS_B"] = public_method

        score = Highest_score["LCS_U"]
        if(LCS_U>=score):
            Highest_score["LCS_U"] = LCS_U
            Highest_method_name["LCS_U"] = public_method

        score = Highest_score["Score_Edit"]
        if(Score_Edit>=score):
            Highest_score["Score_Edit"] = Score_Edit
            Highest_method_name["Score_Edit"] = public_method
        
        #update Avg
        score = Highest_score["Avg"]
        if(Avg>=score):
            Highest_score["Avg"] = Avg
            Highest_method_name["Avg"] = public_method
    
    new_data = {"tm":test_method_class.split("/")[-1], "ground truth fm":focal_method_list, \
                "NC":(Highest_method_name["NC"],Highest_score["NC"]), \
                "NCC":(Highest_method_name["NCC"],Highest_score["NCC"]), \
                "LCS_B":(Highest_method_name["LCS_B"],Highest_score["LCS_B"]), \
                "LCS_U":(Highest_method_name["LCS_U"],Highest_score["LCS_U"]), \
                "ED":(Highest_method_name["Score_Edit"],Highest_score["Score_Edit"]), \
                "LC":(Highest_method_name["LastCall"],Highest_score["LastCall"]), \
                "LCBA":(Highest_method_name["LastCallBeforeAssert"],Highest_score["LastCallBeforeAssert"]), \
                "Combined":(Highest_method_name["Avg"],Highest_score["Avg"])}
    df = df.append(new_data, ignore_index=True)

    if(if_print):
        for key in Highest_score.keys():
            method, score = Highest_method_name[key],Highest_score[key]
            print(key, method, score)
        print()

    for key in Highest_score.keys():
        method, score = Highest_method_name[key],Highest_score[key]
        if(score>0 and method in focal_method_list):
            TP[key] += 1
            visited[key].append(method)
        elif(score>0 and method not in focal_method_list):
            FP[key] += 1
        elif(score <=0):
            Failed[key] += 1
        '''
        elif(score==0 and method in focal_method_list):
            FN[key] += 1
        elif(score==0 and method not in focal_method_list):
            TN[key] += 1
        '''
    return 0

if __name__ == "__main__":
    test2tested_method = {}
    keys = []
    for project in project_list:
        csvfile = open(csv_dir+project+".csv", newline='')
        reader = csv.DictReader(csvfile)
        for data in reader:
            test_fqn = data["test-fqn"]
            tested_method_fqn = data["tested-method-fqn"]
            keys.append((project, test_fqn))

            #get ground truth: focal method
            list_tested_method_fqn = tested_method_fqn.split('.')
            index= list_tested_method_fqn[-1].find("(")
            focal_method = list_tested_method_fqn[-1][0:index]
            
            if(test_fqn not in test2tested_method.keys()):
                test2tested_method[test_fqn] = []
                test2tested_method[test_fqn].append(focal_method)
            else:
                test2tested_method[test_fqn].append(focal_method)
                test2tested_method[test_fqn] = list(set(test2tested_method[test_fqn]))

        csvfile.close()
    
    keys = list(dict.fromkeys(keys))
    for project, test_fqn in keys:
        proj_path = tctracer_path + project
        new_dir_name = test_fqn.replace("()", "")
        new_dir_name = new_dir_name.replace(".", "-")
        dir_path = proj_path+"/"+new_dir_name
        
        for key in visited.keys():
            visited[key] = []

        test_methods_splited = os.listdir(dir_path)
        for file in test_methods_splited:
            test_method = file.split(".")[-3]
            if(if_print):
                print(test_method, dir_path+"/"+file, test2tested_method[test_fqn])
            ans = run_method_level_link(test_method, dir_path+"/"+file, test2tested_method[test_fqn])
            if(ans == 0):
                sum += 1  

        Recall_low += len(test2tested_method[test_fqn])
        for key in visited.keys():
            visited[key] = list(set(visited[key]))
            Recall_high[key] += len(visited[key])
    
    for tech in technique_list:
        acc = TP[tech]/(TP[tech]+FP[tech]+Failed[tech])*100
        prec = TP[tech]/(TP[tech]+FP[tech])*100
        recall = Recall_high[tech]/Recall_low*100
        F1 = 2*prec*recall/(prec+recall)
        print(tech+" acc: {}%".format(acc))
        print(tech+" prec: {}%".format(prec))
        print(tech+" recall: {}%".format(recall))
        print(tech+" F1: {}%".format(F1))
        print()
    
    print("sum : {}".format(sum))
    df.to_csv(tctracer_path + 'rq1.csv', index=True)
