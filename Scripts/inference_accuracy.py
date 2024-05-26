import sys
import pandas as pd
import csv

def ATLAS(assertLineFilePath, predictionFilePath):
    #print(assertLineFilePath, predictionFilePath)
        
    count_perfect = 0
    count_imperfect = 0

    assertLineFile = open(assertLineFilePath, 'r', encoding="utf8")
    predictionFile = open(predictionFilePath, 'r', encoding="utf8")
    
    for line in assertLineFile:
        assertStatement = line.strip()
        predictionLine = predictionFile.readline().strip()
        
        if(assertStatement == predictionLine):
            count_perfect += 1

        else:
            count_imperfect += 1

    assertLineFile.close()
    predictionFile.close()

    print(count_perfect)
    return count_perfect, count_imperfect

def T5(assertLineFilePath, predictionFilePath):
    
    count_perfect = 0
    count_imperfect = 0

    df = pd.read_csv(assertLineFilePath, header=None, sep='\t')
    references=[]
    for item in df[1]:
        assertion = item.lower()
        #assertion = assertion.replace("\"\"", "\"")
        assertion = ''.join(assertion.strip().split(' '))
        references.append(assertion)
    references[1]

    predictionFile = open(predictionFilePath, 'r', encoding="utf8")
    predictionList = predictionFile.readlines()
    
    len_prediction = len(predictionList)

    for idx in range(0, len_prediction):
        prediction = ''.join(predictionList[idx].strip().split(' '))
        if(references[idx] == prediction):
            count_perfect += 1
        else:
            count_imperfect += 1

    predictionFile.close()

    print(count_perfect)
    return count_perfect, count_imperfect

def IR(assertLineFilePath, predictionFilePath):
    return ATLAS(assertLineFilePath, predictionFilePath)

MAG = 'threshold_magnitude'
DIFF = 'threshold_diff'
WEIGHTED = 'threshold_weighted'
IGNORE_MODEL = 'threshold_ignore'
def get_worst_idx(scores, method):
    _min = None
    _min_idx = -1
    for idx, (l0, l1) in enumerate(scores):
        if method == MAG:
            m = l0
        elif method == DIFF:
            m = abs(l0-l1)

        if _min == None or m < _min:
            _min = m
            _min_idx = idx
    return _min_idx

def get_best_idx(scores, max_method):
    _max = None
    _max_idx = -1
    for idx, (l0, l1) in enumerate(scores):
        if not l0: continue
        if max_method == MAG:
            m = l1
        elif max_method == DIFF:
            m = abs(l0-l1)
        if _max == None or m > _max:
            _max = m
            _max_idx = idx
    return _max_idx

def TOGA(assertLineFilePath, predictionFilePath):
    foo = {}
    lines = []
    with open(predictionFilePath) as f:
        reader = csv.reader(f) 
        for row in reader:
            lines += [row]

    lines = lines[1:] # drop header

    for line in lines:
        if not line: continue

        test_num, t, p, l0, l1, test_name, assertion, test, fm = line
        l0, l1 = float(l0), float(l1)
        t, p = int(t), int(p)

        if not test_num in foo:
            foo[test_num] = []

        foo[test_num].append((t, p, l0, l1, test_name, assertion, test, fm))
    
    cor, incor = 0,0

    for test_num, values in foo.items():
        p_1s = [v[1] for v in values]
        scores = [(v[2], v[3]) for v in values]
        t_s = [v[0] for v in values]
        assertions = [v[5] for v in values]
        test = values[0][6]
        fm = values[0][7]

        test_name = values[0][4]

        aggregate_p = None
        if sum(p_1s) == 1:
            aggregate_p = p_1s.index(1)

        elif sum(p_1s) == 0:
            aggregate_p = get_worst_idx(scores, MAG)

        else: # greater than one 1 predicted
            scores_1 = []
            for p, s in zip(p_1s, scores):
                if p:
                    scores_1.append(s)
                else:
                    scores_1.append((None, None))
            aggregate_p = get_best_idx(scores_1, MAG)

        new_ps = [0] * len(values)
        new_ps[aggregate_p] = 1

        correct = True
        for new_p, t in zip(new_ps, t_s):
            if new_p != t:
                correct = False

        if correct:
            cor += 1
        else:
            incor += 1

    print(cor)
    return cor, incor

if __name__=="__main__":
    try:
        method = sys.argv[1]

        if(method == "-atlas" or method == "-ATLAS"):
            ATLAS(sys.argv[2], sys.argv[3])
        elif(method == "-t5" or method == "-T5"):
            if("tsv" not in sys.argv[2]):
                print("Error! T5's ground truth file should be in tsv format.")
            else:
                T5(sys.argv[2], sys.argv[3])
        elif(method == "-ir" or method == "-IR"):
            IR(sys.argv[2], sys.argv[3])
        elif(method == "-toga" or method == "-TOGA"):
            if("csv" not in sys.argv[2]):
                print("Error! TOGA's ground truth file should be in csv format.")
            else:
                TOGA(sys.argv[2], sys.argv[3])
        else:
            print("Count the number of correct numbers and accuracy")
            print("Usage:")
            print("-[approach] [ground_truth_assertion_file] [prediction_assertion_file]")
            print("approach in [ATLAS, atlas, T5, t5, IR, ir, TOGA, toga]")

    except:
        print("Count the number of correct numbers and accuracy")
        print("Usage:")
        print("-[approach] [ground_truth_assertion_file] [prediction_assertion_file]")
        print("approach in [ATLAS, atlas, T5, t5, IR, ir, TOGA, toga]")
