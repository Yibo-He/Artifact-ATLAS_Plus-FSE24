# Artifact-FocalMethodStudy
Artifact of "An Empirical Study on Focal Methods in Deep-Learning-Based Approaches for Assertion Generation".

The datasets used in our experiments can be found at:
https://zenodo.org/records/11312096


## Datasets
- TCTracer: Used to evaluate adapted test-to-code traceability techniques (RQ1)
- ATLAS+: Used to evaluate DL-based assertion generation approaches (RQ2 & RQ3)

### TCTracer
Original TCTracer is available at https://zenodo.org/record/4608587#.ZClJly-9FAc. We split test methods in TCTracer using ```Scripts/split_tctracer.py``` and check the synatx of splitted test methods manually. Splited TCTracer is shown in ```Datasets/TCTracer```.

There are four subjects in TCTracer: commons-io, commons-lang, gson, and jfreechart, and their source code can be found in ```Datasets/TCTracer/subjects```. In each directory (named after the project name, e.g. ```Datasets/TCTracer/subjects/commons-io```), splitted test methods are put in the directories, and unsplit test methods are put in the txt files.

### ATLAS+
ATLAS+ can be found at https://zenodo.org/records/11312096.

Generated by ```Scripts/dataset_variation.py``` and ```Scripts/dataset_variation_new.py``` based on the ATLAS dataset(https://sites.google.com/view/atlas-nmt/home). All entries in ATLAS with syntax errors are removed.

4 Training sets:

- TS-atlas: focal methods without implementation identified by atlas
- TS-null: without focal methods 
- TS-lcba: focal methods without implementation identified by LCBA
- TS-combined: focal methods without implementation identified by combined score

11 Testing sets:
- benchmark-0-atlas: focal methods identified by ATLAS
- benchmark-0-atlas-with-impl: focal methods identified by ATLAS (with implementation)
- benchmark-1-lcba: focal methods identified by LCBA
- benchmark-2-nc: focal methods identified by NC
- benchmark-3-ncc: focal methods identified by NCC
- benchmark-4-lcsb: focal methods identified by LCS_B
- benchmark-5-lcsu: focal methods identified by LCS_U
- benchmark-6-ed: focal methods identified by ED
- benchmark-7-combined: focal methods identified by CS
- benchmark-8-nc-432: successful cases with focal methods identified by NC
- benchmark-9-atlas-432:  control group of benchmark-8


Besides, variant datasets for TOGA (ATLAS star) are generated by ```Scripts/atlas_star_datagen.py```.


## Evaluation
### Test-to-code traceability techniques
Run  ```Scripts/eval_traceability_tech.py```.

### ATLAS
Run a docker container provided by nvidia for tensorflow. More details in https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow .
```
docker pull nvcr.io/nvidia/tensorflow:xx.xx-tf1-py3
docker run -i -d -t --gpus all nvcr.io/nvidia/tensorflow:xx.xx-tf1-py3
```

Training and inference: https://sites.google.com/view/atlas-nmt/home


### T5
Use our script```Scripts/generate_results_T5.py```.

Datasets: ```Variant_ATLAS_for_T5```

Models and other detials: https://github.com/antonio-mastropaolo/T5-learning-ICSE_2021

### IR
Use script ```Retrieval/IR.py``` in https://github.com/yh1105/Artifact-of-Assertion-ICSE22

### TOGA
Datasets: ```Variant_ATLAS_for_TOGA```

We use the docker comtainer: ```docker pull edinella/toga-artifact```. More detials at https://github.com/microsoft/toga

Fine-tuning script: ```toga/model/assertions/run_train.sh```

### Results
Get correct number: ```Scripts/inference_accuracy.py```, 
e.g.:
```
python3 inference_accuracy.py -atlas xxxx/assertLines.txt xxxx/prediction.txt
python3 inference_accuracy.py -t5 xxxx/test.tsv xxxx/prediction.txt
python3 inference_accuracy.py -toga xxxx/test.csv xxxx/prediction.csv
```

Get BLEU: ```multi-bleu.perl```, 
e.g.:

```
./multi-bleu.perl xxxx/assertLines.txt < xxxx/prediction.txt
```

Note for T5, after finetuning a new model, the script ```fine_tuning_t5.ipynb``` on google colab can return the accuracy on the corresponding test set.
