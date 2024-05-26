import os

tctracer_path = "ICSE24-Artifact-FocalMethodStudy/Datasets/TCTracer/"

project_list = ["commons-io", "commons-lang", "gson", "jfreechart"]
assertion_list = ["assertArrayEquals", "assertEquals", "assertFalse", "assertNotEquals", \
"assertNotNull", "assertNotSame", "assertNull", "assertSame", "assertThat", "assertThrows", "assertTrue"]

total = 0
for project in project_list:
    proj_path = tctracer_path + project
    file_name_list = os.listdir(proj_path)
    for file in file_name_list:
        file_path = proj_path+"/"+file
        if(".txt" not in file_path):
            continue
        new_dir_name = file.replace(".txt", "")
        new_dir_name = new_dir_name.replace(".", "-")
        dir_path = proj_path+"/"+new_dir_name
        try:
            os.mkdir(dir_path)
        except FileExistsError:
            ()
        with open(file_path, "r", encoding="utf-8") as test_method:
            test_prefix = []
            lines = test_method.readlines()
            file_num = 0
            depth = 0
            assert_num = 0
            for line in lines:
                is_assert = False
                for assertion in assertion_list:
                    if(assertion in line):
                        is_assert = True
                        break
                depth += line.count("{")
                depth -= line.count("}")
                if(is_assert):
                    with open(dir_path+"/"+file.replace("txt", str(file_num))+".txt", "w", encoding="utf-8") as output:
                        output.write("public class aTest{\n")
                        for item in test_prefix:
                            output.write(item)
                        output.write(line)
                        d = depth
                        while(d>0):
                            output.write(d*"\t"+"}\n")
                            d -= 1
                        output.write("}\n")
                    file_num += 1
                else:
                    test_prefix.append(line)
                    assert_num += 1
            if(file_num == 0):
                with open(dir_path+"/"+file.replace("txt", str(file_num))+".txt", "w", encoding="utf-8") as output:
                    output.write("public class aTest{\n")
                    for item in test_prefix:
                        output.write(item)
                    output.write("}\n")
            total+=assert_num

print(total)
