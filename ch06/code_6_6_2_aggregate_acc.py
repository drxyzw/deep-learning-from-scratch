import os
import pandas as pd
out_list = []
base_dir = "./ch06/"
max_layers = 10
for n_layers in range(1, max_layers + 1):
    layer_dir =  base_dir + "/interpretation_only" + str(n_layers)
    if os.path.exists(layer_dir):
        for size_combo in range(1, 10 + 1):
            combo_dir = layer_dir + "/combo" + str(size_combo)
            if os.path.exists(combo_dir):
                for fileOrDir in os.listdir(combo_dir):
                    # size_combo = combo_dir.count("_") + 1
                    if os.path.isdir(fileOrDir):
                        dir_target = combo_dir + "/" + fileOrDir
                        for filename in os.listdir(dir_target):
                            if filename.startswith("W1_input_"):
                                file_wo_ext = os.path.splitext(filename)[0]
                                test_acc_str = file_wo_ext.split("_")[-1]
                                test_acc = float(test_acc_str)
                                out = {"n_layers": n_layers, "size": size_combo, "combination": str(combo_dir), "test accuracy rate": test_acc}
                                out_list.append(out)
# df_out = pd.DataFrame(columns=["n_layers", "size", "combination", "test accurary rate"])
df_out = pd.DataFrame(out_list)
df_out.to_excel(base_dir + "combo_test_acc.xlsx", index=False)

