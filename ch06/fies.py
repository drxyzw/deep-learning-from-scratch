import os
import shutil

basepath = "./ch06/interpretation_only2"
dirs = os.listdir(basepath)
for dir in dirs:
    if dir.startswith("combo") and len(dir) > 7:
        new_combo_dir = dir[:6]
        new_perm_dir = dir[6:]
        new_combo_full_path = basepath + "/" + new_combo_dir
        os.makedirs(new_combo_full_path, exist_ok=True)
        shutil.move(basepath + "/" + dir, new_combo_full_path)
        os.rename(new_combo_full_path + "/" + dir, new_combo_full_path + "/" + new_perm_dir)
        print(dir)

print("end")