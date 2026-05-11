# %%

import os
import os.path


# %%
dir_list = [
    "amborella",
    "castanea",
    "convolvulaceae",
    "desmodium",
    "eugenia",
    "laurus",
    "litsea",
    "magnolia",
    "monimiaceae",
    "rubus",
    "ulmus",
]

# %%
data_folder = "your_data_folder"
dest_folder = "your_destinatino_folder"

dir_list = [
    "amborella",
    "castanea",
    "convolvulaceae",
    "desmodium",
    "eugenia",
    "laurus",
    "litsea",
    "magnolia",
    "monimiaceae",
    "rubus",
    "ulmus",
]
for dir in dir_list:
    folder = os.path.join(data_folder, dir)
    cmd = (
        "python detect-boxes.py --weights best.pt --conf-thres 0.9 --img-size 640 --source "
        + folder
        + " --project "
        + dest_folder
        + " --name "
        + dir
        + " --device 2,3 --view-img --no-trace"
    )
    print(cmd)
    os.system(cmd)

# %%

print("Done")
