import os
from glob import glob
import re
import json

output_list = [ 
    "Bicycle", 
    "Scooter", 
    "Wheelchair", 
    "Mobility Scooter", 
    "Skateboard" 
]

for classname in output_list:
    output = {}
    for ground_truth_filename in sorted(glob("*.txt")):
        image_filename = re.sub("txt$", "jpg", ground_truth_filename)
        output[image_filename] = []
        with open(ground_truth_filename, "r") as file:
            for line in file:
                detection_classname, *box = line.rstrip().split(",")
                
                if detection_classname == classname:
                    output[image_filename].append(box)

    print(f"ground_truth_{classname}.json")
    with open(f"ground_truth_{classname}.json", "w") as output_file:
        output_file.write(json.dumps(output))