import os
from glob import glob
import re
import json

output_list = [ 
    "Bicycle", 
    "Scooter", 
    "Wheelchair", 
    "Mobility Scooter"
]

for classname in output_list:
    output = {}
    for ground_truth_filename in sorted(glob("*.txt")):
        image_filename = re.sub(r"\.txt$", "", ground_truth_filename)
        output[image_filename] = []
        with open(ground_truth_filename, "r") as file:
            for line in file:
                detection_classname, *box = line.rstrip().split(",")
                
                print(detection_classname, classname)
                if detection_classname.lower() == classname.lower():
                    output[image_filename].append(box)

    print(f"ground_truth_{classname}.json")
    with open(f"ground_truth_{classname}.json", "w") as output_file:
        output_file.write(json.dumps(output))