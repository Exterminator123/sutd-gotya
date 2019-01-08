import os
from glob import glob
import re
import json

# box1, box2: [x1, y1, x2, y2]
def calculate_iou(box1, box2):
    box1 = [float(i) for i in box1]
    box2 = [float(i) for i in box2]

    intersect_x1 = max(box1[0], box2[0])
    intersect_y1 = max(box1[1], box2[1])
    intersect_x2 = min(box1[2], box2[2])
    intersect_y2 = min(box1[3], box2[3])
 
    # return 0 if there isn't an intersection
    if intersect_x2 < intersect_x1 or intersect_y2 < intersect_y1:
        return 0

    intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
    
    # adding the two boxes gives a double count for the intersection
    # therefore union area = box1 area + box2 area - intersection area
    union_area = ((box1[2] - box1[0]) * (box1[3] - box1[1])) + ((box2[2] - box2[0]) * (box2[3] - box2[1])) - intersect_area

    return intersect_area / union_area

classlist = [
    "bicycle", 
    "scooter", 
    "wheelchair", 
    "mobility scooter"
]

# for detection_filename in sorted(glob("*.csv")):
#     print(f"processing {detection_filename}")
#     with open(detection_filename, "r") as file:
#         next(file)
#         for detection_line in file:
#             image_filename, detection_label, *detection_box, _, _ = detection_line.rstrip().split(",")
#             if detection_label not in classlist:
#                 continue

#             try:
#                 image_object = output[detection_label][image_filename]
#             except KeyError:
#                 output[detection_label][image_filename] = { "boxes": [], "scores": [] }
#                 image_object = output[detection_label][image_filename]

#             # iterate through each object in the corresponding ground truth file
#             # if iou > 0, add it to score i guess
            
#             ground_truth_file_name = os.path.join("ground_truth", re.sub(r"jpg$", "txt", image_filename))

#             viable_ious = []

#             if glob(ground_truth_file_name) != []:
#                 with open(ground_truth_file_name, "r") as ground_truth_file:
#                     for truth_line in ground_truth_file:
#                         truth_label, *truth_box = truth_line.rstrip().split(",")
#                         if truth_label.lower() == detection_label.lower() and calculate_iou(detection_box, truth_box) != 0:
#                             viable_ious.append(calculate_iou(detection_box, truth_box))
#             else:
#                 raise Exception(f"Missing ground truth file for {image_filename}")

#             image_object["boxes"].append(detection_box)
#             print(max(viable_ious))

for detection_filename in sorted(glob("*.csv")):
    for output_classname in classlist:
        output = {}
        with open(detection_filename, 'r') as detection_file:
            next(detection_file)
            for detection_line in detection_file:
                image_filename, detection_label, *detection_box, obj_conf, class_conf = detection_line.rstrip().split(",")
                detection_box = [float(i) for i in detection_box]
                if detection_label != output_classname:
                    continue

                keyname = re.sub(r"\.jpg$", "", image_filename)
                keyname = re.sub(r"\.png$", "", keyname)

                try:
                    image_object = output[keyname]
                except KeyError:
                    output[keyname] = { "boxes": [], "scores": [] }
                    image_object = output[keyname]

                image_filename = re.sub(r"jpg$", "txt", image_filename)
                image_filename = re.sub(r"png$", "txt", image_filename)
                ground_truth_file_name = os.path.join("ground_truth", re.sub(r"jpg$", "txt", image_filename))

                viable_ious = []
                if glob(ground_truth_file_name) != []:
                    with open(ground_truth_file_name, 'r') as ground_truth_file:
                        for truth_line in ground_truth_file:
                            truth_label, *truth_box = truth_line.rstrip().split(",")
                            if truth_label.lower() == detection_label.lower():
                                viable_ious.append(calculate_iou(detection_box, truth_box))

                image_object["boxes"].append(detection_box)
                image_object["scores"].append(max(viable_ious or [0]))
                print(viable_ious)
                
        print(f"{detection_filename}/{output_classname}.json")
        output_dir = re.sub("^pmd_yolov3_", "", detection_filename)
        output_dir = re.sub(r"\.weights\.csv$", "_json", output_dir)
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/prediction_{output_classname}.json", "w") as output_file:
            output_file.write(json.dumps(output))


# for classname, items in output.items():
#     with open(f"ground_truth_{classname}.json", "w") as output_file:
#         output_file.write(json.dumps(output))
