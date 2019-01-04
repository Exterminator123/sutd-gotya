import os
from glob import glob
import re
import json

# box1, box2: [x1, y1, x2, y2]
def calculate_iou(box1, box2):
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

# for detection_file_name in sorted(glob("*.csv")):
#     print(detection_file_name)
#     with open(detection_file_name, "r") as detection_file:
#         next(detection_file) # skip the header line
#         for line in detection_file:
#             # csv header: Filename,Label,x1,y1,x2,y2,Object Conf,Class Conf
#             image_filename, detected_label, *detected_bounding_box, object_conf, class_conf = line.split(",")
#             detected_bounding_box = [float(x) for x in detected_bounding_box]
#             print(f"\t{image_filename}:")            

#             ground_truth_file_name = os.path.join("ground_truth", re.sub(r"jpg$", "txt", image_filename))
#             if glob(ground_truth_file_name) != []:
#                 with open(ground_truth_file_name, "r") as ground_truth_file:
#                     for line in ground_truth_file:
#                         truth_label, *truth_bounding_box = line.rstrip().split(",")
#                         truth_bounding_box = [float(x) for x in truth_bounding_box]
#                         iou = calculate_iou(detected_bounding_box, truth_bounding_box) 
#                         if iou != 0:
#                             if detected_label.lower() == truth_label.lower():
#                                 print(f"\t\t{truth_label} correctly detected,  iou: {iou}")
#                             else:
#                                 print(f"\t\tMislabed {truth_label} as {detected_label}, iou: {iou}")
#             else:
#                 raise Exception(f"Missing ground truth file for {image_filename}")

json_dict = { "images": [] }

for i, ground_truth_file_name in enumerate(sorted(glob("ground_truth/*.txt"))):
    ground_truth_image_filename = re.sub(r"txt$", "jpg", ground_truth_file_name)
    ground_truth_image_filename = re.sub(r"^ground_truth\/", "", ground_truth_image_filename)

    # print(f"image {i}: {ground_truth_image_filename}")
    image_dict = { "index": i, "filename": ground_truth_file_name, "results": [] }

    with open(ground_truth_file_name, "r") as ground_truth_file:
        ground_truth_objects = [line for line in ground_truth_file]
        for detection_file_name in sorted(glob("*.csv")):

            result_dict = { "iteration": int(detection_file_name.replace("pmd_yolov3_", "").replace(".csv", "")), "objects": []}
            
            for ground_truth_object in ground_truth_objects:
                truth_label, *truth_bounding_box = ground_truth_object.rstrip().split(",")
                truth_label = truth_label.lower()
                truth_bounding_box = [float(i) for i in truth_bounding_box]



                matching_object_label = None
                matching_object_iou = None

                with open(detection_file_name) as detection_file:
                    next(detection_file)
                    for line in (line for line in detection_file if line.split(",")[0].lower() == ground_truth_image_filename.lower()):
                        image_filename, detected_label, *detected_bounding_box, _, _ = line.split(",")
                        detected_bounding_box = [float(i) for i in detected_bounding_box]

                        iou = calculate_iou(detected_bounding_box, truth_bounding_box)
                        if iou != 0:
                            matching_object_iou = iou
                            matching_object_label = detected_label

                object_dict = {"expected": truth_label, "result": matching_object_label, "iou": matching_object_iou}
                result_dict["objects"].append(object_dict)

            image_dict["results"].append(result_dict)


    json_dict["images"].append(image_dict)

for image in json_dict['images']:
    print(f"image {image['index']}: {image['filename']}")
    for result in image['results']:
        print(f"\tafter {result['iteration']} iterations")
        for obj in result['objects']:
            if not obj['result']:
                print(f"\t\tMissed a {obj['expected']}")
            elif obj['result'] != obj['expected']:
                print(f"\t\tMisidentified {obj['expected']} as {obj['result']}, iou: {obj['iou']}")
            elif obj['result'] == obj['expected']:
                print(f"\t\tCorrectly identified {obj['expected']}, iou: {obj['iou']}")

# print(json.dumps(json_dict))