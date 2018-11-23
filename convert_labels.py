from PIL import Image
import os

"""
Generates YOLOv3-style label files from OID-style label files.
Run this in a Label directory (i.e. OID/Dataset/test/*/Label)
"""

def convert(line, width, height):
    """
    Convert one line of coordinates from OID-style (absolute coordinates) to YOLOv3-style (relative floats around bounding boxes).    
    returns the new line.    
    :param line: a line from an OID label file. (class x-start y-start x-end y-end)
    :param width: width of image
    :param height: height of image
    """
    dw = 1.0 / width
    dh = 1.0 / height
    
    class_name, x1, y1, x2, y2 = line.strip().split(" ")
    x1, y1, x2, y2 = [float(n) for n in (x1, y1, x2, y2)]
    
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    
    box_width = x2 - x1
    box_height = y2 - y1

    #print("x: {} to {} out of {}".format(x1, x2, width))
    #print("\t x_center = {}".format(x_center))
    #print("\t box_width = {}".format(box_width))
    #print("y: {} to {} out of {}".format(y1, y2, height)) 
    #print("\t y_center = {}".format(y_center))
    #print("\t box_height = {}".format(box_height))
    return "{} {} {} {} {}\n".format(class_name,
                                    x_center * dw,
                                    y_center * dh,
                                    box_width * dw,
                                    box_height * dh)
     

if __name__ == "__main__":
    try:
        os.mkdir("Converted")
    except FileExistsError:
        raise Exception("Delete the Converted directory and try again")
    label_files = os.listdir(".")
    for label_file in label_files:
        if ".txt" in label_file:
            image_file = "../{}".format(label_file.replace(".txt", ".jpg")) 
            with open(label_file, "r") as f:
                lines = f.readlines()
                converted_lines = []
                for line in lines:
                    im = Image.open(image_file) 
                    width, height = im.size
                    converted_lines.append(convert(line, width, height))
                with open("Converted/{}".format(label_file), "w") as f:
                    f.writelines(converted_lines)
