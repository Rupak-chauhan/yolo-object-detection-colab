# Augmentation Pipeline

import albumentations as A
import os
import cv2
import numpy as np

img_path = "./datasets/train/images"
labels_path = "./datasets/train/labels"

# Get the list of all files in images directories
dir_list = os.listdir(img_path)
# print("All Images Files :", dir_list)

class AugmentationData:
    def __init__(self):
        print("Inside init :");

    # Converting back to yolo format and same in labels file from the response of generate augmented boxes
    def convert_to_yolo_format(self, data, name):
        print("convert_to_yolo_format :");
        mapping_classes = ['Person', '2-wheeler', '4-wheeler']
        # Define the file path where you want to save the data
        file_path = labels_path + "/" + name
        
        # Open the file in write mode ('w')
        with open(file_path, 'w') as file:
            for tuple_data in data:
                # Get the last element (class type)
                class_name = tuple_data[-1]
                # Find its index in the mapping array
                mapped_index = mapping_classes.index(class_name)
                # Convert the tuple to a list to modify it
                modified_list = list(tuple_data)
                # Insert the mapped index at the beginning of the list
                modified_list.insert(0, mapped_index)
                modified_list.pop()
                # Convert each element of the modified list to a string
                # Join them with a space and create a single string to write to the file
                line = ' '.join(map(str, modified_list))
                # Write the line to the file
                file.write(line + '\n')
        
        # print(f'label Data saved to {file_path}')

    # construct the actual format with mapping classes
    def construct_mapping(self, data):
        print("---------Inside construct_mapping-------------")
        mapping_classes = ['Person', '2-wheeler', '4-wheeler']
        # Convert the data to a list of lists of floats and move the 0th index element to the end
        mapped_data = []
        for sublist in data:
            split_items = sublist[0].split()
            if len(split_items):
                first_element = mapping_classes[int(split_items[0])]
                remaining_elements = [float(item) for item in split_items[1:]]
                remaining_elements.append(first_element)
                mapped_data.append(remaining_elements)
            
        return mapped_data

    # method to choose label for corresponding image and converting it for  albumentations format
    def get_formatted_label(self, fl_name):
        print("---------Inside get_formatted_label -------------")
        fl_label_path = labels_path + "/" + os.path.splitext(fl_name)[0] + ".txt"
        fs = open(fl_label_path);
        fs_content = fs.read().split("\n");
        temp_lst = []
        for each_label in fs_content:
            temp_lst.append([each_label])
        mapped_data = self.construct_mapping(temp_lst)
        return mapped_data
    

    def init_augmentation(self):
        for fl_name in dir_list:
            img = cv2.imread(img_path + "/" + fl_name)
            box_labels = self.get_formatted_label(fl_name)
            self.augmentation_pipeline(img, box_labels, fl_name)
            print("----Data Augmentation Successfully Done-----");
    
    def augmentation_pipeline(self, img, box_labels, fl_name):
        print("---------Inside Augmentation Pipeline-------------")
        
        for val in ["aug_flip_", "aug_rotate_", "aug_scale_"]:
            if val == "aug_flip_":
                transform = A.Compose([
                    A.Flip(p=0.8)
                ], bbox_params=A.BboxParams(format='yolo'))
            
                transformed = transform(image=img, bboxes=box_labels)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                # print("transformed_image :", len(transformed_image), "transformed_bboxes :", transformed_bboxes);
            
                cv2.imwrite(img_path + "/" + val + fl_name, transformed_image)
                self.convert_to_yolo_format(transformed_bboxes, val + os.path.splitext(fl_name)[0] + ".txt")
            
            elif val == "aug_rotate_":
                transform = A.Compose([
                    A.Rotate(limit=180, p=0.8)
                ], bbox_params=A.BboxParams(format='yolo'))
            
                transformed = transform(image=img, bboxes=box_labels)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                # print("transformed_image :", len(transformed_image), "transformed_bboxes :", transformed_bboxes);
                
                cv2.imwrite(img_path + "/" + val + fl_name, transformed_image)
                self.convert_to_yolo_format(transformed_bboxes, val + os.path.splitext(fl_name)[0] + ".txt")
            
            elif val == "aug_scale_":
                transform = A.Compose([
                    A.RandomScale(scale_limit=(0.5, 2.0), p=0.5)
                ], bbox_params=A.BboxParams(format='yolo'))
            
                transformed = transform(image=img, bboxes=box_labels)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                # print("transformed_image :", len(transformed_image), "transformed_bboxes :", transformed_bboxes);
                
                cv2.imwrite(img_path + "/" + val + fl_name, transformed_image)
                self.convert_to_yolo_format(transformed_bboxes, val + os.path.splitext(fl_name)[0] + ".txt")


