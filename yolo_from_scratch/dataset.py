import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    # the csv_file has 2 columns: one each for the image and labels
    # the first column has the name of the image file, e.g. 00001.jpg
    # the second column has the name of the label file, e.g. 00001.txt
    # the image and the label have the same name with different extensions
    # the labels (.txt) file has 5 values per line, namely:
    # class, cx, cy, w, h
    # the box co-ordinates for the bounding box are provided in YOLO format (cxcywh as ratio wrt image sizes)

    def __init__(
            self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotation.iloc[index,1])
        boxes = []

        # read each line from the annotations file (.txt) by accessing the file name
        # from the annotations dataframe (.csv) file and add to the boxes list 
        with open(label_path) as f:
            lines = f.readlines()
            for label in lines:
                class_label, x, y, width, height = [
                    float
                ]
                boxes.append([class_label,x,y,width,height])
        
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index,0])
        boxes = torch.tensor(boxes)
        image = Image.open(img_path)

        # apply any data augmentation to the image, if required
        # we're passing in both the labels as well as the image, because for some augmentations such as
        # image rotation, we'd need to apply the transforms on the labels, too  
        if self.transform:
            image, boxes = self.transform(image, boxes)

        # shape of label_matrix: (S,S,25)
        label_matrix = torch.zeros((self.S,self.S,self.C+(5*self.B))) # since we have only one bounding box per cell

        # the box co-ordinates are given relative to the entire image,
        # we need to convert these to make them relative to each cell 
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # find the cell (row_num:=i, col_num:=j) that the bounding box center point belongs to
            # in other words, find the cell that contains the center of the object
            # since the x and y dimensions are in the range [0,1], and S denotes the number of cells along a row (or column)
            # multiplying x with the number of cells, essentially converts the co-ordinate from range [0,1] to range [0,7] 
            # taking the floor of this value gives 1 value from [0,1,2,3,4,5,6], which is the cell number (along the row for y and along the column for x)
            i, j = int(self.S*y), int(self.S*x)
            
            # finding the co-ordinates relative to the cell
            x_cell, y_cell = (self.S*x)-j, (self.S*y)-i
            # in the following line, we scale the width and the height by the cell number
            width_cell, height_cell = (self.S*width), (self.S*height)
            
            if label_matrix[i,j,20] == 0:
                label_matrix[i,j,20] = 1
                box_coordinates = torch.tensor(
                    [x_cell,y_cell,width_cell,height_cell]
                )
                label_matrix[i,j,21:25] = box_coordinates
                label_matrix[i,j,class_label] = 1
        return image, label_matrix
