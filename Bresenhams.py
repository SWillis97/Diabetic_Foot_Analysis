# ==============================================================================
# Test File for cleaning level after smoothing
# Inputs: - smoothed_image
#         - foot_dim
# Output: Foot_Profile
# ==============================================================================

import numpy as np
import cv2
from PIL import Image,ImageOps
from bresenham import bresenham
import numpy as np
import cv2
from PIL import Image
from bresenham import bresenham

imname = "smoothed.jpg"
image = cv2.cvtColor(cv2.imread(imname), cv2.COLOR_BGR2GRAY)

rows = np.size(image,axis=0)
columns = np.size(image,axis=1)

def Clean_Image(image,rows,columns):
    for i in range(0,rows):
        for j in range(0,columns):
            if image[i][j] <= 200:
                image [i][j] = 0
            else: image[i][j] = 255
    return image

image = Clean_Image(image,rows,columns)
image2 = image
kernel = int(rows/10)
print(kernel)
if kernel%2 == 0:
    kernel+= 1
midpoint = int(kernel/2)
expanded_img = np.array(ImageOps.expand(Image.fromarray(image),border=midpoint+1))

def Find_Search_Area(expanded_img,i,j,kernel):
    search_area = expanded_img[i:i+kernel,j:j+kernel]
    return search_area

def Area_Search(search_area,kernel,midpoint):
    change = False
    for n in range(0,(kernel-1)*2):
        if n < kernel:
            coordinates = [0,n]
        else:
            coordinates = [n-kernel+1,kernel-1]
        points =  np.array(list(bresenham(coordinates[0],coordinates[1],midpoint,midpoint)))[0:midpoint]
        anti_points = np.array(list(bresenham(kernel-1-coordinates[0],kernel-1-coordinates[1],midpoint,midpoint)))[0:midpoint]
        for x in range(0,midpoint):
            if x == 0:
                points_value = search_area[points[x][0]][points[x][1]]
                anti_points_value = search_area[anti_points[x][0]][anti_points[x][1]]
            else:
                points_value = np.hstack((points_value,search_area[points[x][0]][points[x][1]]))
                anti_points_value = np.hstack((anti_points_value,search_area[anti_points[x][0]][anti_points[x][1]]))
        if np.any(points_value==255) == True:
            if np.any(anti_points_value==255) == True:
                change = True
                break
    return change


for i in range(0,rows):
    for j in range(0,columns):
        if image [i][j] == 0:
            search_area = Find_Search_Area(expanded_img,i,j,kernel)
            if np.any(search_area!=0) == True:
                image2[i][j] = 125
                change = Area_Search(search_area,kernel,midpoint)
                if change == True:
                    image[i][j] = 255

Image.fromarray(image).save("final_blob.jpg")
Image.fromarray(image2).save("final_blob_edges.jpg")
Image.fromarray(search_area).save("search_area.jpg")
Image.fromarray(expanded_img).save("blob_expanded.jpg")
