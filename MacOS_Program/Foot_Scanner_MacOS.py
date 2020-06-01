# =============================================================================
# Design Engineering MEng Final Year Project -
# Functions used for the main processing
# Input:  Image of foot from scanner
# Ouputs: - two csv files defining the shape of foot and ulcer for CAD file
#         - API upload of points and other information to google sheets for
#           storage
# =============================================================================

import os
import sys
import cv2
import math
import shutil
import numpy as np
import random as rng
from PIL import Image, ImageOps

class Kmeans(object):
    """Class to do image processing and find ulcer location and foot geometries"""

    def Denoised(self,image):
        """Take the input image and remove obvious noise, like moles or dirt"""
        denoised = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
        return denoised

    def Get_Image_Resolution(self,image):
        """Finding the number of rows and columns in the original image"""
        orig_im_res = (np.size(image,axis=0),np.size(image,axis=1))
        return orig_im_res

    def Clean_Background(self,image,orig_im_res):
        """Remove the background noise from the image. This prevents the K-means
        clustering from assigning two colours to the white background"""
        for i in range(0,orig_im_res[0]):
            for j in range(0,orig_im_res[1]):
                if sum(np.array(image[i][j]))>=650:
                    image[i][j] = [255,255,255]
        return image

    def Alter_Centers(self,centers,number_of_clusters):
        """Function to find the 'most white' and 'most red' clusters in the image"""
        # initiate foot_list for for loop initial if condition list
        foot_list = []

        # create new lists from the centers RGB values
        for i in range(0,number_of_clusters):
            foot_RGB = np.sum(centers[i])
            ulcerR = int(centers[i][0])
            ulcerG = int(centers[i][1])
            ulcerB = int(centers[i][2])
            ulcer_RGB = int(ulcerR-ulcerG-ulcerB)
            if np.size(foot_list) == 0:
                foot_list = foot_RGB
                ulcer_list = ulcer_RGB
            else:
                foot_list = np.hstack((foot_list,foot_RGB))
                ulcer_list = np.hstack((ulcer_list,ulcer_RGB))

        # find the desired indexes
        Background_Index = foot_list.argmax()
        Ulcer_Index = ulcer_list.argmax()
        return Background_Index,Ulcer_Index

    def Clustering(self,image,number_of_clusters):
        """Use of the OpenCV library to perform K-Means clustering on the cleaned image"""
        # converting 3D array into one dimensional list of points for clustering
        pixel_values = image.reshape((-1,3))
        pixel_values = np.float32(pixel_values)

        # finding center points and making list of pixels in terms of which cluster theyre in
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, (centers) = cv2.kmeans(pixel_values, number_of_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        labels = labels.flatten()

        return centers,labels

    def Manipulate_Centers(self,centers,number_of_clusters):
        """function to manipulate the values of the centers to be black and white and seperate"""
        # call function Alter_Centers to get the centers indexes for the next stage
        Background_Index, Ulcer_Index = self.Alter_Centers(centers,number_of_clusters)

        # creating seperate list for processing ulcer and feet seperately.
        # centers was a global variable so had to be broken
        centers_1 = np.array(centers)
        centers_2 = np.array(centers)

        # manipulate lists centers and centers_2 to black and white
        for i in range(0,number_of_clusters):
            if i == Background_Index:
                centers_1[i] = np.array([0,0,0])
            else:
                centers_1[i] = np.array([255,255,255])
            if i == Ulcer_Index:
                centers_2[i] = np.array([255,255,255])
            else:
                centers_2[i] = np.array([0,0,0])

        return centers_1,centers_2

    def Reform_Image(self,centers,labels,orig_image):
        """function to take centers and labels and reform them into a new image"""
        image = centers[labels.flatten()].reshape(orig_image.shape)
        return image

    class Post_Processing(object):
        """Class to deal with both the seperate foot and ulcer images independently"""

        def Add_Border(self,image):
            """function to add black border around image and prevent clipping errors
            in a latter cleaning stage"""
            expanded_img = np.array(ImageOps.expand(Image.fromarray(image),border=100))
            return expanded_img

        def make_BandW(self,image):
            """function to clean image and make strict black and white"""
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
            return blackAndWhiteImage

        def imsmooth(self,image):
            """Removing further noise and smoothing image. For instance, this
            removes gaps between toes"""

            # defining sizes, to search for anomolies around a point to fix
            kernel = np.ones((10,10),np.uint8)
            kernel2 = np.ones((100,100),np.uint8)

            # noise removal from image edges
            img_erosion = cv2.erode(image, kernel, iterations=1)
            img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
            img_closed = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel2)

            # creating contour around the 'edge'. This will overlay the image to form a smooth boundry
            edged = cv2.Canny(img_dilation, 30, 200)
            contours, hierarchy = cv2.findContours(edged,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img_dilation, contours, -1, (0, 255, 0), 3)
            img_closed = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel2)
            edged = cv2.Canny(img_closed, 30, 200)

            # adding contours to the image
            contours, hierarchy = cv2.findContours(edged,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img_closed, contours, -1, (0, 255, 0), 3)
            return img_closed

        def thresh_callback(self,image):
            """find borders and ellipses in black and white images"""
            # run edge detction on images
            threshold = 100
            canny_output = cv2.Canny(image,threshold,threshold*2)
            contours,heirachy = cv2.findContours(canny_output,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            # find rotated rectangles and ellipses for each contour
            minRect = [None]*len(contours)
            minEllipse = [None]*len(contours)
            for i,c in enumerate(contours):
                minRect[i] = cv2.minAreaRect(c)
                if c.shape[0]>5:
                    minEllipse[i] = cv2.fitEllipse(c)

            # get the points for the ellipses and rectangle in numpy array form
            ellipses_points = np.array(minEllipse)
            Rect_Points = np.array(minRect)

            # draw contours, rotated rectangles and ellipses_points
            drawing = np.zeros((canny_output.shape[0],canny_output.shape[1],3), dtype=np.uint8)
            for i,c in enumerate(contours):
                # give random colour
                colour = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                # add contours
                cv2.drawContours(drawing, contours, i, colour)
                # add ellipses
                if c.shape[0] > 5:
                    cv2.ellipse(drawing, minEllipse[i], colour, 2)
                # add rotated rectangles
                box = cv2.boxPoints(minRect[i])
                box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
                cv2.drawContours(drawing, [box], 0, colour)

            for (xA, yA) in list(box):
                # draw circles corresponding to the current points and
                cv2.circle(drawing, (int(xA), int(yA)), 9, (0,0,255), -1)
                cv2.putText(drawing, "({},{})".format(xA, yA), (int(xA - 50), int(yA - 10) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            return Rect_Points,ellipses_points,canny_output,drawing

        def Clean_Image(self,image,rows,columns):
            """clean image to be just 0s and 255s before smoothing toe gaps"""
            for i in range(0,rows):
                for j in range(0,columns):
                    if image[i][j] <= 200:
                        image [i][j] = 0
                    else: image[i][j] = 255
            return image

        def fill_gaps(self,image):
            """function to fill gaps between toes and smooth image generally"""

            kernel = np.ones((100,100),np.uint8)
            original_rows = np.size(image,axis=0)
            original_columns = np.size(image,axis=1)

            angle_whole = 90
            step_number = 15
            angle = int(angle_whole/step_number)
            imageold=image
            completed_angle = 0

            # smooth image and fill toes
            for i in range(0,step_number):
                rotated_image,flag = self.Rect_Measurements.rotate_image(self,image,angle,flag="fill gaps")
                image = cv2.morphologyEx(rotated_image, cv2.MORPH_CLOSE, kernel)
                completed_angle += angle

            # fix image by reversing the rotation
            image,flag = self.Rect_Measurements.rotate_image(self,image,-completed_angle,flag="fill gaps")
            new_im_index = np.array([int((np.size(image,axis=0)-original_rows)/2),int((np.size(image,axis=1)-original_columns)/2)])
            new_im = image[new_im_index[0]:new_im_index[0]+original_rows,new_im_index[1]:new_im_index[1]+original_columns]

            # make sure image is cleaned to discrete values
            new_im = self.Clean_Image(new_im,original_rows,original_columns)
            new_im = cv2.Canny(new_im,100,200)
            return new_im

        class Polar_Functions(object):
            """Class investegating the usefulness of Polar investegation of image
            for in depth detail of foot structure"""

            def Polar_Mapper(self):
                """ function to attempt to investegate the number of toes"""

                print("\nPolar functions currently not in use.\n")

            def __init__(self,canny):
                self.image = canny
                self.Polar_Mapper()

        class Rect_Measurements(object):
            """Class to get measurements for the CAD models"""

            def find_width_and_height(self,alt_rotate):
                """function to find measurements from rotated image"""

                # find size of rotated image
                rows = np.size(alt_rotate, axis=0)
                columns = np.size(alt_rotate, axis=1)

                # initiate row_list for first elif append
                row_list = []
                for i in range(0,rows):
                    found = False
                    for j in range(0,columns):
                        if found == True:
                            break
                        elif alt_rotate[i][j] != 0:
                            row_list.append(i)
                            found = True

                # define row_range
                row_range = np.array([min(row_list), -min(row_list)+max(row_list),
                                      int((max(row_list)-min(row_list))/20),max(row_list)])

                # initiate column_list for first elif append
                column_list = []
                for i in range(0,columns):
                    found = False
                    for j in range(0,rows):
                        if found == True:
                            break
                        elif alt_rotate[j][i] != 0:
                            column_list.append(i)
                            found = True

                # define column_start to normalise the measurements irrelevent of position in image
                column_start = min(column_list)
                #define column_dist to get total pixel length of foot
                column_dist = max(column_list)-min(column_list)
                return row_range,column_start,column_dist

            def rotate_image(self,mat,angle,flag):
                """Rotates an image (angle in degrees) and expands image to avoid cropping"""

                mat = np.array(mat)

                height, width = mat.shape[:2] # image shape has 3 dimensions
                image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

                rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

                # rotation calculates the cos and sin, taking absolutes of those.
                abs_cos = abs(rotation_mat[0,0])
                abs_sin = abs(rotation_mat[0,1])

                # find the new width and height bounds
                bound_w = int(height * abs_sin + width * abs_cos)
                bound_h = int(height * abs_cos + width * abs_sin)

                # subtract old image center (bringing image back to origo) and adding the new image center coordinates
                rotation_mat[0, 2] += bound_w/2 - image_center[0]
                rotation_mat[1, 2] += bound_h/2 - image_center[1]

                # rotate image with the new bounds and translated rotation matrix
                rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

                # recursive loop to correct 90 degree alignment error
                if flag != "fill gaps":
                    row_range,column_start,column_dist = self.find_width_and_height(rotated_mat)
                    if column_dist > row_range[1]:
                        flag = True
                        rotated_mat = self.rotate_image(rotated_mat, -90,flag)
                return rotated_mat, flag

            def Find_Side(self,foot_dim,foot_angle,im):
                """function to find which side of the foot is in the scan"""
                # get gradient and constant from measurements for form y = gx + c
                gradient_angle = math.radians(90+foot_angle)
                gradient = math.tan(gradient_angle)
                constant = foot_dim[1] - gradient*foot_dim[0]

                # get size of image
                columns = np.size(im, axis = 0)
                rows = np.size(im, axis = 1)

                # array for the number of pixels on each side of the line y = gx + c
                side = np.array([0,0])

                # for loop searching image for whit pixels and assigning them a side
                for i in range(0,columns):
                    for j in range(0,rows):
                        if i-gradient*j < constant:
                            if im[i][j] == 255:
                                im[i][j] = 50
                                side[1]+=1
                        else:
                            if im[i][j] == 0:
                                side[0]+=1

                # working out which foot is photographed
                if side[0]<side[1]:
                    side = "left"
                else: side = "right"
                return side,im

            def Find_Point(self,im_list,image,i,j):
                """function to search for single point in the top and bottom layers"""
                point = np.array([i,j,image[i][j]])
                if np.size(im_list) == 0:
                    im_list = point
                else:
                    im_list = np.vstack((im_list, point))
                return im_list

            def Sorter(self,_list_,*argv):
                """function to sort the list for three different use cases"""
                argv = np.array(argv)
                if np.size(argv) > 0:
                    row_ran = argv[0]
                    column_start = argv[1]
                try:
                    np.size(_list_,axis=1)
                    sorted_array = _list_[_list_[:,2].argsort()]
                    sorted_array = sorted_array[::-1]
                    sorted_array = np.array([sorted_array[0][0]-row_ran,
                                             sorted_array[0][1]-column_start])
                # intentional error included to allow the first half of the function to work for other cases
                except UnboundLocalError:
                    sorted_array = sorted_array[::-1]
                    return sorted_array
                # IndexError occurs if list only has one pixel because of a lack of noise
                # this except allows the code to notice and progress
                except IndexError:
                    sorted_array = np.array([_list_[0]-row_ran,
                                             _list_[1]-column_start])
                return sorted_array

            def splitList(self,_list):
                """function to seperate the list of pixels from a row into
                it's two seperate sides"""
                flag = True
                flag2 = True
                for i in range(0,len(_list)):
                    if i == 0:
                        list1 = _list[i]
                    elif flag == True:
                        if _list[i-1][1]-50 < _list[i][1] < _list[i-1][1]+50:
                            list1 = np.vstack((list1, _list[i]))
                        else:
                            list2 = _list[i]
                            flag = False
                    else:
                        list2 = np.vstack((list2, _list[i]))
                return list1,list2

            def findMiddle(self,input_list,max_row_range,row_ran,column_start):
                """functions to find the central point of a list and output in the
                correct format"""

                middle = float(len(input_list))/2
                if middle % 2 != 0:
                    new_list = input_list[int(middle - .5)]
                    new_list = np.array([new_list[0]-row_ran,new_list[1]-column_start])
                else:
                    list_a = input_list[int(middle)]
                    list_b = input_list[int(middle-1)]
                    new_list = np.array([max_row_range-row_ran,
                                         int(list_b[1]+((int(list_a[1]-list_b[1]))/2))-column_start])
                return new_list

            def Heel_fix(self,image,column_start):
                """function to find correct heel columns value"""
                rows = np.size(image,axis=0)
                columns = np.size(image,axis=1)
                point_list = []
                FLAG = False
                for i in range(0,rows):
                    if FLAG == False:
                        for j in range(0,columns):
                            if FLAG == False:
                                if image[rows-i-1][j] != 0:
                                    if image[rows-i-1][j] >= 200:
                                        FLAG = True
                                        point = j-column_start
                            else:
                                break
                    else:
                        break
                return point

            def Search_Rotated_Image(self,alt_rotate,row_range,column_start):
                """Python function to search through rotated image at defined
                intervals for pixels for the .csv file output"""
                # Initiate lists
                Bottom_List = []
                Top_List = []
                Middle_List = []

                rows = np.size(alt_rotate, axis=0)
                columns = np.size(alt_rotate, axis=1)

                # search for loop
                for i in range(row_range[0],row_range[0]+row_range[2]*20+1,row_range[2]):
                    # reinitiate list every loop
                    line_points = []
                    for j in range(0,columns):
                        if alt_rotate[i][j] != 0:
                            points = np.array([i,j,alt_rotate[i][j]])
                            if i == row_range[0]:
                                Top_List = self.Find_Point(Top_List,alt_rotate,i,j)
                            elif i == max(range(row_range[0], row_range[0]+row_range[2]*20+1,
                                                row_range[2])):
                                Bottom_List = self.Find_Point(Bottom_List,alt_rotate,i,j)
                            else:
                                if np.size(line_points) == 0:
                                    line_points = points
                                else:
                                    line_points = np.vstack((line_points, points))
                    if np.size(line_points) > 0:
                        # slit the list to prevent data corruption
                        list1,list2 = self.splitList(line_points)
                        # find points from lists
                        point1 = self.Sorter(list1,row_range[0],column_start)
                        point2 = self.Sorter(list2,row_range[0],column_start)
                        current = np.array([[point1[0],point1[1]],[point2[0],point2[1]]])
                        if np.size(Middle_List) == 0:
                            Middle_List = current
                        else:
                            Middle_List = np.vstack((Middle_List,current))

                Original_Bottom_Point = self.findMiddle(Bottom_List,row_range[3],row_range[0],column_start)
                # extra function to correct error in heel column value
                Bottom_point = np.array([Original_Bottom_Point[0],self.Heel_fix(alt_rotate,column_start)])
                whole_points = np.vstack((self.findMiddle(Top_List,row_range[0],
                                                     row_range[0],column_start),
                                          Middle_List,
                                          Bottom_point))
                return whole_points

            def Rearrange_Rect_Points(self,whole_points):
                """Function to rearrange list of points into format for CAD
                spline"""
                order = np.array([[0],[39],[1],[38],[2],[37],
                         [3],[36],[4],[35],[5],[34],
                         [6],[33],[7],[32],[8],[31],
                         [9],[30],[10],[29],[11],[28],
                         [12],[27],[13],[26],[14],[25],
                         [15],[24],[16],[23],[17],[22],
                         [18],[21],[19],[20]])
                whole_points = np.hstack((whole_points, order))
                whole_points = self.Sorter(whole_points)
                whole_points = np.delete(whole_points,2,axis=1)
                # include first point as last to create closed body in CAD
                whole_points = np.vstack((whole_points,whole_points[0]))
                # convert to format for API
                whole_points = whole_points.ravel().tolist()
                return whole_points

            def __init__(self,canny,list,image):
                self.rect_center = np.array([list[0][0][0],list[0][0][1]])
                self.rect_dim = np.array([list[0][1][0],list[0][1][1]])
                self.rect_angle = list[0][2]
                self.footSide,self.splitim = self.Find_Side(self.rect_center,self.rect_angle,image)
                self.rotated,self.flag = self.rotate_image(canny,self.rect_angle,flag=False)
                # depending on the reponse to the reccursive loop this solves problems found while debugging
                if np.size(self.rotated) == 2:
                    self.rotated = self.rotated[0]
                self.row_range,self.column_start,self.column_dist = self.find_width_and_height(self.rotated)
                self.whole_points_unformatted = self.Search_Rotated_Image(self.rotated,self.row_range,self.column_start)
                self.whole_points = self.Rearrange_Rect_Points(self.whole_points_unformatted)

        class Ellipse_Measurements(object):
            """class to get four points for .csv file for CAD package relative to the
            angle of the foot, ulcer and foot size"""

            def List_Rearrange(self,list):
                """small function to take input data and convert to 1D numpy array"""
                try:
                    new_list = np.hstack((np.array(list[0]),np.array(list[1]),list[2]))
                except IndexError:
                    new_list = np.hstack((np.array(list[0][0]),np.array(list[0][1]),list[0][2]))
                return new_list

            def Get_Center(self,foot_dim,ellipse_dim):
                """function to get core measurements for ulcer position relative to foot"""
                ydiff = foot_dim[1]-ellipse_dim[1]
                xdiff = foot_dim[0]-ellipse_dim[0]
                hypotenuse = math.sqrt(xdiff**2+ydiff**2)
                if ydiff > 0:
                    if xdiff == 0:
                        theta = -foot_dim[4]
                    elif xdiff < 0:
                        theta = math.degrees(math.acos(ydiff/hypotenuse))-foot_dim[4]
                    else:
                        theta = 270 + math.degrees(math.acos(xdiff/hypotenuse))-foot_dim[4]
                elif ydiff == 0:
                    if xdiff <= 0:
                        theta = 90-foot_dim[4]
                    elif xdiff > 0:
                        theta = 270-foot_dim[4]
                else:
                    if xdiff == 0:
                        theta = 180-foot_dim[4]
                    elif xdiff < 0:
                        theta = 90 + math.degrees(math.acos(abs(xdiff)/hypotenuse))-foot_dim[4]
                    else:
                        theta = 180 + math.degrees(math.acos(abs(ydiff)/hypotenuse))-foot_dim[4]
                if theta >= 360:
                    theta -= 360
                elif theta < 0:
                    theta += 360
                return theta,hypotenuse

            def Get_Ulcer_Center_Pixel(self,foot_dim,theta,distance):
                """function to get the coordinates of the ulcer center pixel position"""
                center_point = np.array([(foot_dim[3]/2)-(distance*math.cos(math.radians(theta))),
                                        (foot_dim[2]/2)+(distance*math.sin(math.radians(theta)))])
                return center_point

            def Get_Ellipse_Dimensions(self,foot_dim,ellipse_dim,points):
                """function to calculate the relative angle of the ellipse shape and
                the location of it's widest and thinest points"""
                pointy = points[0]
                pointx = points[1]
                ulcer_length = ellipse_dim[3]/2
                ulcer_width = ellipse_dim[2]/2
                relative_ulcer_theta = ellipse_dim[4] - foot_dim[4]
                points = np.array([[pointy-(ulcer_length*math.cos(math.radians(relative_ulcer_theta))),
                                    pointx+(ulcer_length*math.sin(math.radians(relative_ulcer_theta)))],
                                    [pointy-(ulcer_width*math.cos(math.radians(relative_ulcer_theta+90))),
                                    pointx+(ulcer_width*math.sin(math.radians(relative_ulcer_theta+90)))],
                                    [pointy+(ulcer_length*math.cos(math.radians(relative_ulcer_theta))),
                                    pointx-(ulcer_length*math.sin(math.radians(relative_ulcer_theta)))],
                                    [pointy+(ulcer_width*math.cos(math.radians(relative_ulcer_theta+90))),
                                    pointx-(ulcer_width*math.sin(math.radians(relative_ulcer_theta+90)))],
                                    [pointy-(ulcer_length*math.cos(math.radians(relative_ulcer_theta))),
                                    pointx+(ulcer_length*math.sin(math.radians(relative_ulcer_theta)))]])
                points = points.ravel().tolist()
                return points

            def Reflect_XDist(self,ulcer_points,foot_dim):
                """second stage of correcting errors"""
                ulcer_points = np.array(ulcer_points)
                for i in range(0,len(ulcer_points)):
                    if i%2 != 0:
                        ulcer_points[i] = foot_dim[3]-ulcer_points[i]
                ulcer_points = ulcer_points.tolist()
                return ulcer_points

            def Reflect_Points(self,ulcer_points):
                """if image was rotated by further 90 degrees this begins to correct it"""
                ulcer_points = (ulcer_points[1],ulcer_points[0],ulcer_points[3],ulcer_points[2],ulcer_points[5],ulcer_points[4],ulcer_points[7],ulcer_points[6],ulcer_points[9],ulcer_points[8])
                return ulcer_points

            def __init__(self,foot_rect,ellipse_dim,flag):
                self.foot_dim = self.List_Rearrange(foot_rect)
                self.ellipse_dim = self.List_Rearrange(ellipse_dim)
                self.theta,self.center_distance = self.Get_Center(self.foot_dim,self.ellipse_dim)
                self.points = self.Get_Ulcer_Center_Pixel(self.foot_dim,self.theta,self.center_distance)
                self.ulcer_points = self.Get_Ellipse_Dimensions(self.foot_dim,self.ellipse_dim,self.points)
                if flag == True:
                    self.ulcer_points = self.Reflect_Points(self.ulcer_points)
                    self.ulcer_points = self.Reflect_XDist(self.ulcer_points,self.foot_dim)

        def __init__(self,image,case,new_path,*argv):
            self.enlarged = self.Add_Border(image)
            if case == "foot":
                Image.fromarray(self.enlarged).save(new_path+"/Step 05 - Enlarged Foot Image.jpg")
            self.strict_BandW_image = self.make_BandW(self.enlarged)
            if case == "foot":
                Image.fromarray(self.strict_BandW_image).save(new_path+"/Step 06 - B&W Foot Image.jpg")
            self.smoothed_image = self.imsmooth(self.strict_BandW_image)
            if case == "foot":
                Image.fromarray(self.smoothed_image).save(new_path+"/Step 07 - Smoothed Foot Image.jpg")
            rng.seed(12345)
            self.rect,self.ellipse,self.canny,self.thresh_drawing = self.thresh_callback(self.smoothed_image)
            if case == "foot":
                self.rect_base = self.rect
                self.canny = self.fill_gaps(self.smoothed_image)
                Image.fromarray(self.canny).save(new_path+"/Step 08 - Cleaned Toes Canny Edge Detect.jpg")
                self.rect = self.Rect_Measurements(self.canny,self.rect,self.strict_BandW_image)
                Image.fromarray(self.rect.rotated).save(new_path+"/Step 09 - Rotated Canny.jpg")
                self.canny = self.Polar_Functions(self.canny)
            if case == "ulcer":
                for i in range(0,2):
                    if i == 0:
                        self.foot_rect = np.array(argv[i])[0]
                    else:
                        self.flag = argv[1]
                self.ellipse = self.Ellipse_Measurements(self.foot_rect,self.ellipse,self.flag)

    def FormatCsv(self,data,csvName):
        """function to format data into correct format for csv for CAD positioning"""
        data = np.reshape(np.array(data),(-1,2))
        length = np.size(data,axis=0)
        data = np.c_[data,np.zeros(length)]
        np.savetxt(csvName, data, delimiter=",")

    def __init__(self,imname,cluster_number,new_path):
        self.image = cv2.cvtColor(cv2.imread(imname), cv2.COLOR_BGR2RGB)
        Image.fromarray(self.image).save(new_path+"/Step 01 - Original Image.jpg")
        self.number_of_clusters = cluster_number
        self.image_resolution = self.Get_Image_Resolution(self.image)
        self.denoised = self.Denoised(self.image)
        Image.fromarray(self.denoised).save(new_path+"/Step 02 - Denoised Image.jpg")
        self.clean_background = self.Clean_Background(self.denoised,self.image_resolution)
        Image.fromarray(self.clean_background).save(new_path+"/Step 03 - Clean Background.jpg")
        self.cluster_centers, self.cluster_labels = self.Clustering(self.clean_background,self.number_of_clusters)
        self.background_centers, self.ulcer_centers = self.Manipulate_Centers(self.cluster_centers,self.number_of_clusters)
        self.Original_KMeans_Image = self.Reform_Image(self.cluster_centers,self.cluster_labels,self.image)
        Image.fromarray(self.Original_KMeans_Image).save(new_path+"/Step 04 - Original K-Means Image.jpg")
        self.foot_image = self.Post_Processing(self.Reform_Image(self.background_centers,self.cluster_labels,self.image),"foot",new_path)
        self.ulcer_image = self.Post_Processing(self.Reform_Image(self.ulcer_centers,self.cluster_labels,self.image),"ulcer",new_path,self.foot_image.rect_base,self.foot_image.rect.flag)
        self.FormatCsv(self.foot_image.rect.whole_points,new_path+"/foot.csv")
        self.FormatCsv(self.ulcer_image.ellipse.ulcer_points,new_path+"/ulcer.csv")
        Image.fromarray(self.ulcer_image.enlarged).save(new_path+"/Step 10 - Enlarged Ulcer Image.jpg")
        Image.fromarray(self.ulcer_image.strict_BandW_image).save(new_path+"/Step 11 - B&W Ulcer Image.jpg")
        Image.fromarray(self.ulcer_image.canny).save(new_path+"/Step 12 - Ulcer Canny Edge Detect.jpg")

def spliter(imname):
    """function to convert string into list of characters"""
    return [char for char in imname]

def get_directory_path(imname):
    """function to remove .jpg from end of name and turn it into a unique directory path to save files in"""
    imname = spliter(imname)
    name = "/"+''.join(imname[0:len(imname)-4])
    return name

def name_error_loop(current_path,new_dir_path,imname):
    """loop to allow for another attempt at inputting a new directory name if the previously entered one already exists"""
    try:
        new_name = input("Please enter a new directory name: ")
        new_path = current_path+"/"+new_name
        os.mkdir(new_path)
        Kmeans(imname,6,new_path)
        os.remove(current_path+"/"+imname)
        print("Image processing completed")
    except FileExistsError:
        print("This directory already exists.")
        name_error_loop(current_path,new_dir_path,imname)

def rename_loop(current_path,new_dir_path,imname):
    """loop to allow data to be saved to new directory and not overwrite existing data"""
    rename = input("Enter Y or N: ")
    if rename == "Y":
        name_error_loop(current_path,new_dir_path,imname)
    elif rename == "N":
        print("Terminating program now...")
    else:
        print("Sorry, that was not a valid answer please enter Y or N as a single character.")
        rename_loop(current_path,new_dir_path,imname)

def input_loop(current_path,new_dir_path,imname):
    """loop to check if the user wants to overwrite existing data"""
    answer = input("Enter Y or N: ")
    if answer == "Y":
        shutil.rmtree(new_dir_path)
        os.mkdir(new_dir_path)
        Kmeans(imname,6,new_dir_path)
        os.remove(current_path+"\\"+imname)
        print("Image processing completed")
    elif answer == "N":
        print("Would you like to save the outputs under a new directory name?")
        rename_loop(current_path,new_dir_path,imname)
    else:
        print("Sorry, that was not a valid answer please enter Y or N as a single character.\nWARNING: THIS WILL DELETE THE OLD DIRECTORY AND REPLACE IT!")
        input_loop(current_path,new_dir_path,imname)

def increase_cluster_and_run(imname,new_dir_path,current_cluster,current_path):
    """function to increase cluster number until run"""
    current_cluster += 1
    print("currently attempting "+str(current_cluster)+" clusters.")
    try:
        if current_cluster <= 10:
            Kmeans(imname,current_cluster,new_dir_path)
            os.remove(current_path+"/"+imname)
            print("Image processing completed")
        else:
            print("increasing clusters not worked, evaluate image by checking lighting and for duplicates and retry.")
    except TypeError:
        increase_cluster_and_run(imname,new_dir_path,current_cluster)

def Process_Images(imname,current_path):
    """function to process any image existing in the current directory"""
    new_dir_path = current_path+get_directory_path(imname)
    print("new path is ", new_dir_path)
    current_cluster = 6
    try:
        os.mkdir(new_dir_path)
        Kmeans(imname,6,new_dir_path)
        os.remove(current_path+"/"+imname)
        print("Image processing completed")
    except TypeError:
        increase_cluster_and_run(imname,new_dir_path,current_cluster,current_path)
    except FileExistsError:
        print("The image "+imname+" has already been processed.\nAnswer Y to overwrite file and N to save outputs to a seperate directory.\nPlease enter Y/N:\nWARNING: THIS WILL DELETE THE OLD DIRECTORY AND REPLACE IT!")
        input_loop(current_path,new_dir_path,imname)

if __name__=='__main__':
    try:
        print("WARNING: ENSURE NO IMAGES IN THIS DIRECTORY ARE COPIES OF ANOTHER.\nTHIS WILL CAUSE AN ERROR AND THE PROCESS TO BREAK.\nALL IMAGES MUST BE ORIGINALS.")
        current_path = os.getcwd()+"/Foot Scanner/Foot Images"
        print("current path is ", current_path)
        file_list = np.array(os.listdir(current_path))
        jpg_list = []
        for i in range(0,np.size(file_list)):
            name_string = np.array(spliter(file_list[i]))
            if np.size(name_string)>4:
                jpg_check = name_string[np.size(name_string)-4:np.size(name_string)]
                if jpg_check[0] == '.' and jpg_check[1] == 'j' and jpg_check[2] == 'p' and jpg_check[3] == 'g':
                    jpg_list = np.hstack((jpg_list,file_list[i]))
        for i in range(0,np.size(jpg_list)):
            print("\n"+jpg_list[i], " is the image currently being processed\n")
            Process_Images(jpg_list[i],current_path)
    except Exception as e: # This exception will allow the error message to remain open in the command terminal for debugging
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno+"\n"+e)
        input("Enter anything to close.")
