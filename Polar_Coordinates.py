import numpy as np
from time import time
import cv2
import argparse
import random as rng
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
import math
from googleapiclient.discovery import build
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pygsheets
from matplotlib import pyplot as plt

t0 = time()
rng.seed(12345)

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(creds)

SPREADSHEET_ID = '1L2VWtasQRDJmJpyNOIgYb428CDCmd-vCOT0sgNhjc1k'
worksheet_name = 'ulcer_sheet'

def thresh_callback(image, thresh):

    threshold = thresh
    canny_output = cv2.Canny(image, threshold, threshold * 2)


    contours,hierachy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the rotated rectangles and ellipses for each contour
    minRect = [None]*len(contours)
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv2.fitEllipse(c)
    # Draw contours + rotated rects + ellipses

    ellipses_points = np.array(minEllipse)
    Rect_Points = np.array(minRect)

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i, c in enumerate(contours):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # contour
        cv2.drawContours(drawing, contours, i, color)
        # ellipse
        if c.shape[0] > 5:
            cv2.ellipse(drawing, minEllipse[i], color, 2)
        # rotated rectangle
        box = cv2.boxPoints(minRect[i])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv2.drawContours(drawing, [box], 0, color)

    for (xA, yA) in list(box):
        # draw circles corresponding to the current points and
        cv2.circle(drawing, (int(xA), int(yA)), 9, (0,0,255), -1)
        cv2.putText(drawing, "({},{})".format(xA, yA), (int(xA - 50), int(yA - 10) - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    return Rect_Points,ellipses_points,canny_output

def add_border(input_image, output_image, border):
    img = Image.open(input_image)
    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border)
    else:
        raise RuntimeError('Border is not an integer or tuple!')
    bimg.save(output_image)
    return bimg

def imsmooth(img, kernel_value):

    # defining a size to search around a point for anomolies to fix
    kernel = np.ones((kernel_value,kernel_value), np.uint8)

    # defining larger kernel for use in the closing process
    kernel_value2 = int(kernel_value*10)
    kernel2 = np.ones((kernel_value2,kernel_value2), np.uint8)

    # noise removal from the edges of the image
    img_erosion = cv2.erode(img, kernel, iterations=1)
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

def make_BandW(originalImage,kernel_value,iterations,case):
    print(originalImage[0][0])
    if case == "foot":
        cv2.imwrite('enlarged.jpg',originalImage)
        originalImage = add_border("enlarged.jpg", output_image='enlarged.jpg',
                border=100)
        originalImage = np.array(originalImage)
    print(originalImage[0][0])
    # cleaning kmeans image to be binary for smoothing methods
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    # calling smoothing function
    blackAndWhiteImage = imsmooth(blackAndWhiteImage,kernel_value)
    if case == "foot":
        cv2.imwrite('foot_stage_four.jpg',blackAndWhiteImage)
    elif case == "ulcer":
        cv2.imwrite('ulcer_stage_four.jpg',blackAndWhiteImage)
    return blackAndWhiteImage

def Denoised(image):

    # using denoising method to remove anomolies in the foot
    denoised = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    return denoised

def Kmeans_run(imname,k):

    # reading image
    image = cv2.imread(imname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # calling denoising function
    image = Denoised(image)

    # cleaning image background to improve kmeans effectiveness
    row = np.size(image, axis = 0)
    column = np.size(image, axis = 1)
    for i in range(0,row):
        for j in range(0,column):
            if sum(np.array(image[i][j])) >= 700:
                image[i][j] = [255,255,255]

    im = Image.fromarray(image)
    im.save('1st_step.jpg')

    # converting 3D image rgb array into on dimensional list of RGB points for clustering
    pixel_values = image.reshape((-1,3))
    pixel_values = np.float32(pixel_values)

    # finding center points and making list of pixels in terms of which cluster theyre in
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()

    # creating seperate list for processing ulcer and feet seperately. For some reason the np.array() part
    # was needed to stop changing one from changing the other
    centers_2 = np.array(centers)

    # calculations to find the background and ulcer clusters respectively and find which 'label' relates
    a_list = []
    b_list = []
    for i in range(0,k):
        if np.size(a_list) == 0:
            a_list = np.array(np.sum(centers[i]))
            b = int(centers[i][0])
            c = int(centers[i][1])
            d = int(centers[i][2])
            b_list = np.array(b-c-d)
        else:
            a_new = np.sum(centers[i])
            a_list = np.hstack((a_list,a_new))

            b = int(centers[i][0])
            c = int(centers[i][1])
            d = int(centers[i][2])
            e = np.array(b-c-d)
            b_list = np.hstack((b_list,e))


    a_index = a_list.argmax()
    b_index = b_list.argmax()

    # adjust center RGB values to black and white for binary image
    for i in range(0,k):
        if i == a_index:
            centers[i] = np.array([0,0,0])
        else:
            centers[i] = np.array([255,255,255])

    for i in range(0,k):
        if i == b_index:
            centers_2[i] = np.array([255,255,255])
        else:
            centers_2[i] = np.array([0,0,0])

    # reform images to original size post processing
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    ulcer_image = centers_2[labels.flatten()]
    ulcer_image = ulcer_image.reshape(image.shape)

    # calling make_BandW function for both ulcer and foot
    foot_image = make_BandW(segmented_image,10,5,"foot")
    ulcer_image = make_BandW(ulcer_image,5,5,"case")

    # performing bounding processes on foot profile and ulcer profile to get dimensions and relative location
    Rfoot_points,Efoot_points,foot_canny = thresh_callback(foot_image,100)
    Rulcer_points,Eulcer_points,ulcer_canny = thresh_callback(ulcer_image,100)

    return Rfoot_points[0],Eulcer_points[0],foot_image,foot_canny

def findMiddle(input_list,max_row_range,row_ran,column_start):
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

def splitList(_list):
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

def Sorter(_list_,*argv):
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
    except UnboundLocalError:
        sorted_array = sorted_array[::-1]
        return sorted_array
    except IndexError:
        sorted_array = np.array([_list_[0]-row_ran,
                                 _list_[1]-column_start])
    return sorted_array

def Find_Point(im_list,image,i,j):
    point = np.array([i,j,image[i][j]])
    if np.size(im_list) == 0:
        im_list = point
    else:
        im_list = np.vstack((im_list, point))
    return im_list #done

def find_width_and_height(alt_rotate):
    rows = np.size(alt_rotate, axis=0)
    columns = np.size(alt_rotate, axis=1)
    row_list = []
    for i in range(0,rows):
        found = False
        for j in range(0,columns):
            if found == True:
                break
            elif alt_rotate[i][j] != 0:
                row_list.append(i)
                found = True

    row_range = np.array([min(row_list), -min(row_list)+max(row_list),
                          int((max(row_list)-min(row_list))/10),max(row_list)])

    column_list = []
    for i in range(0,columns):
        found = False
        for j in range(0,rows):
            if found == True:
                break
            elif alt_rotate[j][i] != 0:
                column_list.append(i)
                found = True

    column_start = min(column_list)
    column_dist = max(column_list)-min(column_list)
    return row_range,column_start,column_dist # done

def rotate_image(mat, angle):
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
    row_range,column_start,column_dist = find_width_and_height(rotated_mat)
    print(column_dist, row_range[1])
    if column_dist > row_range[1]:
        rotated_mat = rotate_image(rotated_mat, -90)

    return rotated_mat #done

def Find_Side(x_centre,y_centre,foot_angle,im):
    """function to find which side of the foot is in the scan"""

    gradient_angle = math.radians(90+foot_angle)
    gradient = math.tan(gradient_angle)
    constant = y_centre - gradient*x_centre

    columns = np.size(im, axis = 0)
    rows = np.size(im, axis = 1)

    side = np.array([0,0])

    for i in range(0,columns):
        for j in range(0,rows):
            if i-gradient*j < constant:
                if im[i][j] == 1:
                    side[1]+=1
            else:
                if im[i][j] == 1:
                    side[0]+=1

    if side[0]>side[1]:
        side = "left"
    else: side = "right"

    return side #done

def Polar_Mapper(canny):
    row = np.size(canny,axis=0)
    column = np.size(canny,axis=1)
    ones = 0
    zeros = 0

    coords = []

    for i in range(0,row):
        for j in range(0,column):
            if canny[i][j] == 255:
                x_diff = i - Fpoints[0][1]
                y_diff = j - Fpoints[0][0]
                if x_diff == 0 and y_diff < 0:
                    theta = 360+Fpoints[2]+90
                elif x_diff == 0 and y_diff > 0:
                    theta = 180+Fpoints[2]+90
                elif y_diff == 0 and x_diff > 0:
                    theta = 90+Fpoints[2]+90
                elif y_diff == 0 and x_diff < 0:
                    theta = 270+Fpoints[2]+90
                elif x_diff >= 0 and y_diff < 0:
                    theta = math.degrees(math.atan((x_diff/y_diff)))+90+Fpoints[2]+90
                elif x_diff >= 0 and y_diff > 0:
                    theta = math.degrees(math.atan((x_diff/y_diff)))+90+Fpoints[2]+90
                elif x_diff <= 0 and y_diff > 0:
                    theta = math.degrees(math.atan((x_diff/y_diff)))+270+Fpoints[2]+90
                elif x_diff <= 0 and y_diff < 0:
                    theta = math.degrees(math.atan((x_diff/y_diff)))+270+Fpoints[2]+90
                theta = int(theta)
                polar_dist = math.sqrt(abs(x_diff)**2+abs(y_diff)**2)
                if np.size(coords) == 0:
                    theta_list = [theta]
                    coords = [x_diff,y_diff,theta,polar_dist]
                else:
                    if theta not in theta_list:
                        theta_list = np.hstack((theta_list,theta))
                        coords_new = [x_diff,y_diff,theta,polar_dist]
                        coords = np.vstack((coords,coords_new))

    for i in range(0,np.size(theta_list)):
        if coords[i,[2]] < 0:
            coords[i,[2]] += 360
        if coords[i,[2]] >360:
            coords[i,[2]] -= 360

    coords = np.array(sorted(coords, key=lambda x:x[2]))

    x = coords[:,[2]].ravel()
    y = coords[:,[3]].ravel()
    y = np.gradient(y)
    y = np.gradient(y)
    plt.plot(x,y)
    plt.show()

Fpoints,Upoints,image,canny = Kmeans_run('foot.jpg',6)
relative_angle = Upoints[2]-Fpoints[2]

x_change = Fpoints[0][0]-Upoints[0][0]
y_change = Fpoints[0][1]-Upoints[0][1]

# depending on how the image has been recorded, different orientations couls be used.
if Fpoints[1][0] > Fpoints[1][1]:
    foot_length = Fpoints[1][0]
    foot_width = Fpoints[1][1]
    ulcer_length = Upoints[1][0]
    ulcer_width = Upoints[1][1]
    if x_change > 0 and y_change > 0:
        polar = -math.degrees(math.atan(np.abs(x_change)/np.abs(y_change))) - Fpoints[2] + 90
    elif x_change < 0 and y_change > 0:
        polar = math.degrees(math.atan(np.abs(x_change)/np.abs(y_change))) - Fpoints[2] + 90
    elif x_change < 0 and y_change < 0:
        polar = 180-math.degrees(math.atan(np.abs(x_change)/np.abs(y_change))) - Fpoints[2] + 90
    elif x_change > 0 and y_change < 0:
        polar = 180+math.degrees(math.atan(np.abs(x_change)/np.abs(y_change))) - Fpoints[2] + 90
else:
    foot_length = Fpoints[1][1]
    foot_width = Fpoints[1][0]
    ulcer_length = Upoints[1][1]
    ulcer_width = Upoints[1][0]
    if x_change > 0 and y_change > 0:
        polar = -math.degrees(math.atan(np.abs(x_change)/np.abs(y_change))) - Fpoints[2]
    elif x_change < 0 and y_change > 0:
        polar = math.degrees(math.atan(np.abs(x_change)/np.abs(y_change))) - Fpoints[2]
    elif x_change < 0 and y_change < 0:
        polar = 180-math.degrees(math.atan(np.abs(x_change)/np.abs(y_change))) - Fpoints[2]
    elif x_change > 0 and y_change < 0:
        polar = 180+math.degrees(math.atan(np.abs(x_change)/np.abs(y_change))) - Fpoints[2]

#Finding which side foot is being looked at
side = Find_Side(Fpoints[0][0],Fpoints[0][1],Fpoints[2],image)

center_dist = math.sqrt((x_change)**2+(y_change)**2)
measurements = [foot_width,foot_length,center_dist,polar,ulcer_width,ulcer_length,relative_angle,side]

angle = Fpoints[2]
print(angle)
image = canny

alt_rotate = rotate_image(image,angle)
alt_rotate2 = Image.fromarray(alt_rotate)
alt_rotate2.save('rotated.jpg')
rows = np.size(alt_rotate, axis=0)
columns = np.size(alt_rotate, axis=1)

row_range,column_start,column_dist = find_width_and_height(alt_rotate)

Bottom_List = []
Top_List = []
whole_points = []
for i in range(row_range[0], row_range[0]+row_range[2]*10+1, row_range[2]):
    line_points = []
    for j in range(0,columns):
        if alt_rotate[i][j] != 0:
            points = np.array([i, j,alt_rotate[i][j]])
            if i == row_range[0]:
                Top_List = Find_Point(Top_List,alt_rotate,i,j)
            elif i == max(range(row_range[0], row_range[0]+row_range[2]*10+1,
                                row_range[2])):
                Bottom_List = Find_Point(Bottom_List,alt_rotate,i,j)
            else:
                if np.size(line_points) == 0:
                    line_points = points
                else:
                    line_points = np.vstack((line_points, points))

    if np.size(line_points) > 0:
        print(line_points)
        list1,list2 = splitList(line_points)
        point1 = Sorter(list1,row_range[0],column_start)
        point2 = Sorter(list2,row_range[0],column_start)
        current = np.array([[point1[0],point1[1]],[point2[0],point2[1]]])
        if np.size(whole_points) == 0:
            whole_points = current
        else:
            whole_points = np.vstack((whole_points,current))

whole_points = np.vstack((findMiddle(Top_List,row_range[3],
                                     row_range[0],column_start),
                          whole_points,
                          findMiddle(Bottom_List,row_range[3],
                                     row_range[0],column_start)))

order = np.array([[0],[19],[1],[18],[2],[17],
         [3],[16],[4],[15],[5],[14],
         [6],[13],[7],[12],[8],[11],
         [9],[10]])
whole_points = np.hstack((whole_points, order))
whole_points = Sorter(whole_points)
whole_points = np.delete(whole_points,2,axis=1)
whole_points = whole_points.ravel()
whole_points = whole_points.tolist()
measurements = measurements + whole_points

def find_sheet_id_by_name(sheet_name):
    # ugly, but works
    sheets_with_properties = API \
        .spreadsheets() \
        .get(spreadsheetId=SPREADSHEET_ID, fields='sheets.properties') \
        .execute() \
        .get('sheets')

    for sheet in sheets_with_properties:
        if 'title' in sheet['properties'].keys():
            if sheet['properties']['title'] == sheet_name:
                return sheet['properties']['sheetId']

sheet = client.open('ulcer_sheet').sheet1
#list_hashes = sheet.get_all_records()
#print(list_hashes)

def push_data_to_gsheet(dataContents, sheet_id):

    gc = pygsheets.authorize(service_file='client_secret.json') #authorization
    worksheet = gc.open('ulcer_sheet').sheet1 #opens the first sheet in "Sign Up"

    cells = worksheet.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
    end_row = len(cells) # THIS FINDS THE INDEX NUMBER OF AN EMPTY CELL

    dataContents = np.array(dataContents)
    dataContents = ','.join(str(n) for n in dataContents)

    body = {
        'requests': [{
            'pasteData': {
                "coordinate": {
                    "sheetId": sheet_id,
                    "rowIndex": end_row,  # means it appends onto an empty row
                    "columnIndex": "0",
                 },
                "data": dataContents,
                    "type": 'PASTE_NORMAL',
                    "delimiter": ',',
            }
        }]
    }
    request = API.spreadsheets().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=body)
    response = request.execute()
    return response

API = build('sheets', 'v4', credentials=creds)
push_data_to_gsheet(
        dataContents=measurements,
        sheet_id=find_sheet_id_by_name(worksheet_name)
)
