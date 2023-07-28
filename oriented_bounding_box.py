import cv2
import json
import os

import numpy as np

folder_path = r'C:\Users\sonali.sahu\Downloads\American Sign Language Poly.v4-v4.yolov5-obb\train\images_annotated'
classes_path = r'C:\Users\sonali.sahu\Downloads\American Sign Language Poly.v4-v4.yolov5-obb\train\images\classes.txt'
folder = os.listdir(folder_path)
file_names = []
for file in folder:
    if file.endswith('.jpg'):
        file_names.append(file[:-4])

for file in file_names:
    img_path = os.path.join(folder_path, file + '.jpg')
    json_path = os.path.join(folder_path, file + '.json')

    img = cv2.imread(img_path)
    with open(json_path) as json_file:
        data = json.load(json_file)

    class_label = data['shapes'][0]['label']
    points_list = data['shapes'][0]['points']
    points = []
    for each_list in points_list:
        points.append(tuple([int(float(each)) for each in each_list]))

    black_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    pts = np.array(points)

    cv2.fillPoly(black_image, [pts], (255, 255, 255))

    canny = cv2.Canny(black_image, 0, 255)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

    count = sorted(contours, key=cv2.contourArea)
    cnt = count[-1]
    rect = cv2.minAreaRect(cnt)
    box_float = cv2.boxPoints(rect)
    box_int = np.intp(box_float)

    img = cv2.drawContours(contour_img, [box_int], 0, (0, 0, 255), 2)
    with open(classes_path, 'r') as text_index:
        lines = text_index.readlines()
        lines_li = []
        for line in lines:
            lines_li.append(line.rstrip('\n'))
        index = lines_li.index(class_label)
        # print(index)
    file_name = file
    full_path = os.path.join(folder_path, file_name+'.txt')
    with open(full_path, 'w+') as file_:
        for point in box_float:
            file_.write(f'{point[0]} {point[1]} ')
        file_.write(f'{class_label} {index}')
    print(file)
    cv2.imshow("Original_Image", img)
    cv2.imshow("black_Image", black_image)
    cv2.imshow("Canny_Image", canny)
    cv2.imshow("Contour", contour_img)
    cv2.waitKey(0)
