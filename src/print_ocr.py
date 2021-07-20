from google.cloud import vision
import io
import os
import re
import glob
import string
import json
import boto3
import csv
import cv2
import numpy as np
import math

# -*- coding: utf-8 -*-

# Configure environment for google cloud vision
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "client_secrets.json"

# Create a ImageAnnotatorClient
VisionAPIClient = vision.ImageAnnotatorClient()

path = r'C:\Users\User\Documents\prescription-ocr\data\pages'

def word_segment(image, word_result, image_name):
    word_list = []
    box_list = []
    for (idx, item) in enumerate(word_result[1:]):
        word_list.append(item.description)
        box_list.append(item.bounding_poly)
        clone = image.copy()
        p1x, p1y = item.bounding_poly.vertices[0].x, item.bounding_poly.vertices[0].y
        p2x, p2y = item.bounding_poly.vertices[1].x, item.bounding_poly.vertices[1].y
        p3x, p3y = item.bounding_poly.vertices[2].x, item.bounding_poly.vertices[2].y
        p4x, p4y = item.bounding_poly.vertices[3].x, item.bounding_poly.vertices[3].y
        cnt = np.array([
            [[p1x, p1y]],
            [[p2x, p2y]],
            [[p3x, p3y]],
            [[p4x, p4y]]
        ])

        angle = np.rad2deg(np.arctan2(item.bounding_poly.vertices[2].y - item.bounding_poly.vertices[3].y,
         item.bounding_poly.vertices[2].x - item.bounding_poly.vertices[3].x))
        # print(item.description)
        # print(angle)
        x_max = max(p1x,p2x, p3x, p4x)
        x_min = min(p1x,p2x, p3x, p4x)
        y_max = max(p1y,p2y, p3y, p4y)
        y_min = min(p1y,p2y, p3y, p4y)

        cx = x_min + (x_max-x_min)/2
        cy = y_min + (y_max-y_min)/2
        
        height = math.sqrt((p4x-p1x) * (p4x-p1x) + (p4y-p1y) * (p4y-p1y)) * 1.1
        width = math.sqrt((p4x-p3x) * (p4x-p3x) + (p4y-p3y) * (p4y-p3y)) + height * 0.2

        rect = ((cx, cy), (width, height), angle)
        # print(rect)

        # the order of the box points: bottom left, top left, top right, bottom right
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # print("bounding box: {}".format(box))

        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # corrdinate of the points in box points after the rectangle has been straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(clone, M, (width, height))

        (wt, ht) = (128, 32)
        (h, w) = (warped.shape[0], warped.shape[1])
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
        img = cv2.resize(warped, newSize)
        target = np.ones([ht, wt]) * 255
        target[0:newSize[1], 0:newSize[0]] = img
        
        cv2.imwrite('out/{}_{}.png'.format(image_name, idx), target)
        # print(idx, image_name)
        cv2.waitKey(0)

for filename in glob.glob(os.path.join(path, '*.*')):

    with io.open(filename, 'rb') as image_file:
        content = image_file.read()

    # Send the image content to vision and stores text-related response in text
    # pylint: disable=no-member
    image = vision.types.Image(content=content)
    response = VisionAPIClient.document_text_detection(image=image, image_context={"language_hints": ["en"]})

    document = response.full_text_annotation
    word_result = response.text_annotations
    print(document)
    image = cv2.imread(filename, 0)
    
    image_name = os.path.basename(filename).split('.')[0]
    word_segment(image, word_result,image_name)

    
    # to identify and compare the break object (e.g. SPACE and LINE_BREAK) obtained in API response
    breaks = vision.enums.TextAnnotation.DetectedBreak.BreakType

    # generic counter
    c = 0

    # List of lines extracted
    lines = []

    # List of corresponding confidence scores of lines
    confidence = []

    # Initialising list of lines
    lines.append('')

    # Initialising list of confidence scores
    confidence.append(2)

    # Loop through all symbols returned and store them in lines list alongwith
    # corresponding confidence scores in confidence list
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        lines[c] = lines[c] + symbol.text
                        if re.match(r'^[a-zA-Z]+\Z', symbol.text) or symbol.text.isdigit():
                            confidence[c] = min(confidence[c], symbol.confidence)
                        if symbol.property.detected_break.type == breaks.LINE_BREAK or \
                                symbol.property.detected_break.type == breaks.EOL_SURE_SPACE:
                            c += 1
                            lines.append('')
                            confidence.append(2)
                        elif symbol.property.detected_break.type == breaks.SPACE or \
                                symbol.property.detected_break.type == breaks.SURE_SPACE:
                            lines[c] = lines[c] + ' '

    # Total number of lines
    linecount = len(lines)

    # Initialising all variables
    raw = ''  # To store all lines
    checktext = ''  # Generic string variable to store surrounding lines

    # Loop through all lines to check for all required fields like aadhar no. , date of birth, address and so on
    for index, line in enumerate(lines):

        # To store all lines for exporting later as raw output
        raw = raw + line + "\n"

        # Total number of characters in line
        length = len(line)


    # print(raw)
    client = boto3.client(service_name='comprehendmedical', region_name='us-east-1')
    result = client.detect_entities_v2(Text= raw)
    entities = result['Entities']
    medication = []
    medical_condition = []
    phi = []
    ttp = []
    anatomy = []
    te = []

    for entity in entities:
        # print('Entity', entity)
        attribute = ''
        if 'Attributes' in entity:
            for item in entity['Attributes']:
                attribute = attribute + ", " + item['Type'] + ":" + item['Text'] + ", score: " + str(round(item['Score'], 2))

        text_data = str(entity['Type']) + ": " + str(entity['Text']) + ", score: " + str(round(entity['Score'], 2)) + attribute
        
        # not GENERIC_NAME: VISAKHAPATNAM, BRAND_NAME: Kolkata, GENERIC_NAME: Nagerbazar, NAME: D. M., NAME: M.D, 
        # TEST_NAME: Kolkata TEST_NAME: Ph TEST_NAME: Sat NAME: Kolkata DOC_DETAILS: MBBS, MD, DM - Gastroenterology
        # BRAND_NAME: MS Ramaiah Nagar
        if entity['Category'] == 'MEDICATION' and entity['Score'] > 0.5:
            medication.append(text_data)
        
        if entity['Category'] == 'MEDICAL_CONDITION' and entity['Score'] > 0.5:
            medical_condition.append(text_data)

        if entity['Category'] == 'PROTECTED_HEALTH_INFORMATION' and entity['Score'] > 0.5:
            if entity['Text'].isdigit() and (1000000000 <= int(entity['Text']) <= 9999999999):
                phi.append("PHONE: "+ str(entity['Text']))
            elif any(item in entity['Text'] for item in ['Hospital', 'HOSPITAL']):
                phi.append("HOSPITAL_NAME: "+ str(entity['Text']))
            else:
                phi.append(text_data)
                

        if entity['Category'] == 'TEST_TREATMENT_PROCEDURE' and entity['Score'] > 0.5:
            ttp.append(text_data)

        if entity['Category'] == 'ANATOMY' and entity['Score'] > 0.5:
            anatomy.append(text_data)

        if entity['Category'] == 'TIME_EXPRESSION' and entity['Score'] > 0.5:
            te.append(text_data)
    
    line = io.StringIO()
    writer = csv.writer(line)
    
    with open('text.csv', 'a', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        if medication:
            writer.writerow([os.path.basename(filename), "MEDICATION"] + medication)
        if medical_condition:
            writer.writerow([os.path.basename(filename), "MEDICAL_CONDITION"] + medical_condition)
        if phi:
            writer.writerow([os.path.basename(filename), "PROTECTED_HEALTH_INFORMATION"] + phi)
        if ttp:
            writer.writerow([os.path.basename(filename), "TEST_TREATMENT_PROCEDURE"] + ttp)
        if anatomy:
            writer.writerow([os.path.basename(filename), "ANATOMY"] + anatomy)
        if te:
            writer.writerow([os.path.basename(filename), "TIME_EXPRESSION"] + te)
        writer.writerow([])
    
    