
# ********************************************************************************************
# CAMERA_1 & MODEL_1 CODE DETAILS. (CODE DETECTION FROM WAGON AND MAPPING WITH IR DATA)
# ********************************************************************************************

# All Camera's IP address Details:...
# **************************************

# Cam 1 IP = 192.168.2.241
# Cam 2 IP = 192.168.2.242
# Cam 3 IP = 192.168.2.243

# All Camera's User name and Password:...
# *****************************************

# User Name - admin
# Password - Password@123

# Importing required libraries...
import datetime
import cv2 as cv2
import logging
from logging.handlers import TimedRotatingFileHandler
import time
import numpy as np
import re
# from OCR_Model import Easy_OCR
from New_OCR_Process import Easy_OCR
from Data_Mapping import Datamapping
from Data_Mapping import Revers_Replace
from Data_Mapping import Indexplus_Data
from Data_Mapping import Indexminos_Data
from Data_Mapping import Index_selection
from Data_Mapping import break_code_index

# Assigning IP address to variable...
ip = "192.168.2.241"
ip1 = "192.168.2.242"
ip2 = "192.168.2.243"

# Assigning all variable to model_1...

detcode = []
sec = 0.0
framerate = 1.0
out = None
mapping = 0
irmapped = 0
t1 = time.time()
fullcode = ""
last3dig = 0
blankocr = 0
save_image = 0
terminate = 0
OCRresult = 0
couplingcount = 0
digit_9 = 0
digit_10 = 0
allcode = []
Wagon_number = 0
coupling = 0
det = 0
temp_list1 =[]


l2 = datetime.datetime.now()

# LOGGER INFO FOR CODE REFERENCE... (Debug checking)
l1 = datetime.datetime.now()
logger = logging.getLogger("Rotating Log")
logger.setLevel(logging.INFO)
handler = TimedRotatingFileHandler("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Log_Details/Cam_1_Info/logeer_{}.log".format(l1),when="m", interval=60)
logger.addHandler(handler)

# DEFINING EASY_OCR with language English...
# reader = easyocr.Reader(['en'])

print("Start....!")

# Assigning YOLO MODEL into our net variable...
net = cv2.dnn.readNet('/home/vert/PycharmProjects/JSW_ballari_Wagon_Number_Detect/yolov3-config/new/yolov3_training_final.weights',
        '/home/vert/PycharmProjects/JSW_ballari_Wagon_Number_Detect/yolov3-config/new/yolov3_testing.cfg')

# Assigning class for the model to detect...

classes = []
with open("/home/vert/PycharmProjects/JSW_ballari_Wagon_Number_Detect/yolov3-config/classes .txt", "r") as f:classes = f.read().splitlines()

# Using opencv to capture images from camera...
cap = cv2.VideoCapture()

# Camera connection using credentials details...
# cap.open("rtsp://admin:Password@123@{}/Streaming/channels/1/?tcp".format(ip))

# cap.open('/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/Videos/5_video_2021-04-20_Night_02_40_05.330307.avi')
cap.open("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/Videos/Old_video/1_video_2021-04-19_Night_23_28_00.706496.avi")
# cap.open("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/Videos/V1_1.avi")

# cap.open("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_2_INFO/Video/02_video_2021-04-20 16_31_38.426915.avi")

# CREATE TEXT FILE WHERE WE WILL SAVE THE RESULT...
# logger.info("Creating text file")
# touch.touch("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/OCR_Result/{}.txt".format(datetime.datetime.now()))
# logger.info("Text File has been Created")

# Running while loop into the camera_1 to perform our logic...
while True:

    # print("Cam1_alive-{}".format(datetime.datetime.now()))

    # Model is Trying to get live images from camera_1...
    try:
        # GET CURRENT TIME/SEC
        # t2=time.time()
        # if t2-t1>3540:
        #     logger.info("Its 1 hr...will create new video")
        #     t1=time.time()
        #     out=None

        # Reading Images from live camera...
        ref, img = cap.read()
        # print("ll")

        # Code Running Status...
        logger.info("Cam_1_alive Image has been capture-{}".format(datetime.datetime.now()))

        # Checking blank images from live camera...
        if img is None:
            print("Blank Image")
            logger.info("Blank image-{}".format(datetime.datetime.now()))
            continue

        # Model will SKIP every time 2 FRAMES...
        sec = sec + framerate
        if sec % 2 != 0:
            continue

        # Assigning color to the image
        colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

        # font style
        font = cv2.FONT_HERSHEY_PLAIN

        # RESIZING image FRAME to display...
        scale_percent = 40
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)

        # Resize original image to display according to our requirement...
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        height, width, _ = img.shape
        size = (width, height)

        # Taking Height and Weight from resize image...
        (W, H) = (None, None)
        if W is None and H is None:

            # Height and Weight taken from image, using images shape and index...
            (H, W) = img.shape[:2]
            logger.info("Image Height and Weight has been captured")

        # Convert image to blob format and send to vision model to detect object from the image...
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        ln = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(ln)
        logger.info("Image data has been convert into blob format")

        # Assign empty list to append all detected details...
        boxes = []
        confidences = []
        classIDs = []


        # Running for loop into the layer output which came from blob format to know the score...
        for output in layerOutputs:

            for detection in output:
                # logger.info("running for loop in layerOutputs data")

                # Taking score from the detection object...
                scores = detection[5:]

                # Taking max score from the detection object...
                classID = np.argmax(scores)
                confidence = scores[classID]
                # logger.info("confidence score has been capture-{}".format(confidence))
                if confidence >= 0.4 and confidence <= 0.5:
                    logger.info("confidence score has been capture-{}".format(confidence))


                # Getting the confidence score from the detected object...
                if confidence >= 0.5:
                    logger.info("confidence score has been capture-{}".format(confidence))

                    # Creating bounding box into the image...
                    box = detection[0:4] * np.array([W-2, H-7, W-2, H-7])
                    (centerX, centerY, width, height) = box.astype("int")
                    x_a = int(centerX - (width / 2))
                    y_a = int(centerY - (height / 2))

                    # Appending box, confidence and class...
                    boxes.append([x_a, y_a, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    logger.info("Class, Confidence score and Boxes has been appended")

        # Finally here we can come to know Whether model detected any object or not...
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        # If len of indexes is greater then 0 means object detected...
        if len(idxs) == 0:
            terminate += 1

        # If length of idex > then o that means model has detect something...
        if len(idxs) > 0:
            terminate = 0

            # Flatten the data which model has been detected...
            for i in idxs.flatten():

                # Creating bounding box into the code images from the wagon...
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(img, (x, y), (x+1 + w+1, y+3 + h+2), color, 2)
                # print(mapping)
                logger.info("running for loop into blob format")
                logger.info("mapping value-{}".format(mapping))

                # check previously if any code is mapped with IR data. If mapped then code will not go inside the if condition, it will go directly else to continue...
                if mapping != 1:
                    # print("mapping value -{}".format(mapping))

                    # If our model detected code then we need to perform below steps....
                    if "code" == classes[classIDs[i]]:
                        logger.info("Code has been detected from wagon")

                        couplingcount = 0

                        # Wagon_number += 1
                        # Croping code image for OCR

                        img_crop = img[y-5:y-5 + h+15, x-5:x-5 + w+15]
                        # cv2.putText(img_crop, "Code", (10, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        save_image += 1

                        # Once crop done we need to check index shape value for reading easyocr...
                        if img_crop.shape[1] != 0 and img_crop.shape[0] != 0:

                            # Define color code for text
                            color1 = (0, 0, 255)

                            # Writing text in image
                            cv2.putText(img, "Code", (x, y), font, 0.7, color1, 1,cv2.LINE_AA)

                            cv2.imwrite("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Crop_Images/image_{}.jpg".format(save_image), img_crop)

                            logger.info("Only code part has been croped from image and send to OCR")
                            # result = reader.readtext(img_crop)

                            # Easy ocr processin
                            codex = Easy_OCR(img_crop)
                            if codex:
                                if len(codex) > 0:
                                    temp_list1 = re.findall('\d+', codex)
                                    print("Easy OCR result-{}".format(temp_list1))

                                    allcode.append(temp_list1)
                                    # maxc = max(allcode, key=allcode.count)
                                # print("maximum number coming -{}".format(maxc))
                                # if len(allcode)> 0:
                                #     print(allcode)
        #                     # logger.info("Easy ocr has been processed")
        #
        #                     with open("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/OCR_Result/OCR_Result {}.txt".format(l2), "a") as f:
        #                         f.write("\n")
        #                         f.write(str(codex))
        #                         f.write("\n")
        #
        #                     # Data mapping with IR data
        #                     # path = "/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/Videos/Old_video/5_video_2021-04-20_Night_02_40_05.330307.csv"
        #                     # path = "/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/Videos/Old_video/3_video_2021-04-20_Night_00_42.csv"
        #                     path  = ("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/IR_Data/Master sheet_15 (copy).xlsx")
        #
        #                     # print("mapping start with ir data")
        #                     mappingcode  = Datamapping(path, codex)
        #
        #                     # detcode.append(mappingcode)
        #
        #                     finalcodemaping = Revers_Replace(str(mappingcode))
        #                     final = finalcodemaping[4:-2]
        #
        #                     print("Mapping IR Code-{}".format(final))
        #
        #                     if mappingcode:
        #                         detcode.append(final)
        #                         # print(detcode)
        #
        #                         mapping = 1
        #
        #                     # print("Mapping value before-{}".format(mapping))
        #                     logger.info("ir data has been mapped")
        #
        #             else:
        #
                    if "coupling" == classes[classIDs[i]]:
                        couplingcount += 1
                        if couplingcount == 1:
                            print("coupling has been detected")
                            maxc = max(allcode, key=allcode.count)
                            print("maximum number coming -{}".format(maxc))
                            allcode = []
                            # print("after restart all code value-{}".format(allcode))
    #
        #                     # Writing coupling in to the image
        #                     cv2.putText(img, "Coupling", (x, y), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        #
        #                     couplingcount += 1
        #                     logger.info("Coupling has been detected from Rake... restarting all variable")
        #                     print("Coupling has been detected from Rake... restart all variable...........")
        #                     # print("Mapping value-{}".format(mapping))
        #
        #                     if couplingcount == 1 and mapping == 0:
        #
        #                         detfirstvalue = detcode[0]
        #
        #                         path = ("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/IR_Data/Master sheet_15 (copy).xlsx")
        #
        #
        #                         #  Getting first detection index number
        #                         v = Index_selection(path,str(detcode[0]) )
        #
        #                         if v <= 4:
        #
        #                             z = Indexplus_Data(path, detcode)
        #                             detcode.append(z)
        #                             print(detcode)
        #                             print("Mapping IR code-{}".format(z))
        #
        #                             mapping = 1
        #
        #                         else:
        #
        #                             if mapping == 0:
        #
        #                                 z = Indexminos_Data(path, detcode)
        #                                 detcode.append(z)
        #                                 print(detcode)
        #                                 print("Mapping IR code-{}".format(z))
        #
        #                                 mapping = 1
        #
        #                     if couplingcount == 1:
        #
        #                         with open("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/OCR_Result/OCR_Result {}.txt".format(l2), "a") as f:
        #                             f.write("\n")
        #                             f.write("*" * 20)
        #                             f.write("\n")
        #
        #                     # Restarting all variable...
        #                     mapping = 0
        #                     codelist = []
        #                     allcode = []
        #                     sec = 0.0
        #                     framerate = 1.0
        #                     last3dig = 0
        #                     out = None
        #
        #                     continue
        #
        #         else:
        #
        #             if "coupling" == classes[classIDs[i]]:
        #
        #                 # Writing coupling in to the image
        #                 cv2.putText(img, "Coupling", (x, y), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        #
        #                 couplingcount += 1
        #
        #                 logger.info("Coupling has been detected from Rake..restart all variable")
        #                 print("Coupling has been detected from Rake..restart all variable")
        #                 print("Mapping value-{}".format(mapping))
        #
        #                 if couplingcount == 1 and mapping == 0:
        #
        #                     detfirstvalue = detcode[0]
        #
        #                     path = ("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/IR_Data/Master sheet_15 (copy).xlsx")
        #
        #                     v = Index_selection(path, str(detcode[0]))
        #
        #                     if v <= 4:
        #
        #                         z = Indexplus_Data(path, detcode)
        #                         detcode.append(z)
        #                         print(detcode)
        #                         print("Mapping IR code-{}".format(z))
        #
        #                         mapping = 1
        #
        #                     else:
        #
        #                         if mapping == 0:
        #
        #                             print("else block")
        #
        #                             z = Indexminos_Data(path, detcode)
        #                             detcode.append(z)
        #                             print(detcode)
        #                             print("Mapping IR code-{}".format(z))
        #
        #                             mapping = 1
        #
        #                 if couplingcount == 1:
        #
        #                     with open("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/OCR_Result/OCR_Result {}.txt".format(l2), "a") as f:
        #                         f.write("\n")
        #                         f.write("*" * 20)
        #                         f.write("\n")
        #
        #                 # Restarting all variable...
        #                 mapping = 0
        #                 codelist = []
        #                 allcode = []
        #                 sec = 0.0
        #                 framerate = 1.0
        #                 last3dig = 0
        #                 out = None
        #
        #                 continue
        #
        #                 # continue
        # # print("mnj")
        cv2.imshow("Video Image", img)
        cv2.waitKey(1)

    # Get exception when any issue happen in above process...
    except Exception as e:
        print("Theres an exception....-{}".format(e))