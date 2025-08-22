import cv2
import numpy as np
import utils

#################
path = './Images/2.jpg'
widImg = 700
heiImg = 700
questions = 5
choices = 5
ans = [0,2,0,1,3]
################## 

img = cv2.imread(path)

# preprocessing.......
img = cv2.resize(img,(widImg, heiImg))
imgControus = img.copy()
imgFinal = img.copy()
imgBiggestControus = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

countrous, heirachy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgControus,countrous,-1,(0,255,0),10)

# find rectangles
rectCorner = utils.rectContour(countrous)
biggestCorner = utils.getCornerPoints(rectCorner[0])
gradePoints = utils.getCornerPoints(rectCorner[1])
# print(biggestCorner)

if biggestCorner.size!=0 and gradePoints.size !=0:
    cv2.drawContours(imgBiggestControus, biggestCorner, -1, (0,255,0), 20)
    cv2.drawContours(imgBiggestControus, gradePoints, -1, (255,0,0), 20)
    biggestCorner = utils.reorder(biggestCorner)
    gradePoints = utils.reorder(gradePoints)
    
    pt1 = np.float32(biggestCorner)
    pt2 = np.float32([[0,0], [widImg,0], [0, heiImg], [widImg, heiImg]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widImg, heiImg))
    
    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0,0], [325,0], [0, 150], [325, 150]])
    matrixG = cv2.getPerspectiveTransform(ptG1,ptG2)
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
    # cv2.imshow("Grade", imgGradeDisplay)
    
    # Apply threshold for finding marking circle, were unmarked -> 0 pixel and marked -> high pixel
    imgWrapGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWrapGray,170,255,cv2.THRESH_BINARY_INV)[1]
    # marked circle
    cv2.imshow("marked circle", imgThresh)
    boxes = utils.splitBoxes(imgThresh)
    
    # getting pixel values of each boxes
    myPixelVal = np.zeros((questions,choices))
    countR = 0
    countC = 0
    for image in boxes:
        totalPixel = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixel
        countC+=1
        if(countC == choices): countR+=1; countC=0
    # print(myPixelVal)
    
    # finding index values of markings....
    myIndex = []
    for x in range(0,questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr==np.amax(arr))
        # print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    print(myIndex)
    
    # Grading....
    grading = []
    for x in range(0,questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    # print(grading)
    score = (sum(grading)/questions)*100
    print(score)
    
    # Display Answers
    imgResult = imgWarpColored.copy()
    imgResult = utils.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
    imgRawDrawing = np.zeros_like(imgWarpColored)
    imgRawDrawing = utils.showAnswers(imgRawDrawing,myIndex, grading, ans, questions, choices)
    inverse_matrix = cv2.getPerspectiveTransform(pt2,pt1)
    imgInWraped = cv2.warpPerspective(imgRawDrawing, inverse_matrix, (widImg, heiImg))
    
    imgRawGrade = np.zeros_like(imgGradeDisplay)
    cv2.putText(imgRawGrade, str(int(score))+"%",(60,100), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,255), 3)
    inverse_matrixG = cv2.getPerspectiveTransform(ptG2,ptG1)
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, inverse_matrixG, (widImg, heiImg))
    
    imgFinal = cv2.addWeighted(imgFinal,1,imgInWraped,1,0)
    imgFinal = cv2.addWeighted(imgFinal,1,imgInvGradeDisplay,1,0)
    
imgBlank = np.zeros_like(img)
imageArray = ([img,imgGray, imgBlur, imgCanny],
               [imgControus,imgBiggestControus,imgWarpColored,imgBlank],
               [imgResult,imgRawDrawing,imgInWraped,imgFinal],
               )
imgStacked = utils.stackImges(imageArray,0.3)

cv2.imshow('Final Images', imgFinal)
cv2.imshow('Stacked Images', imgStacked)
cv2.waitKey(0)