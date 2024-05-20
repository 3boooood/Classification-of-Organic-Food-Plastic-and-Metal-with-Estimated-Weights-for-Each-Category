import cv2 
import numpy as np
import math
from keras.models import load_model


def AreaEstimator(i):
    img = cv2.imread(i)
    CImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,127,180,cv2.THRESH_BINARY_INV)

    contours,ret = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(CImg, contours, -1, (0,255,0), 5)
    #cv2.drawContours(thresh, contours, -1, (0,255,0), 5)
    #cv2.drawContours(gray, contours, -1, (0,255,0), 5)
    
    LArea=0
    for j in range(0,len(contours)):
        area = cv2.contourArea(contours[j])
        if(area>LArea):
            LArea = area
            LIndex = j
            BRect = cv2.boundingRect(contours[j])
   
    x,y,w,h = BRect
    print(LArea,w,h)
    cv2.drawContours( CImg, contours,LIndex, ( 0, 255, 0 ), 5 )
    cv2.imshow("s",CImg)
    depth = math.sqrt((w**2)+(h**2))
    return (LArea,img, gray, thresh, CImg,depth)
def WasteClassifier(i):
    img = cv2.imread(i)
    img = cv2.resize(img,(224,224))
    img = np.array(img)
    img = img / 255.0 # normalize the image
    img = img.reshape(1, 224, 224, 3) # reshape for prediction
    model = load_model("Waste35.h5")
    preds = model.predict(img)
    print(type(preds))
    print(preds)
    preds = preds.tolist()[0]
                    
    pred = preds.index(max(preds))
    print(pred)
    if pred == 0:
        label = 'Organic'
    elif pred == 1:
        label = 'Metal'
    else:
        label = 'Plastic'
    return label
def Pixel2M(area,depth):
    #38924.5 = 58.5cm
    # area * factor * depth / 100 *1000

    volume = area * depth * 0.0015029094 / 100 
    return volume
def WeightEstimator(i):
    Label = WasteClassifier(i)
    Area, im, gr, thr, CI, depth = AreaEstimator(i)
    volume = Pixel2M(Area,depth)
    weight = 0
    match Label:
        case "Organic":
            #.05KG
            mass = 0.05 * volume
            weight = mass * 9.8
        case "Metal":
            #7.784G
            mass = 0.007784 * volume
            weight = mass* 9.8
        case "Plastic":
            #.9G
            mass = 0.0009 * volume
            weight = mass * 9.8
        case _:
            print("Some kind of wierd behaviour happened")
    return weight, im, gr, thr, CI, Label
print(WeightEstimator("Miscellaneous Trash_83.jpg"))



    