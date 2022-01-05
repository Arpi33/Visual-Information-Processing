# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:01:23 2021

@author: 60112
"""

from skimage import io, exposure, morphology, filters, color, \
                    segmentation, feature, measure, img_as_float, img_as_ubyte
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import PIL
from PIL import Image
import os, sys  
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk 
from skimage.segmentation import watershed
#warnings.simplefilter("ignore")
from scipy import ndimage as ndi
from skimage import img_as_ubyte
from sklearn.cluster import KMeans
from skimage.measure import label as skilabel

from scipy import ndimage

### Libraries
import numpy as np
import cv2
from skimage import filters as skifilters
from scipy import ndimage
import skimage
from skimage import filters

from sklearn.cluster import KMeans
from skimage.measure import label as skilabel

from scipy import ndimage

#color moment
#
### Libraries
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm

from sklearn.metrics import confusion_matrix, f1_score, precision_score,  recall_score, accuracy_score, classification_report


"""
resize all images in folder
folder1= "./Eczema/Acute/"
folder2= "./Eczema/Chronic/"
def resize (folder):
    for i in os.listdir(folder):
        file = f"{folder}\\{i}"
        im = Image.open(file)
        im = im.resize((400, 300), Image.ANTIALIAS)
        im.save(file,quality=95)
        
resize(folder1)
resize(folder2)"""

#*****************************************************************************************
#reading image from file(Acute&Chronic)
diseases = [('Acute','acute'),
           ('Chronic','chronic')]

image_db = []
for c in diseases:  
    image_list0 = []
    path = './Eczema/'+ c[0] +'/*.jpg'
    for filename in glob.glob(path):
        im = cv2.imread(filename)
        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        image_list0.append(img)
    image_db.append(image_list0)
    
    
#for fair judgement we take equal number of image sfrom each folder
image_list=[]
image_db_fair = [ic[:396] for ic in image_db]
print(len(image_db_fair[0]))
image_list= np.array(image_db_fair)



#original images list append to another list as backup
count=0
original_images=[]
for i in range(0,len(image_list)):
    for im in range(0,len(image_list[i])):
        input_img=image_list[i][im]
        original_images.append(input_img)
        count+=1
        print(count)

#len(original_images)

#****************************************************************************************

#Image Pre-processing
#****************************************************************************************

#create the mask to filter
def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask


kernel = np.ones((2,2),np.uint8)

#denoise image
@adapt_rgb(each_channel)
def median_filter_each(image):
    return filters.median(image, kernel)

#this function purpose is to remove hair and water mark from image
def occlusion_removal(img,threshold=10,kernel=8,minArea=30000):
    # Remove Dark Hair Occlusions in Dermatoscopic Images via LUV Color Space
    luv       = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)

    # Morphological Closing via Spherical SE
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))
    closing   = cv2.morphologyEx(luv, cv2.MORPH_CLOSE, kernel)

    # Generate Masks via Hysteresis Thresholding Difference Image in L Channel
    diffc      = closing[:,:,0]-luv[:,:,0]
    maskc      = (skifilters.apply_hysteresis_threshold(diffc,threshold,70)).astype(np.uint8)*255
    
    # Remove Side Components
    label_im, nb_labels = ndimage.label(maskc)
    sizes               = ndimage.sum(maskc, label_im, range(nb_labels + 1))
    temp_mask           = sizes > minArea
    maskc               = (temp_mask[label_im]*255).astype(np.uint8)
  
    mask_3dc   = maskc[:,:,None] * np.ones(3,dtype=np.uint8)[None, None, :]
    basec      = cv2.bitwise_not(maskc)
    base_3dc   = basec[:,:,None] * np.ones(3,dtype=np.uint8)[None, None, :]

    # Restitch Preprocessed Image
    preimagec  = ((base_3dc/255)*luv).astype(np.uint8)
    postimagec = ((mask_3dc/255)*closing).astype(np.uint8)
    fullc      = preimagec + postimagec
    outputc    = cv2.cvtColor(fullc, cv2.COLOR_Luv2RGB)

    return outputc, maskc

#Equalize the image contrast using  the Contrast-Limited Adaptive 
#Histogram Equalization (CLAHE) method.
def clahe_LAB(img,clip=0.9,tile=8):
    # Contrast Limited Adaptive Histogram Equalization in LAB Color Space
    lab        = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)                          
    clahe      = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile))     
    lab[:,:,0] = clahe.apply(lab[:,:,0])                                       
    output     = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)                          
    return output


#segmwet the  skin region using threshold and watershed algorithm 
def segment_image(img):
    #converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    lower_HSV_values = np.array([0, 0, 0], dtype = "uint8")
    upper_HSV_values = np.array([60, 255, 255], dtype = "uint8")

    lower_YCbCr_values = np.array((0, 138, 67), dtype = "uint8")
    upper_YCbCr_values = np.array((255, 173, 133), dtype = "uint8")

    #A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
    mask_YCbCr = cv2.inRange(img_YCrCb, lower_YCbCr_values, upper_YCbCr_values)
    mask_HSV = cv2.inRange(img_HSV, lower_HSV_values, upper_HSV_values) 

    binary_mask_image = cv2.add(mask_HSV,mask_YCbCr)
    
    
    knl = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    knl2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    image_foreground = cv2.erode(binary_mask_image, knl,iterations = 1)     	#remove noise
    dilated_binary_image = cv2.dilate(binary_mask_image, knl2,iterations = 8)   #The background region is reduced a little because of the dilate operation
    ret,image_background = cv2.threshold(dilated_binary_image,1,128,cv2.THRESH_BINARY)  #set all background regions to 128

    image_marker = cv2.add(image_foreground,image_background)   #add both foreground and backgroud, forming markers. The markers are "seeds" of the future image regions.
    image_marker32 = np.int32(image_marker) #convert to 32SC1 format

    cv2.watershed(img,image_marker32)
    m = cv2.convertScaleAbs(image_marker32) #convert back to uint8 

    #bitwise of the mask with the input image
    ret,image_mask = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    output= cv2.bitwise_and(img,img,mask = image_mask)
    return output


#function to apply all above methods
def pre_processing(img,seg=False):
    
    # Occlusion Removal
    occ,mask1 = occlusion_removal(img)
    # CLAHE Preprocessing
    cl   = clahe_LAB(occ)
    #denoise
    img=median_filter_each(cl)
    if seg:
    # segment only interested region
        img=segment_image(img)
    return img


#plotting images
new = cv2.imread("demo.jpg")
img = cv2.cvtColor(new, cv2.COLOR_BGR2RGB)

# Occlusion Removal
hr1,mask1 = occlusion_removal(img)

#denoised image
hr2 = median_filter_each(hr1)

# CLAHE Preprocessing
cl1   = clahe_LAB(hr2)

"""plt.figure(figsize=(12,8))
plt.subplot(141),    plt.imshow(img), plt.title('Original Image'), plt.axis("off")
plt.subplot(142),    plt.imshow(hr1), plt.title('Occlusion Removal'), plt.axis("off")
plt.subplot(143),    plt.imshow(hr2), plt.title('Denoised'), plt.axis("off")
plt.subplot(144),    plt.imshow(cl1), plt.title('Enhanced Image'), plt.axis("off") 

#plt.subplot(133),    plt.imshow(cl1), plt.title('Original Image'), plt.axis("off")
plt.axis('off')
plt.savefig('foo.png')  """


plt.imshow(img), plt.title('Original Image'), plt.axis("off")
plt.imshow(hr1), plt.title('Occlusion Removal'), plt.axis("off")
plt.imshow(hr2), plt.title('Denoised'), plt.axis("off")
plt.imshow(cl1), plt.title('Enhanced Image'), plt.axis("off")



new1 = cv2.imread("demo1.jpg")
img = cv2.cvtColor(new1, cv2.COLOR_BGR2RGB)
plt.imshow(img), plt.title('Original Image'), plt.axis("off")
plt.imshow(pre_processing(img,False)), plt.title('Processed image'), plt.axis("off")
plt.imshow(pre_processing(img,True)), plt.title('Segmented image'), plt.axis("off")

#save the segmented images to directory
"""count=1
for i in range(0,len(image_list)):
    for im in range(0,len(image_list[i])):
        input_img=image_list[i][im]
        img = pre_processing(input_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("Segmented/"+str(count)+".jpg",img)
        count+=1 """


#append data without segmentation to see the result comparison in classification 
count=0
pre_process=[]
for i in range(0,len(image_list)):
    for im in range(0,len(image_list[i])):
        input_img=image_list[i][im]
        img = pre_processing(input_img,False)
        pre_process.append(img)
        count+=1
        print(count)

#*************************************************************************************



#*************************************************************************************

#Feature Extraction

#texture feature
def GLCM(image, channel=3, bit_depth=8):  
        GLCM_0  = greycomatrix(image[:,:,0],  [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=2**bit_depth)
        GLCM_1  = greycomatrix(image[:,:,1],  [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=2**bit_depth)
        GLCM_2  = greycomatrix(image[:,:,2],  [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=2**bit_depth)
        contrast_0  = greycoprops(GLCM_0,  'contrast').mean()
        contrast_1  = greycoprops(GLCM_1,  'contrast').mean()
        contrast_2  = greycoprops(GLCM_2,  'contrast').mean()
        dissim_0    = greycoprops(GLCM_0,  'dissimilarity').mean()
        dissim_1    = greycoprops(GLCM_1,  'dissimilarity').mean()
        dissim_2    = greycoprops(GLCM_2,  'dissimilarity').mean()
        correl_0    = greycoprops(GLCM_0,  'correlation').mean()
        correl_1    = greycoprops(GLCM_1,  'correlation').mean()
        correl_2    = greycoprops(GLCM_2,  'correlation').mean()
        homo_0      = greycoprops(GLCM_0,  'homogeneity').mean()
        homo_1      = greycoprops(GLCM_1,  'homogeneity').mean()
        homo_2      = greycoprops(GLCM_2,  'homogeneity').mean()
        return [ contrast_0, dissim_0, correl_0, homo_0, contrast_1, dissim_1,
                 correl_1, homo_1, contrast_2, dissim_2, correl_2, homo_2 ]


def entropyplus(image):   
    histogram         = np.histogram(image, bins=2**8, range=(0,(2**8)-1), density=True)
    histogram_prob    = histogram[0]/sum(histogram[0])    
    single_entropy    = np.zeros((len(histogram_prob)), dtype = float)
    for i in range(len(histogram_prob)):
        if(histogram_prob[i] == 0):
            single_entropy[i] = 0;
        else:
            single_entropy[i] = histogram_prob[i]*np.log2(histogram_prob[i]);
    smoothness   = 1- 1/(1 + np.var(image/2**8))            
    uniformity   = sum(histogram_prob**2);        
    entropy      = -(histogram_prob*single_entropy).sum()
    return smoothness, uniformity, entropy



def entropyplus_3(image):
    smoothness_0, uniformity_0, entropy_0 = entropyplus(image[:,:,0])
    smoothness_1, uniformity_1, entropy_1 = entropyplus(image[:,:,1])
    smoothness_2, uniformity_2, entropy_2 = entropyplus(image[:,:,2])
    return [ smoothness_0, uniformity_0, entropy_0, smoothness_1, uniformity_1, 
             entropy_1, smoothness_2, uniformity_2, entropy_2 ]


### Color Features
def color_moments(image, channel=3):       
        mean_0 = np.mean(image[:,:,0])
        mean_1 = np.mean(image[:,:,1])
        mean_2 = np.mean(image[:,:,2])
        std_0  = np.std(image[:,:,0])
        std_1  = np.std(image[:,:,1])
        std_2  = np.std(image[:,:,2])
        skew_0 = skew(image[:,:,0].reshape(-1))
        skew_1 = skew(image[:,:,1].reshape(-1))
        skew_2 = skew(image[:,:,2].reshape(-1))
        kurt_0 = kurtosis(image[:,:,0].reshape(-1))
        kurt_1 = kurtosis(image[:,:,1].reshape(-1))
        kurt_2 = kurtosis(image[:,:,2].reshape(-1))
        return mean_0, std_0, skew_0, kurt_0, mean_1, std_1, skew_1, kurt_1, mean_2, std_2, skew_2, kurt_2



def extract_features(image,mask=None):    
    # Color Spaces: I/O ------------------------------------------------------------------------------------------------------------------------------------------------------
    img_RGB               = image
    img_HSV               = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_LAB               = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Lab)
    #img_YCrCb             = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)
    circa_mask            = create_circular_mask(image.shape[0], image.shape[1], radius = 300).astype(bool)
    
    masked_lesion_RGB     = np.ma.array(np.multiply(img_RGB,   np.dstack((circa_mask,circa_mask,circa_mask)))  ,mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    masked_lesion_HSV     = np.ma.array(np.multiply(img_HSV,   np.dstack((circa_mask,circa_mask,circa_mask)))  ,mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    masked_lesion_LAB     = np.ma.array(np.multiply(img_LAB,   np.dstack((circa_mask,circa_mask,circa_mask)))  ,mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
   # masked_lesion_YCrCb   = np.ma.array(np.multiply(img_YCrCb, np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))  
    
    # Color Moments ----------------------------------------------------------------------------------------------------------------------------------------------------------------
    mean_R, std_R, skew_R, kurt_R, mean_G,  std_G,  skew_G,  kurt_G,  mean_B,  std_B,  skew_B,  kurt_B   = color_moments(masked_lesion_RGB,     channel=3)
    mean_H, std_H, skew_H, kurt_H, mean_S,  std_S,  skew_S,  kurt_S,  mean_V,  std_V,  skew_V,  kurt_V   = color_moments(masked_lesion_HSV,     channel=3)
    mean_L, std_L, skew_L, kurt_L, mean_A,  std_A,  skew_A,  kurt_A,  mean_b,  std_b,  skew_b,  kurt_b   = color_moments(masked_lesion_LAB,     channel=3)
    #mean_Y, std_Y, skew_Y, kurt_Y, mean_Cr, std_Cr, skew_Cr, kurt_Cr, mean_Cb, std_Cb, skew_Cb, kurt_Cb  = color_moments(masked_lesion_YCrCb,   channel=3)
    


    # Graylevel Co-Occurrence Matrix -------------------------------------------------------------------------------------------------------------------------
    GLCM_RGB   = GLCM(masked_lesion_RGB,   channel=3)
    GLCM_HSV   = GLCM(masked_lesion_HSV,   channel=3)
    GLCM_LAB   = GLCM(masked_lesion_LAB,   channel=3)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------

    
    # Smoothness, Uniformity, Entropy -----------------------------------------------------------------------------
    entropyplus_RGB  = entropyplus_3(masked_lesion_RGB)
    entropyplus_HSV  = entropyplus_3(masked_lesion_HSV)
    #--------------------------------------------------------------------------------------------------------------    
    
    
    features = [ mean_R, std_R, skew_R, mean_G,  std_G,  skew_G,  mean_B,  std_B,  skew_B,   
                 mean_H, std_H, skew_H, mean_S,  std_S,  skew_S,  mean_V,  std_V,  skew_V,   
                 mean_L, std_L, skew_L, mean_A,  std_A,  skew_A,  mean_b,  std_b,  skew_b,
                 #mean_Y, std_Y, skew_Y, mean_Cr, std_Cr, skew_Cr, mean_Cb, std_Cb, skew_Cb
               ]
 
    features = np.concatenate((features, GLCM_RGB, GLCM_HSV, GLCM_LAB,  entropyplus_RGB,entropyplus_HSV),axis=0)

    return features

# Feature Vector Length (Redundancy Check)
#demo = np.array(extract_features(image_list[0][0]))
#print("Feature Vector Shape: "+str(demo.shape))
#demo

#*****************************************************************************************


#reading from segmented image file

image_db1 = []
path = './segmented/*.jpg'
for filename in glob.glob(path):
    im = cv2.imread(filename)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    image_db1.append(img)
    #image_db1.append(image_list)
    
    
len(image_db1)
image_db1 = np.array(image_db1)

#apply feature extraction on segmented images 
eczema=[]
counter=0
for i in range(0,len(image_db1)):
    input_img=image_db1[i]
    features = extract_features(input_img)
    features = np.expand_dims(np.asarray(features),axis=0)
    eczema.append(features)
    counter = counter + 1
    print(counter)


#apply feature extraction on non-segmented images 
clean_img = np.array(pre_process)

eczema2=[]
counter=0
for i in range(0,len(clean_img)):
    input_img=clean_img[i]
    features = extract_features(input_img)
    features = np.expand_dims(np.asarray(features),axis=0)
    eczema2.append(features)
    counter = counter + 1
    print(counter)


data1=np.squeeze(np.array(eczema))  #segmented
data2=np.squeeze(np.array(eczema2)) #not segmented

labels = []
for c in diseases:
    labels.extend([c[0] for i in range(396)])
    
    
    
#train model

X =data2  #data2
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10, stratify=y)



# clf = svm.SVC(gamma='scale', class_weight='balanced')
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

print("Training the SVM classifier...")

param_grid = {'C': [1, 1e1, 1e2, 1e3, 5e3, 1e4],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
              'class_weight': [None, 'balanced']}
clf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
clf = clf.fit(X_train, y_train)

print("Best estimator found by Grid Search:")
print(clf.best_estimator_)

y_pred = clf.predict(X_test)


short_labels = [i[0] for i in diseases]

score = clf.score(X_test, y_test)
score=round(score,2)
print("Test Accuracy: ", score )
y_pred = clf.predict(X_test)

print(" ")
print("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
n_samples = np.sum(cm, axis=1)
dcm = [cm[i]/n for i, n in enumerate(n_samples) ]
print(cm)
#print("Normalized Confusion Matrix")
#print(np.squeeze(dcm))
print(classification_report(y_test, y_pred, labels=short_labels))


import seaborn as sns 
ax= plt.subplot()
sns.heatmap(dcm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(short_labels); ax.yaxis.set_ticklabels(short_labels);


def showImage(img, titlestr=""):
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.xticks([]), plt.yticks([])  
    plt.title(titlestr)

#plot and predict on test data set
def checkPredictions(img_set, img_set_labels, idx, data, original_Images):   
    index = 0  # Index indicator
    for i, img in enumerate(data):                                           
        if (np.array_equal(img_set[idx], img)) == True:  # If the row value is equal to the row value in "data" array.                     
            index = i               # If true, that's the index
            break
            
       # Using the index, can pinpoint which of the actual picture on the "original_Images"
    showImage(original_Images[index])
    print("Prediction:",clf.predict(img_set[idx].reshape(1, -1)))  
    print("Original class:",img_set_labels[idx]) 
    

checkPredictions(X_test, y_test, 9,  data1, original_images)

#apply on unseen data
def predict(img,val=False):
    
    img = cv2.cvtColor(new, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img,(400, 300))
    img1=pre_processing(img1,val)
    img1=extract_features(img1)
    
    res= clf.predict(img1.reshape(1,-1))
    t=str(res[0])
    print("This picture is identified as: ",t)
    plt.imshow(img)  
    
new = cv2.imread("new.jpg")    
predict(new)