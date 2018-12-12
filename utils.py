import keras
from keras.models import *
from keras.layers import *
from keras import backend as k
from keras.optimizers import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import pickle
import cv2
from sklearn.utils import shuffle
import glob
from PIL import Image
import time
import inflect
import requests
import json
import random
import sys
from operator import itemgetter
import re
#to match substring in string
import fuzzysearch
from fuzzysearch import find_near_matches

#parallel computing
from multiprocessing import Pool

#structured data from text
from pdf2image import convert_from_path

#url open to get image
import urllib.request
from urllib.request import urlopen

#for data augmentation
import imgaug as ia
from imgaug import augmenters as iaa

#plot
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import matplotlib.patches as patches

###################################################################################################
###################################### download data from www #####################################
###################################################################################################

#download image with url if not already in the folder
def get_image(url, path, name):
    k = 0
    #if we dont already have it, download it
    if len(glob.glob(path))<1:
        #download until get image or until 10 trials of same image
        while k<10:
            try:
                response = None
                response = urlopen(url)
                img = Image.open(response)
                img.save(path)
                k = 10
                del img
            except KeyboardInterrupt:
                    raise
            except Exception as e:
                if response is not None:
                    if response.getcode() in [200, 404]: 
                        print('Not able to SAVE image for species %s and url %s, lets STOP due to: \n %s'%(name,str(url),e))
                        print(response.getcode())
                        k = 10
                    else:
                        print('Not able to SAVE image for species %s and url %s, lets RETRY due to: \n %s'%(name,str(url),e))
                        print(response.getcode())
                        k = k+1 
                        time.sleep(5)
                else: 
                        print('Not able to SAVE image for species %s and url %s, lets STOP due to: \n %s'%(name,str(url),e))
                        k = 10  #e.g. HTTP Error 404: Not Found

                        
#search wikipedia translation of the title
#more parameter at: https://www.mediawiki.org/w/api.php?action=help&modules=query%2Blanglinks
def search_wikipedia_laguage(text, language='en'):
    #wiki query with properties 'langlinks' to ask for all opssible language translation given on that page with limit to 500 language
    #(default is 10, 500 is maximum)
    url = 'https://'+language+'.wikipedia.org/w/api.php?action=query&format=json&prop=langlinks&lllimit=500&llprop=langname|autonym&titles=%s&redirects=1'% requests.utils.quote(text)
    while True:
        try:
            #call API
            content = requests.get(url).content
            content = json.loads(content)
            #content[1][0].upper()==text.upper(): #if exact match in the title
            return(content)
        except KeyError:
            print('species %s failed, try again' % text)
#for more general info (sentences, url ...)
#url = 'https://'+language+'.wikipedia.org/w/api.php?action=opensearch&search=%s&limit=1&namespace=0&format=json&redirects=resolve&prop=langlinks' % requests.utils.quote(text)
#note that we have put 'limit=1' as we are searching for the exact amtch (car pour le reste on va faire avec d'autre
#technique comme la distance entre le smots etc)
#to search wiki sumamry: wikipedia.summary(x)

    
#used for allrecipes.com for example    
# Get HTML content if not already existing in our files (not to request it two times)
def get(id, url, path_):
    # Check if file html was already dowloaded
    cache_path = os.path.join(path_, 'cache')
    path = os.path.join(cache_path, '%d.html' % id)
    if os.path.exists(path):
        with open(path, 'rb') as file:
            content = file.read()
    # Otherwise get page content
    else:
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        while True:
            try:
            # Write cache
                with open(path, 'wb') as file:
                    page = requests.get(url+str(id), timeout=5)
                    content = page.content
                    file.write(content)
                break
            except KeyboardInterrupt:
                raise
            except:
                print('page %d failed, try again' % id)
    return content

def parse(content,end_of_title,encoding='utf-8'):
    # Check if page is valid
    try:
        tree = html.fromstring(content.decode(encoding))
        title = tree.xpath('head/title')[0].text
        if not title.endswith(end_of_title):
            return None
    except :
        print('error')
        tree=None
    return tree    
    
    
    
###################################################################################################
#################################### structured data from text ####################################
###################################################################################################

#take as input an image ( np.array or PIL image)
def frompng2images(img, path, page_id=0, plot_=0):
    
    #convert to numpy if its not
    if type(img) is not np.ndarray:
        img = np.asarray(img)
    
    #some operation will directly impact the input image, so we must keep a copy of it
    imCopy = img.copy()
    
    ### convert to grayscale ###
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ### find contour ###
    #for better accuracy we use binary images before finding contours, applying threshold: 
    #if pixel is above 200 (first value, reducing to 160 may lead to to much images) we assign 255 (second value), 
    #below we assign 0 (third value).
    ret,thresh = cv2.threshold(imgray,200,255,0)
    image, contours, hierarchy =  cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #create a list of rectangle which may correspond to an image
    li_bbox = []
    for contour in contours:
        poly = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, False), False)
        x, y, w, h = cv2.boundingRect(poly)
        #remove if its really small compared to initial (i.e. smaller than 10%) image or equal to the initial image (or half the page
        #in case the book was scanned two pages at a time (horizontally or vertically))
        hi, wi, ci = imCopy.shape
        #avoid: not to smalle image (bad quality or can even be logo etc)
        #avoid: equal to the hole page
        #avoid: equal to half page when scanned with two page on the width
        #avoid: equal to half page when scanned with two page on the height
        if (h>(hi*0.1)) & (w>(wi*0.1)) & \
        ((h<(hi*0.95))|(w<(wi*0.95))) & \
        ((h<(hi*0.95))|(w>(wi*0.55))|((wi*0.45)>w)) & \
        ((w<(wi*0.95))|(h>(hi*0.55))|((hi*0.45)>h)):
            li_bbox.append((x,y,w,h))
    
    #remove images included in another image
    li_bbox = remove_embedded_bbox(li_bbox)
    
    if plot_==1:
        for bbox in li_bbox:
            x,y,w,h = bbox
            # Create figure and axes
            fig,ax = plt.subplots(1)
            # Display the image
            ax.imshow(imCopy)
            # Create a Rectangle patch
            rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            plt.show()
    
    #create directory if not existing
    if not os.path.exists(path):
        os.makedirs(path)
    
    #save
    for image_id,bbox in enumerate(li_bbox):
        x,y,w,h = bbox
        img_to_save = Image.fromarray(imCopy[y:y+h,x:x+w])
        img_to_save.save(os.path.join(path,'p'+str(page_id)+'_i'+str(image_id)+'.png'))  
        del img_to_save
      
    #TODO: verify if useful:
    del img
    del imCopy
    del imgray
        
        
def from_path_scannpdf_book_2image(path, path_save, nbrp=2, plot_=0):
    pages = convert_from_path(path)
    print('There is %d pages in the book'%len(pages))
    for i,page in enumerate(pages):
        frompng2images(img=page, path=path_save, page_id=i, plot_=plot_)    
    del pages
    
###################################################################################################
################################### preprocessing fct for image ###################################
###################################################################################################

#remove from alist of rectangles (tuples: (x,y,w,h)), the rectangle embedded in another one
#note that the (0,0) point in an image is up left.
def remove_embedded_bbox(li_bbox, plot_bbox=0):
    
    #sort (smaller to bigger) list of rectangles by the highest height (to make things more efficient)
    li_bbox = sorted(li_bbox,key=itemgetter(3))
    
    #initialize list of rectangle to return
    li_bbox_r = li_bbox.copy()
    
    #remove all rectangle when its included in another one. Hence we will compare each rectangle with the one having
    #a higher height only (for efficiency). as soon as we see that the rectangle is included in another one we will remove it and pass 
    #to the next one
    for i,bbox in enumerate(li_bbox):
        for bbox2 in li_bbox[i+1:]:
            x1, y1, w1, h1 =  bbox
            x2, y2, w2, h2 =  bbox2
            if (w1<w2) & (x1>x2) & (y1>y2) & (x1+w1<x2+w2) & (y1+h1<y2+h2):
                li_bbox_r.remove(bbox)

                #plot (to debug)
                if plot_bbox==1:
                    #print(x1, y1, w1, h1)
                    #print('is included in :')
                    #print(x2, y2, w2, h2)
                    # Create figure and axes
                    fig,ax = plt.subplots(1)
                    # Display the image
                    ax.imshow(np.zeros(shape=(max(y1+h1,y2+h2)+50,max(x1+w1,x2+w2)+50)))
                    # Create a Rectangle patch
                    rect = patches.Rectangle((x1,y1),w1,h1,linewidth=1,edgecolor='r',facecolor='none')
                    # Add the patch to the Axes
                    ax.add_patch(rect)

                    rect = patches.Rectangle((x2,y2),w2,h2,linewidth=1,edgecolor='r',facecolor='none')
                    # Add the patch to the Axes
                    ax.add_patch(rect)
                    plt.show()
                break
    return(li_bbox_r)
#small test for embeded images
#li_bbox = [(1281, 79, 933, 1425), (1557, 600, 282, 396)]
#remove_embedded_bbox(li_bbox,plot_bbox=1)


#take an image and return the image withou reflect
def data_augmentation_remove_reflect(img):
    '''r: Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.'''
    #convert it to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #should be more robust gray = cv2.GaussianBlur(gray, (41, 41), 0) #flou une image en utilisant un filtre gaussian
    #keep only the brightess part
    mask_ = cv2.inRange(gray, 0, 150)
    #put zero pixel to non-zero pixels and vise versa so that we can use inpaint
    mask_ = np.array([[1 if j==0 else 0 for j in i] for i in mask_]).astype('uint8')
    #image inpainting is used. The basic idea is simple: Replace those bad marks with its neighbouring pixels 
    #non-zero pixels corresponds to the area which is to be inpainted.
    #Radius of a circular neighborhood of each point inpainted that is considered by the algorithm. : 3
    result = cv2.inpaint(img,mask_,5,cv2.INPAINT_TELEA)
    return(result)

#from an image and mask, keep only the part of the image that intersect with the mask
def KeepMaskOnly(image, mask, debug_text):
    try:
        #instead of one channel produce 3 exact same channel
        mask = np.stack((mask.reshape([mask.shape[0], mask.shape[1]]),)*3, -1)
        #outside the mask put to black color
        image[~mask]=0
    except:
        print(mask.shape)
        print(debug_text)
    return(image)


#replace black pixel by smoothing with adequate color to keep all info, removing shape 
def remove_shape_keep_all_info(img):
    #create mask (s.t. Non-zero pixels indicate the area that needs to be inpainted)
    mask_ = np.array([[1 if j==0 else 0 for j in i] for i in cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_ = cv2.dilate(mask_, kernel, iterations=4)
    result = cv2.inpaint(img,mask_,5,cv2.INPAINT_TELEA) #,cv2.INPAINT_TELEA, INPAINT_NS
    return(result)


#resize the image regarding either width or height, keeping ratio
def image_resize_keeping_prop(image, width = None, height = None, inter = cv2.INTER_AREA):

    #initialize
    dim = None
    (h, w) = image.shape[:2]

    #if both the width and height are None, then return the original image
    if width is None and height is None:
        return image
    
    #if both is not nonw return error
    if (width is not None) & (height is not None):
        print('ERROR: you should give either the width or the height, not both')
        sys.exit()

    #if width is None, then height is not None
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    #otherwise the height is None and width is not None
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    #resize the image in convenient manner and return it
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized
'''or use 
import imutils
img = imutils.resize(image, width=500) '''


#function that resize image keeping the aspect raito and adding less possible black pixel. In other words:
#function that takes as input an image, change the magnitude of the image to fit better the new dimension (n1, n2) and
# finaly resize it to fit exactly the dimension keeping the intiial aspect ration and hence adding black pixel where 
#its needed.
def adjust_size(image, h, w):
    
    #initialize
    dim = None
    (hi, wi) = image.shape[:2]
    
    #change image resolution
    #as we dont want to remove some pixel, we will resize the image keeping the initial aspect ratio, making sur that
    #both width and height stay smaller than the one wanted (w,h)
    #if the initial image is more 'flat-rectangle' than the target dimension, then resize w.r.t.  the width
    if (hi/wi)<(h/w):
        image = image_resize_keeping_prop(image, width=w)
    #if the nitial image is more 'height-rectangle' than the target dimension, then resize w.r.t.  the height
    else:
        image = image_resize_keeping_prop(image, height=h)
    
    #change dimension
    (hi, wi) = image.shape[:2]
    #finally resize to fit the exact target dimension by adding black pixel where its needed
    l = int((w-wi)/2)
    r = w-(wi+l)
    t = int((h-hi)/2)
    b = h-(hi+t)
    image = cv2.copyMakeBorder(image,t,b,l,r,cv2.BORDER_CONSTANT) #top, bottom, left, right #,value=[0,0,0]
    
    return(image)

#to use the preprocessing step used in inceptionv3 training for other purpose as well (in testing for example)
#augmenta should be used in training essentially
def image_augmentation_with_maskrcnn(ID, n1, n2, image_path, mask_path, augmentation=None, normalize=False, preprocessing=None,
                                    plot_3_image=False):
    
    #downlaod image and mask
    P = os.path.join(mask_path,'mask_output_'+ID+'.pkl') #in the next version dont save with mask_output in front
    mask = pickle.load(open(P, 'rb'))
    mask = mask['unique_binary_mask']
    image = cv2.imread(os.path.join(image_path, ID+'.jpg'))
    b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])
    if plot_3_image==True:
        image1 = image.copy()
    
    #keep only mask info
    image = KeepMaskOnly(image,mask,str(ID))
    
    #remove black area at maximum by zoomimg keeping aspect ratio
    boxes = pickle.load(open(P, 'rb'))['rois']
    y1, x1, y2, x2 = boxes[0]
    image = image[y1:y2, x1:x2]
    image = adjust_size(image, n1, n2)
    if plot_3_image==True:
        image2 = image.copy()

    
    #replace black pixel by smoothing with adequate color to keep all info, removing shape 
    image = remove_shape_keep_all_info(image)
    if plot_3_image==True:
        image3 = image.copy()

    #normalize
    if normalize == True:
        normalizedImg = np.zeros((200, 200))
        normalizedImg = cv2.normalize(image, normalizedImg, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX) 
        image = normalizedImg
        
    #preprocessing
    if preprocessing is not None:
        image = preprocessing(image)
        
    #augment image
    if augmentation is not None:
        image = augmentation.augment_image(image)
        
    #for debuging   
    if plot_3_image==True:
        return(image1, image2, image3)
    
    return(image)


###################################################################################################
########################################## datagenerator ##########################################
###################################################################################################

#in case you have images without need to do other specific preprocessing (i.e. not put black partout) you can simply use:
'''test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary') ... '''

#copy from internet then small modifications
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, image_path, mask_path, batch_size, n_rows, n_cols, n_channels, n_classes, 
                 augmentation=None, preprocessing=None, shuffle=True, normalize=False, age=None):
        self.n1 = n_rows
        self.n2 = n_cols
        self.batch_size = batch_size
        self.image_path = image_path
        self.mask_path = mask_path
        self.labels = labels
        self.age = age
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.normalize = normalize
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.on_epoch_end()

    def __len__(self):
        'number of step per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        return(self.__data_generation(list_IDs_temp))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels) where n_sampled=batch_size
        # Initialization
        X = np.empty((self.batch_size, self.n1, self.n2, self.n_channels))
        a = np.empty((self.batch_size), dtype=int)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):     
            #handle image
            image = image_augmentation_with_maskrcnn(ID=ID, n1=self.n1,n2=self.n2, 
                                                     image_path=self.image_path, mask_path=self.mask_path,
                                                     augmentation=self.augmentation,
                                                     normalize=self.normalize,
                                                     preprocessing=self.preprocessing)
            X[i,] = image
            #handle class
            y[i] = self.labels[ID]
            #handle age
            if self.age is not None:
                #handle age
                a[i] = self.age[ID]

        #handle age
        if self.age is not None:
            return [X,a], keras.utils.to_categorical(y, num_classes=self.n_classes)
        else:
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    
###################################################################################################
########################################### other - old ###########################################
###################################################################################################
    
#join several dico together without duplicate info but with all possible info
def join_dico(li_s):
    l = len(li_s)
    if l==1:
        return(li_s[0])
    else:
        s1 = li_s[0]
        s2 = li_s[1]
        s = s1.copy()
        for k,v in s2.items():
            if k in s:
                s[k] = s[k] + ' /-/ ' + s2[k]
                s[k] = ' /-/ '.join(list(set([i.strip() for i in s[k].split('/-/')])))
            else:
                s[k] = s2[k]
        r = li_s[2:]
        if len(r)>0:
            return(join_dico(li_s = r+[s]))
        else:
            return(s)
#small example 
#s1 = {'parent':'afdsf','a':'12'}
#s2 = {'a':'sdfsdf /-/ df /-/12','cac':'q'}
#s3 = {'a':'sdfsdf1213'}
#s4 = {'hello':'new'}
#join_dico([s1,s2,s3,s4]) 


#take two strings as input. rule pour enlever les pluriels/singulier: guarder celui qui permettra de retrouver l'autre (car pas forcément le cas
#dans les deux sens) ou alors garder celui qui est ordonner alphabetiquement le premier (si les deux peuvent induire l'autre)
#example d'utilisation: si initialement dans notre liste d'ingredient il y a une forme qui ne permet pas de retourner 
#à son autre form, alors il faudra l'updater avec l'autre
engine = inflect.engine()
def keep_goodone_singplu(x1,x2):
    x1_s = engine.plural(x1)
    x2_s = engine.plural(x2)
    if (x1_s==x2) and (x2_s==x1):
        return([sorted([x1,x2])[0]])
    elif (x1_s==x2):
        return([x1])
    elif (x2_s==x1):
        return([x2])
    else:
        return([x1,x2])
#e.g.
#keep_goodone_singplu('pie cakes','pie cake')
#the plural are: pie cakess, pie cakes -->'pie cake'


#output a itertools of tuples of all possible combinations of 2 elements form the list 'li'
def all_subsets(li):
    return chain(*map(lambda x: combinations(li, x), range(2, 3)))
#for subset in all_subsets([1,3,5]):
#    print(subset)
#-->:
#(1, 3)
#(1, 5)
#(3, 5)


#It simply says that a tree is a dict whose default values are trees.
def tree(): return defaultdict(tree)
#from tree to dict
def dicts(t): return {k: dicts(t[k]) for k in t}
#iteration
def add(t, path):
    for node in path:
        t = t[node]
        
        
#put the values of all keys except the one in li_ke together (they should be list) (used in below fct)
def all_except_keys(dico,li_ke):
    r = []
    dico_ = dico.copy()
    for k in li_ke:
        dico_.pop(k,None)
    r = list(dico_.values())
    r = [i for sublist in r for i in sublist]
    return(set(r))        


#from a string x withoutwhitespace, and a list of whitespace index, it return the string wiht the adequate whitespace
def from_string_without_whitespace_to_string_withwithespace(x, li_index):
    initial_length = len(x)+len(li_index)
    for i in range(initial_length):
        if i in li_index:
            x = x[0:i] + ' ' + x[i:]
    return(x)
#small example
#x = 'ab asd whfjzf gdzf  fuj'
#x_ = ''.join(x.split(' '))
#li_index = [m.start() for m in re.finditer(' ', x)]
#from_string_without_whitespace_to_string_withwithespace(x_, li_index)


#given a list of text without any whitespace and a list of whitespace index corresponding to its original whitespace
#places, it will outputa list with whitespace at the coret places (not at end or begning of entries if their was any)
def from_string_without_whitespace_to_string_withwithespace(li_x, li_index):
    
    #removing space at end and begining
    li_x = [x.strip() for x in li_x]
    
    #initialisation
    x = ''.join(li_x)
    li_x_r = []
    initial_length = len(x)+len(li_index)
    li_split_index = [len(x) for x in li_x]
    li_split_index = [sum(li_split_index[0:i])-1 for i in range(1,len(li_split_index))]
    last_split_index = -1
    
    #pass through each index
    for i in tqdm.tqdm(range(initial_length)):
        #if it should have a whitespace at this index
        if i in li_index:
            x = x[0:i] + ' ' + x[i:]
            li_split_index = [i+1 for i in li_split_index]
            #print('whitespace',i,li_split_index)
            
        #if it should be splitted at this place
        if i in li_split_index:
            #print('splitted',i,li_split_index)
            li_x_r.append(x[last_split_index+1:i+1])
            last_split_index = i
            
    #add last part and return
    li_x_r.append(x[last_split_index+1:])
    return([i.strip() for i in li_x_r if i!=' '])
#small example: from a text and a list of title, without taking into account whitespace, we want to split it, keeping
#at the end the whitespace too
#text = 'hello snake1 and goodbye snake  2b jkjk labla snake3 '
#li_title = ['snake1', 'snake 2', 'snake3']
#text_nws = ''.join(text.split(' '))
#li_title_nws = [''.join(x.split(' ')) for x in li_title]
#print(li_title_nws)
#pattern = ''
#for p in li_title_nws:
#    pattern = pattern+'|'+p
#pattern = pattern.strip('|')
#print(pattern)
#pattern = re.compile(r'(%s)'%pattern)
#li_text_nws = pattern.split(text_nws)
#print(li_text_nws)
#li_ws_index = [m.start() for m in re.finditer(' ', text)]
#print(li_ws_index)
#from_string_without_whitespace_to_string_withwithespace(li_text_nws, li_ws_index)


#from doxc file extract all the bold text , outputing one string. the idea is to extract letter by letter and when one is not bold, we will not add to the ouput (except when its a whitespace and before was a bold letter
def extract_bold_text(document):
    li_bolds = []
    for para in document.paragraphs:
        last_was_bold = 0
        li_bolds.append(' ')
        for run in para.runs:
            if run.bold:
                li_bolds.append(run.text)
                last_was_bold = 1
            elif (last_was_bold==1) & (run.text==' '):
                li_bolds.append(run.text)
                last_was_bold = 0
    return(''.join(li_bolds))


#from a text (chapter text) and with a list of bold-title to find, we will output the text splitted with the titles
#or closest matched titles
def from_chapter_to_structured_data(text, li_title):
    
    #remove all whitespace as these are not equally ditributed in the bold or in the text outptu
    text_nws = ''.join(text.split(' '))
    li_title_nws = [''.join(x.split(' ')) for x in li_title]
    
    #get index of whitespace in original text
    li_ws_index = [m.start() for m in re.finditer(' ', text)]
    
    #create a list of titles which all match 
    li_matched_title = []
    title_not_matched = []
    li_distance = []
    for i,title in enumerate(li_title):
        r = find_near_matches(title, text_nws, max_deletions=max(int(0.10*len(title)),1), 
                              max_insertions=max(int(0.05*len(title)),1), max_substitutions=0)
        if len(r)==1:
            li_matched_title.append(text[r[0][0]:r[0][1]])
            li_distance.append(r[0][2])
        #keep track of non-matched title (to add rules perhaps or allow more flexibility: TODO)
        elif len(r)==0:
            print(title)
            title_not_matched.append(title)
        else:
            print(r)

    #create a list from text by splitting it with the titles
    pattern = ''
    for p in li_matched_title:
        pattern = pattern+'|'+p.replace('(','\(').replace(')','\)').replace('|','\|') #caractere not supp in regex without backslash
    pattern = pattern.strip('|')
    pattern = re.compile(r'(%s)'%pattern)
    li_text_nws = pattern.split(text_nws)
    
    #compute and return the splited list with adequate whitespace
    r = from_string_without_whitespace_to_string_withwithespace(li_text_nws, li_ws_index)
    return(r, title_not_matched, li_matched_title, li_distance)



###################################################################################################
####################################### manage data  for ML #######################################
###################################################################################################

def split_test_train_within_cat(df,p_test, category_to_split_within, id_to_split_with):
    
    #create lists (one test one train) of id within each category 
    li_test = []
    li_train = []
    for i,j in df.groupby([category_to_split_within]):
        li = list(j[id_to_split_with].unique())
        #shuffle list
        random.shuffle(li)
        n1 = int(len(li)*p_test)
        li_test.extend(li[0:n1])
        li_train.extend(li[n1:])
        
    #create associated dataframes    
    df_test = df[df[id_to_split_with].isin(li_test)]
    df_train = df[df[id_to_split_with].isin(li_train)]
    return(df_test, df_train)

###################################################################################################
############################################### plot ##############################################
###################################################################################################

# will be used in the next fct. Its is used to create a list of length x for the explode parameter in the donut plot
def list_same_number_with_threshold(x, v, nbr_set, nbr_without_explode):
    if x<nbr_without_explode:
        return(np.repeat(0,x))
    else:
        li = list(np.repeat(0,nbr_without_explode))
        r = x-nbr_without_explode
        n = int(r/nbr_set)
    for i in range(nbr_set):
        li.extend(list(np.repeat(v*(i+1), n)))
    if len(li)<x:
        li.extend(list(np.repeat(v*nbr_set,x-len(li))))
    return(li)    


#create a donut plot based on two lists, one for the names, and one for the associated quantity
def donut_plot(li_labels, li_sizes, path, min_val=None, v=0.3, nbr_without_explode=50, fontsize_=6, circle_hole=0.75, nbr_set=5):
    
    #sort list of tuples according to second value
    t = [(li_labels[i], li_sizes[i]) for i in range(len(li_labels))]
    t.sort(key=lambda x: x[1])
    t.reverse()
    if min_val==None:
        li_labels = [i[0] for i in t]
        li_sizes = [i[1] for i in t]
    else:
        li_labels = [i[0] for i in t if i[1]>=min_val]
        li_sizes = [i[1] for i in t if i[1]>=min_val]

    #plot
    fig1, ax1 = plt.subplots()
    li_ = list_same_number_with_threshold(len(li_labels),v=v, nbr_set=nbr_set, nbr_without_explode=nbr_without_explode)
    ax1.pie(li_sizes, labels=li_labels, startangle=90, explode=li_, rotatelabels=True, textprops={'fontsize': fontsize_})
    
    #circle
    centre_circle = plt.Circle((0,0),circle_hole,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax1.axis('equal')  #ensures that pie is drawn as a circle
    plt.tight_layout()
    
    #save and show
    plt.savefig(path, dpi=300, format='png', bbox_inches='tight')
    plt.show()        
    
    
    
    
    