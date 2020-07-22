'''
Questo script ci permette di scaricare foto da Flickr in base a degli specifici tag.
'''

import cv2
import flickrapi
import ftfy
import json
import pickle  
import csv
import requests
import sys, os, os.path
from skimage.measure import compare_ssim as ssim
import numpy as np
from sys import argv
from shutil import copyfile
import time, datetime
from urllib.request import urlretrieve

# questa funzione verifica l'esistenza di una foto scaricata 
def isMissing(imagePath):
    for referenceImage in nullImages:
        refImgPath = "data/" + referenceImage
        refImg = cv2.imread(refImgPath)
        curImg = cv2.imread(imagePath)
        
            
        refImg = cv2.cvtColor(refImg, cv2.COLOR_BGR2GRAY)
        curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)
        
        curImg = cv2.resize(curImg,(refImg.shape[1], refImg.shape[0]))
        
        
        s = ssim(refImg,curImg)
        print ("SSIM with null picture:\t"+str(s))
        if s>0.99:
            return True
    return False
    

def sanitize_text(s):
    s = str(s)
    s = s.replace("&#39;"," ")
    s = s.replace("'"," ")
    s = s.replace("’"," ")
    s = s.replace("\""," ")
    s = s.replace("\n"," ")
    s = ftfy.fix_text(s)
    return s
        
def photo_crawling(photo_id,):
    
    try:
   
        check = False
        print ("\nProcessing photo with FlickrId:\t" +photo_id)

        print ("Getting photo info...")
        response = flickr.photos.getInfo(api_key = api_key, photo_id=photo_id)            
        print ("request result:\t"+response['stat'])
        photo_info = response['photo']
        ext = 'jpg'
        photo_url = 'https://farm'+str(photo_info['farm'])+'.staticflickr.com/'+str(photo_info['server'])+'/'+photo_id+'_'+str(photo_info['secret'])+'_b.'+ext
       
  
        
        # Download the photo and check if still available
        img_path = "images/"+photo_id+".jpg" 
        httpRes = urlretrieve(photo_url, img_path)
        abs_path = os.path.abspath(img_path)
            
        check = isMissing(img_path)
        print ("Is missing:\t" + str(check))
        if check:
            copyfile(abs_path, "missing/"+photo_id)
            os.remove(abs_path)
            abs_path = os.path.abspath("missing/"+photo_id)            
        
        return True   
 
    except Exception as e:
        print ("ERROR with Photo\t"+photo_id+":")
        print (str(e))
        return False
        

def photos_analysis(images_list):

    err_list = []

    print ("\n\nSTART\n************\tUTC\t"+str(datetime.datetime.utcnow())+"\t************")
    print ("Analysing\t"+str(len(images_list))+"\t photos")

    for photo_id in images_list:
       try:
           res = photo_crawling(photo_id) 
           if res:
            print ("Image\t"+photo_id+"\tanalyzed.")
           else:
            print ("Image\t"+photo_id+"\terror occurred.")
            err_list.append(photo_id)
       except Exception as e:
        print ("ERROR (external loop):")
        print ("FlickrId:\t"+str(photo_id))
        err_list.append(photo_id)
        print (str(e))
        
    print ("Error images list:")
    print (err_list)
    print ("Images with errors:")
    print (len(err_list))

    print ("\n\nEND\n************\tUTC\t"+str(datetime.datetime.utcnow())+"\t************")


#-------------------------------------------------------------------------------------------------------

# Null image name
nullImages= ["flickrMissing", "flickrNotFound"]

# Credenziali Flickr
api_key = u'6566c2c10b2ca094b9636585d96aeb79'
api_secret = u'd237a27b1cf58bc2'

# Le seguenti due righe sono state eseguite una volta sola per generare il token di autorizzazione    
flickr = flickrapi.FlickrAPI(api_key, api_secret)
flickr.authenticate_via_browser(perms='read')

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')


#-----------------------------------------------------------------------DOWNLOAD IMMAGINI
tag="sculpture, marble, white" #lo script è stato eseguito più volte cambiando questo tag per scaricare anche immagini
                #di edifici e sculture.
                
response = flickr.photos.search(api_key = api_key,tags=tag, tag_mode='all' )

#Creo una lista di id delle foto che verranno scaricate
photo_list=[]
for record in response['photos']['photo']:
    s=record["id"]
    photo_list.append(s)
#--------------------

photos_analysis(photo_list)
