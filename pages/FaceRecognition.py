import streamlit as st 
import json
import numpy as np
import tempfile
from PIL import Image
from FR.new_rec import *

def face_recognition():
    st.header('Intelligent Face Recognition System')
    vdo = st.file_uploader('Upload a Video', type=['mp4'])
    img = st.file_uploader('Upload the target image', type=['jpeg', 'jpg', 'png'])
    if st.button('Find'):
        if img is not None:
            with open('./FR/Images/target.jpg', 'wb') as f1:
                f1.write(img.getvalue())
        
        if vdo is not None:
            with open('./FR/Videos/footage.mp4', 'wb') as f2:
                f2.write(vdo.getvalue())
                
                
        enc_list = encode_recognize()
        ret = cam_read(enc_list)
        if ret:
            os.system("ffmpeg -i cv2_Output.mp4 -vcodec libx264 FR_Output.mp4 -y")
            vid_path = open('FR_Output.mp4', 'rb')
            vid_bytes = vid_path.read()
            st.video(vid_bytes)
                
                
face_recognition()
                
                
                    
        
        
            
            
  
        
        
    
    
    