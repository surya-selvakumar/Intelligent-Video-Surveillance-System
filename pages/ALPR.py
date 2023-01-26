import streamlit as st 
import os


def main():
    st.header('Number Plate Recognition System')
    target = st.text_input('Enter LP Number')
    up_vdo = st.file_uploader('Upload a Video', type=['mp4'])
    
    if up_vdo is not None:
        with open('./Videos/footage.mp4', 'wb') as f:
            f.write(up_vdo.getvalue())
            
    
        choice = st.radio('Search -> Results', options=['Search', 'Show Results'], label_visibility='hidden', horizontal=True)
        
        if choice=='Search':
            os.system("py detect_plates.py {}".format(target))
            st.warning('Search Completed: Results saved at ALPR_Record.csv')
            
        if choice=='Show Results':
            os.system("ffmpeg -i cv2_out.mp4 -vcodec libx264 alpr_out.mp4 -y")
            st.video(open('alpr_out.mp4', 'rb').read())
        
    
    
main()
    
    

    