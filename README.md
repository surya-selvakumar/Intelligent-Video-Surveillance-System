
# Intelligent-Video-Surveillance-System

Automated License Plate Recognition System
The system enables the user to search for a given license plate number from the footage and identify the timestamp at which the vehicle(car) was appeared. 
The timestamps are stored in the ALPR_Record.csv file 

Facial Recognition System
The FR System utilizes the facial encodings to identify the matches from the given video
The timestamp at which the face is identified is stored in the FR_Record.csv file

The annotated video undergoes ffmpeg code conversion inorder to be displayable in the streamlit UI


## Run Locally

Clone the project

```bash
  git clone https://github.com/surya-selvakumar/Intelligent-Video-Surveillance-System.git
```

Go to the project directory

```bash
  cd Intelligent-Video-Surveillance-System
```



## Steps to run

Create a VirtualEnvironment
```bash
  py -m venv venv
  source venv/scripts/activate
```
To install the requirements
```bash
  pip install requirements.txt
```

To run the WebApp
```bash
  streamlit run Main.py
```

ffmpeg command to convert the video form
```bash
  ffmpeg -i cv2_out.mp4 -vcodec libx264 alpr_out.mp4 -y
  ffmpeg -i cv2_Output.mp4 -vcodec libx264 FR_Output.mp4 -y
```


## License

[MIT](https://choosealicense.com/licenses/mit/)

