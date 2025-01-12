"""
Script to find the I-frames from a video and extract them using ffprobe and OpenCV
IMPORTANT!! Before running the program, make sure to install ffmpeg and add it to the path

@author: Thant Zin Htoo PAING
"""

import os
import cv2
import subprocess
__path__ = os.path.dirname(os.path.abspath(__file__)) # define the path to the current script


directory = r"path/to/videos" # path to the videos. Do not forget to change the path to your own path

filelist = [] # list to store the video files
# loop to store the video files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.mp4'):
        filelist.append(os.path.join(directory, filename))

# Create a new directory/ folder to save the I-frames
new_dir = os.path.join(directory, 'I-frames')
if not os.path.exists(new_dir):
    os.makedirs(new_dir)



def get_frame_types(video_fn):
    """
    function to get the frame types(here we get I frames) of the video
    """
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)
    
def save_i_keyframes(video_fn):
    frame_types = get_frame_types(video_fn)
    i_frames = [x[0] for x in frame_types if x[1]=='I']
    if i_frames:
        basename = os.path.splitext(os.path.basename(video_fn))[0]

        # #create a new directory with the base name
        # output_dir = os.path.join(__path__,basename)
        # os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_fn)
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            outname = os.path.join(new_dir, f'{basename}_i_frame{frame_no}.jpg')
            cv2.imwrite(outname, frame)
            print ('Saved: '+outname)
        cap.release()
    else:
        print ('No I-frames in '+video_fn)
   
if __name__ == '__main__':
    for filename in filelist:
        save_i_keyframes(filename)