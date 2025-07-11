import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import random
#import pyautogui
from math import hypot
import matplotlib.pyplot as plt
import webbrowser
import os
import math
#import requests 
#from streamlit_lottie import st_lottie 

#mp_drawing = mp.solutions.drawing_utils
#mp_face_mesh = mp.solutions.face_mesh
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


st.title('Body Gesture Controlled Gaming Application!')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Body Gesture Controlled Gaming Application')
st.sidebar.subheader('Dashboard')

@st.cache_resource()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

game_mode = st.sidebar.selectbox('Choose the Game',
['About App','Subway Surfers','Bubble Game','Dino Game','Snake Game','Fruit Ninja','Pong Game','Temple Run']
)

if game_mode =='About App':
    #st.markdown('In this application we are using **MediaPipe** for creating a Face Mesh. **StreamLit** is to create the Web Graphical User Interface (GUI) ')
    st.subheader('Experiance the Next-Gen Gaming')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    #st.video('https://www.youtube.com/watch?v=FMaNNXgB_5c&ab_channel=AugmentedStartups')
    
    #svg_file = open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/collage.svg', 'rb')
    #st.image(svg_file)

    with open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/collage.svg', 'r') as svg_file:
        svg_content = svg_file.read()

    #Display the SVG using st.markdown()
    st.markdown(svg_content, unsafe_allow_html=True)

    #st.video('https://www.youtube.com/watch?v=dQw4w9WgXcQ')
    st.video('https://youtu.be/fnvCe9N6yfA')
    st.markdown('''
          # Subheading \n 
            This Application enables to play games using body gestures.\n
             blah blah blah \n
           
            ok ok **ok** \n
            
            Also check us out various games
            - [Subway Surfers]
            - [Snake Game]
            - [Facebook]
            - [Discord]
        
            
             
            ''')
elif game_mode =='Bubble Game':
    mp_drawing = mp.solutions.drawing_utils

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Enable Webcam and Play Now')
    #record = st.sidebar.checkbox("Record Video")
    #if record:
    #   st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )
    # max faces

    st.sidebar.write("Download the game and  run on your PC")
    with open("C:/Users/Admin/Desktop/Gesture_Gaming/Zips/Subway_Surfers.zip", "rb") as file:
        btn = st.sidebar.download_button(
            label="Download Here",
            data=file,
            file_name="Subway_Surfers.zip",
            mime="application/zip"
          )
          
    #max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Bubble Game')

    stframe = st.empty()

    #video_file_buffer = False#st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    if not use_webcam:   
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/circle.png')
        #image = Image.open('ss.jpg')      
        st.image(image, caption='Gameplay of the game',width=400)
        st.subheader(':blue[Game Description]')
        st.text('''User has to hold the hand as shown in the above image and play\n
The blue bubbles are randomly generated in the screen.\n
The objective is to touch as many as blue bubbles    as possible. \n
The Game can also be played by downloading directly on your devices.
        ''')
        

    if use_webcam:
        vid = cv2.VideoCapture(0)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        #<--!codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)-->
        #<--!codec = cv2.VideoWriter_fourcc('V','P','0','9')-->
        #<--!out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

        #st.sidebar.text('Input Video')
        #st.sidebar.video(tfflie.name)
        fps = 0
        i = 0
        drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)


        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Detected Faces**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        score=0
        face_count=1
        
        x_enemy=random.randint(50,600)
        y_enemy=random.randint(50,400)

        def enemy():
            global score,x_enemy,y_enemy
            #x_enemy=random.randint(50,600)
            #y_enemy=random.randint(50,400)
            cv2.circle(image, (x_enemy,y_enemy), 25, (0, 200, 0), 5)
            #score=score+1

        # Use this line to capture video from the webcam
        cap = cv2.VideoCapture(0)


        # Set the title for the Streamlit app
        #st.title("Video Capture with OpenCV")

        frame_placeholder = st.empty()

        # Add a "Stop" button and store its state in a variable
        stop_button_pressed = st.button("Stop")

        with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
            prevTime = 0
            while cap.isOpened() and not stop_button_pressed:
                ret, frame = cap.read()

                if not ret:
                    st.write("The video capture has ended.")
                    break

                # You can process the frame here if needed
                # e.g., apply filters, transformations, or object detection

                # Convert the frame from BGR to RGB format
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                face_count = len(faces)
                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                image = cv2.flip(image, 1)
                
                imageHeight, imageWidth, _ = image.shape
        
                results = hands.process(image)
        
        
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
                font=cv2.FONT_HERSHEY_SIMPLEX
                color=(255,0,255)
                text=cv2.putText(image,"Score",(480,30),font,1,color,4,cv2.LINE_AA)
                text=cv2.putText(image,str(score),(590,30),font,1,color,4,cv2.LINE_AA)
        
                enemy()
        
                if results.multi_hand_landmarks:
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                                )
        
        
                if results.multi_hand_landmarks != None:
                    for handLandmarks in results.multi_hand_landmarks:
                        for point in mp_hands.HandLandmark:
            
                
                            normalizedLandmark = handLandmarks.landmark[point]
                            pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
                
                            point=str(point)
                            #print(point)
                            if point=='HandLandmark.INDEX_FINGER_TIP':
                                try:
                                    cv2.circle(image, (pixelCoordinatesLandmark[0], pixelCoordinatesLandmark[1]), 25, (0, 200, 0), 5)
                                    #print(pixelCoordinatesLandmark[1])
                                    if pixelCoordinatesLandmark[0]==x_enemy or pixelCoordinatesLandmark[0]==x_enemy+10 or pixelCoordinatesLandmark[0]==x_enemy-10:
                                        #if pixelCoordinatesLandmark[1]==y_enemy or pixelCoordinatesLandmark[1]==y_enemy+10 or pixelCoordinatesLandmark[1]==y_enemy-10:
                                    #if pixelCoordinatesLandmark[1]==y_enemy or pixelCoordinatesLandmark[1]==y_enemy+10 or pixelCoordinatesLandmark[1]==y_enemy-10:
                                        print("found")
                                        x_enemy=random.randint(50,600)
                                        y_enemy=random.randint(50,400)
                                        score=score+1
                                        font=cv2.FONT_HERSHEY_SIMPLEX
                                        color=(255,0,255)
                                        text=cv2.putText(frame,"Score",(100,100),font,1,color,4,cv2.LINE_AA)
                                        enemy()
                                except:
                                    pass
                
                #cv2.imshow('Hand Tracking', image)
                #time.sleep(1)
        
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print(score)
                    break



                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Display the frame using Streamlit's st.image
                #frame_placeholder.image(image, channels="RGB")

                # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
                if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed: 
                    use_webcam=False
                    break


                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                #Dashboard
                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
                kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

                frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
                frame = image_resize(image = frame, width = 640)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                stframe.image(image,channels = 'BGR',use_column_width=True)

    #st.text('Video Processed')

    #output_video = open('output1.mp4','rb')
    #out_bytes = output_video.read()
    #st.video(out_bytes)

    #vid.release()
    #out. release()

elif game_mode =='Subway Surfers':
    from time import time
    import time as tp

    mp_pose = mp.solutions.pose
    pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
    pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils 

    st.set_option('deprecation.showfileUploaderEncoding', False)

    
    use_webcam = st.sidebar.button('Enable Webcam and Play Now')
    #record = st.sidebar.checkbox("Record Video")
    #if record:
    #    st.checkbox("Recording", value=True)
    
    
    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )
    # max faces

    st.sidebar.write("Download the game and  run on your PC")
    with open("C:/Users/Admin/Desktop/Gesture_Gaming/Zips/Subway_Surfers.zip", "rb") as file:
        btn = st.sidebar.download_button(
            label="Download Here",
            data=file,
            file_name="Subway_Surfers.zip",
            mime="application/zip"
          )
    
    #max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')
    
    st.markdown(' ## Subway Surfers')

    stframe = st.empty()
    #video_file_buffer = False#st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    #vid = cv2.VideoCapture(0)
    if not use_webcam:
        col1, col2 = st.columns(2)
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/ss.jpg')     
        col1.image(image)
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/ss_running.jpg')     
        col2.image(image)
        #st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row; justify-content:center;}</style>', unsafe_allow_html=True)
        st.markdown(
            """
            <style>
            .center {
                display: flex;
                justify-content: center;
            }
            </style>
            """
            , unsafe_allow_html=True
        )

        # Add buttons
        #st.markdown('<div class="center">' +
        #            '<button style="margin: 5px;">Button 1</button>' +
        #           '<button style="margin: 5px;">Button 2</button>' +
        #            '<button style="margin: 5px;">Button 3</button>' +
        #            '</div>'
        #            , unsafe_allow_html=True)
        #if st.button('Button 1'):
        #   use_webcam=True 

        use_webcam = st.button("Enable Webcam and Play Now!")


        st.markdown('''
          ### Game Description \n 
            Subway Surfers is a popular endless runner game that combines high-speed action, vibrant visuals, and exciting gameplay.\n
            How about playing the same game with **Body Gestures?** :sunglasses: \n
             \n

             Enable the webcam to begin the fun.  \n
            ''')


        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/handjoineddistance.png')     
        st.image(image, caption='Subway Surfers',width=400)
        st.text('''After the camera turns up Join both the hands to start the game\n
The basic 4 operations are :
    - Left
    - Right
    - Jump
    - Crouch
''')
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/horizontal_movements.png')     
        st.image(image, caption='Illustatraion of Horizontal Movements')
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/vertical_movements.png')     
        st.image(image, caption='Illustatraion of Vertical Movements')
        st.markdown('''### As shown in the above images,\n
    - To Move Left  :  Move to left such that that your **Right Shoulder** lies in the **Left half** the vertical line drawn 
    - To Move Right :  Move to left such that that your **Left Shoulder** lies in the **Right half** the vertical line drawn\n
    - To Jump       :  Jump such that both of your shoulders lies **Above** the horizontal line drawn
    - To Crouch     :  Crouch such that both of your shoulders lies **Below** the horizontal line drawn\n\n
#### Forcing the player to literally jump amd move for gameplay not only enhaces the user experaince and also maintains the physical fitness

        ''')

    if use_webcam:
        cap = cv2.VideoCapture(0)
        cap.set(3,1280)
        cap.set(4,960)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))

        #<--!codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)-->
        #<--!codec = cv2.VideoWriter_fourcc('V','P','0','9')-->
        #<--!out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

        #st.sidebar.text('Input Video')
        #st.sidebar.video(tfflie.name)
        fps = 0
        i = 0
        drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)


        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Detected Faces**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)

        def detectPose(image, pose, draw=False, display=False):
        
            output_image = image.copy()
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(imageRGB)
            if results.pose_landmarks and draw:
                        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),thickness=3, circle_radius=3),connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),thickness=2, circle_radius=2))
            
            if display:
            
                plt.figure(figsize=[22,22])
                plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
                plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
            
            else:
                return output_image, results




        def checkHandsJoined(image, results, draw=False, display=False):
        
            height, width, _ = image.shape
            
        
            output_image = image.copy()
            
            left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)


            right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
            
            euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                        left_wrist_landmark[1] - right_wrist_landmark[1]))
            

            if euclidean_distance < 130:
                
                hand_status = 'Hands Joined'  
                color = (0, 255, 0)
                
            
            else:
            
                hand_status = 'Hands Not Joined' 
                color = (0, 0, 255)
                
            if draw:

                cv2.putText(output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
                cv2.putText(output_image, f'Distance: {euclidean_distance}', (10, 70),
                            cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
                
        
            if display:

                plt.figure(figsize=[10,10])
                plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
            
        
            else:
                return output_image, hand_status



        def checkLeftRight(image, results, draw=False, display=False):
        
        
            horizontal_position = None
            
            
            height, width, _ = image.shape
            
            
            output_image = image.copy()
        
            left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)

        
            right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)
            
        
            if (right_x <= width//2 and left_x <= width//2):
                
                horizontal_position = 'Left'

        
            elif (right_x >= width//2 and left_x >= width//2):
                horizontal_position = 'Right'
            
        
            elif (right_x >= width//2 and left_x <= width//2):
                horizontal_position = 'Center'
                
        
            if draw:

                
                cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2,(0, 255, 0), 3)
                
                
                cv2.line(output_image, (width//2, 0), (width//2, height), (255, 255, 255), 2)
                
        
            if display:
                plt.figure(figsize=[10,10])
                plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
            

            else:
            
                return output_image, horizontal_position


        def checkJumpCrouch(image, results, MID_Y=250, draw=False, display=False):
        
        
        
            height, width, _ = image.shape
            
        
            output_image = image.copy()
            
            
            left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)

            
            right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)

            actual_mid_y = abs(right_y + left_y) // 2
            
            lower_bound = MID_Y-15
            upper_bound = MID_Y+100
            
            
            if (actual_mid_y < lower_bound):
                posture = 'Jumping'
            
        
            elif (actual_mid_y > upper_bound):
                posture = 'Crouching'
            
            else:
                posture = 'Standing'
                
        
            if draw:

                
                cv2.putText(output_image, posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                cv2.putText(output_image,"Join both hands to re-run or to get skateboard",(450,height-40),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                cv2.putText(output_image,"Click on Camera window and press esc to exit",(450,height-10),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                cv2.line(output_image, (0, MID_Y),(width, MID_Y),(255, 255, 255), 2)
                

            if display:

            
                plt.figure(figsize=[10,10])
                plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
            
            else:
            
                return output_image, posture

        # Use this line to capture video from the webcam
        cap = cv2.VideoCapture(0)
        cap.set(3,1280)
        cap.set(4,960)


        # Set the title for the Streamlit app
        #st.title("Video Capture with OpenCV")

        #cv2.namedWindow('Subway Surfers with Pose Detection', cv2.WINDOW_NORMAL)
        #tp.sleep(3)

        #pyautogui.keyDown(key='ctrl')            
        #pyautogui.typewrite('n')
        #pyautogui.keyUp(key='ctrl')  

        webbrowser.open("https://poki.com/en/g/subway-surfers#")
        #tp.sleep(3)

        #pyautogui.keyDown(key='win')            
        #pyautogui.press('left') 
        #pyautogui.keyUp(key='win') 

        #pyautogui.typewrite('\n')
        time1 = 0

        game_started = False   


        x_pos_index = 1


        y_pos_index = 1


        MID_Y = None


        counter = 0

        num_of_frames = 10


        frame_placeholder = st.empty()

        # Add a "Stop" button and store its state in a variable
        stop_button_pressed = st.button("Exit Game")


        while cap.isOpened() and not stop_button_pressed:
            
            ok, frame = cap.read()
            
            if not ok:
                st.write("The video capture has ended.")
                break

            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            face_count = len(faces)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            frame = cv2.flip(frame, 1)
            
        
            frame_height, frame_width, _ = frame.shape
            
            frame, results = detectPose(frame, pose_video, draw=game_started)

        
            
            if results.pose_landmarks:
                
                if game_started:
                    
                    #--------------------------------------------------------------------------------------------------------------
                    
                
                    frame, horizontal_position = checkLeftRight(frame, results, draw=True)
                    
                    if (horizontal_position=='Left' and x_pos_index!=0) or (horizontal_position=='Center' and x_pos_index==2):
                        
                        
                        #pyautogui.press('left')
                        
                    
                        x_pos_index -= 1               

                    
                    elif (horizontal_position=='Right' and x_pos_index!=2) or (horizontal_position=='Center' and x_pos_index==0):
                        
                        
                        #pyautogui.press('right')
                        
                        
                        x_pos_index += 1
                    
                    #--------------------------------------------------------------------------------------------------------------
                
                
                else:
                    
                    cv2.putText(frame, 'JOIN BOTH HANDS TO START THE GAME.', (5, frame_height - 10), cv2.FONT_HERSHEY_PLAIN,2, (0, 255, 0), 3)
                    
                #------------------------------------------------------------------------------------------------------------------
                

                if checkHandsJoined(frame, results)[1] == 'Hands Joined':

                    #pyautogui.click(x=1000, y=500, button='right')
                    
                    counter += 1

                    if counter == num_of_frames:

                    
                        #----------------------------------------------------------------------------------------------------------
                        
                    
                        if not(game_started):
                            game_started = True

                            left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)

                            right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)

                
                            MID_Y = abs(right_y + left_y) // 2
                            #pyautogui.click(x=800, y=500, button='left')
                        
                    
                        else:
                            continue
                        
                            #pyautogui.press('space')
                        
                        #----------------------------------------------------------------------------------------------------------
                        
                        counter = 0

                
                else:

                    counter = 0
                    
                #------------------------------------------------------------------------------------------------------------------

                #------------------------------------------------------------------------------------------------------------------
                
                if MID_Y:
                    
                    frame, posture = checkJumpCrouch(frame, results, MID_Y, draw=True)
                    
                
                    if posture == 'Jumping' and y_pos_index == 1:
                        #pyautogui.press('up')
                        y_pos_index += 1 

                
                    elif posture == 'Crouching' and y_pos_index == 1:
                        #pyautogui.press('down')
                        y_pos_index -= 1
                    
                
                    elif posture == 'Standing' and y_pos_index   != 1:
                        y_pos_index = 1
                
                #------------------------------------------------------------------------------------------------------------------
            
                
            else:
                counter = 0
                
            #----------------------------------------------------------------------------------------------------------------------
            
            
            time2 = time()
        
            if (time2 - time1) > 0:
                frames_per_second = 1.0 / (time2 - time1)
                cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            
            
            time1 = time2
            
            #----------------------------------------------------------------------------------------------------------------------
                    
            #cv2.imshow('Subway Surfers with Pose Detection', frame)

            k = cv2.waitKey(1) & 0xFF    
        
            if(k == 27 or (k==113)):
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   #######################################################################
            # Display the frame using Streamlit's st.image
            #frame_placeholder.image(image, channels="RGB")

            # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
            if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
                use_webcam=False 
                break

                        
    #cap.release()
    #cv2.destroyAllWindows()
            



            
    #--------------------------------------------------------------------------------------------------------------------
            #currTime = time.time()
        # fps = 1 / (currTime - prevTime)
            #prevTime = currTime
            fps=1
            #face_count = 1
            #Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(frames_per_second)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            stframe.image(image,channels = 'BGR',use_column_width=True)

        #st.text('Video Processed')

        #output_video = open('output1.mp4','rb')
        #out_bytes = output_video.read()
        #st.video(out_bytes)

        #cap.release()
        #out. release()

elif game_mode =='Dino Game':
    from cvzone.HandTrackingModule import HandDetector
    import math
    detector = HandDetector(detectionCon=0.8, maxHands=1)
    #cap = cv2.VideoCapture(0)

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    #record = st.sidebar.checkbox("Record Video")
    #if record:
    #    st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )
    # max faces

    st.sidebar.write("Download the game and  run on your PC")
    with open("C:/Users/Admin/Desktop/Gesture_Gaming/Zips/Subway_Surfers.zip", "rb") as file:
        btn = st.sidebar.download_button(
            label="Download Here",
            data=file,
            file_name="Subway_Surfers.zip",
            mime="application/zip"
          )

    #max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Dino Game')

    stframe = st.empty()

    #video_file_buffer = False#st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    if not use_webcam:   
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/dino.jpg')
        #image = Image.open('ss.jpg')      
        st.image(image, caption='Gameplay of Dino Game')
        st.markdown('''
          ### Game Description \n 
            Dino Game is a popular endless browser game developed by Google and built into the Google Chrome web browser.\n
            How about playing the same game with **Hand Gestures?** :sunglasses: \n
             \n

             Enable the webcam to begin the fun.  \n
            The motive of the game is to go ahead as far as possible in the endless.\n
            The only operation performed is to make our little Dino jump when it comes acroos the obstacles.
            ''')
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/dino1.png')     
        st.image(image, caption='Gameplay of Dino Game',width=400)
        st.markdown('''
            Keep the Hand as shown in the image above.\n
            Increaase the distance between ur Index finger and thumb finger to jump and restore back to the original position \n
            ''')

    if use_webcam:
        vid = cv2.VideoCapture(0)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        #<--!codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)-->
        #<--!codec = cv2.VideoWriter_fourcc('V','P','0','9')-->
        #<--!out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

        #st.sidebar.text('Input Video')
        #st.sidebar.video(tfflie.name)
        fps = 0
        i = 0
        #drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)


        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Detected Faces**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)
        
        #pyautogui.keyDown(key='ctrl')            
        #pyautogui.typewrite('n')
        #pyautogui.keyUp(key='ctrl')  

        webbrowser.open("https://offline-dino-game.firebaseapp.com/")
        time.sleep(3)

        #pyautogui.keyDown(key='win')            
        #pyautogui.press('left') 
        #pyautogui.keyUp(key='win') 

        #pyautogui.typewrite('\n')


        # Use this line to capture video from the webcam
        cap = cv2.VideoCapture(0)


        # Set the title for the Streamlit app
        #st.title("Video Capture with OpenCV")

        frame_placeholder = st.empty()

        # Add a "Stop" button and store its state in a variable
        stop_button_pressed = st.button("Exit Game")
        prevTime=0
        while True and not stop_button_pressed:
            success, img = cap.read()
            if not success:
                st.write("The video capture has ended.")
                continue
            if stop_button_pressed:
                use_webcam=False
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            face_count = len(faces)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            hands,img=detector.findHands(img)
            if hands:
                hand1=hands[0]
                lmList1=hand1["lmList"]
                
                x1,y1,_=lmList1[4]
                x2,y2,_=lmList1[8]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                length = math.hypot(x2 - x1, y2 - y1)
                if length<25:
                    cv2.circle(img, (cx, cy), 10, (0,255, 0), cv2.FILLED)
                #print(length)
                vol = np.interp(length, [40, 50], [0, 1])
                print(int(vol))
                #if vol==1:
                    #pyautogui.press("space")
            #cv2.imshow("image",img)
            cv2.waitKey(1)   

            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Display the frame using Streamlit's st.image
            #frame_placeholder.image(image, channels="RGB")

            # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
            if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
                #pyautogui.click(x=800, y=500, button='left')
                #pyautogui.keyDown(key='ctrl')            
                #pyautogui.typewrite('w')
                #pyautogui.typewrite('w')
                #pyautogui.keyUp(key='ctrl')  
                use_webcam=False
                break


            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            #face_count=1
            #Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(image,(0,0),fx = 0.8 , fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            stframe.image(image,channels = 'BGR',use_column_width=True)

        #vid.release()
        #out. release()

elif game_mode =='Snake Game':
    import cvzone
    from cvzone.HandTrackingModule import HandDetector
    import winsound
    import pygame
    from pygame import mixer
    import math

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    #record = st.sidebar.checkbox("Record Video")
    #if record:
    #   st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )
    # max faces

    st.sidebar.write("Download the game and  run on your PC")
    with open("C:/Users/Admin/Desktop/Gesture_Gaming/Zips/Subway_Surfers.zip", "rb") as file:
        btn = st.sidebar.download_button(
            label="Download Here",
            data=file,
            file_name="Subway_Surfers.zip",
            mime="application/zip"
          )

    #max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Snake Game')

    stframe = st.empty()

    #video_file_buffer = False#st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    if not use_webcam:   
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/snake_game.jpg')    
        st.image(image, caption='Gameplay of the game')
        st.markdown('''
          ### Game Description \n 
            Snake game is a classical game.\n
            How about playing the same game with **Hand Gestures?** :sunglasses: \n
             \n

             Enable the webcam to begin the fun.  \n
            The objective of this game is to feed the snake with Donuts\n
            The Snake gets drawn as u move your hand from your index finger.\n
            Make the snake eat as many as donuts as possible.\n
            The Game ends when the The head of 
            ''')


    #vid = cv2.VideoCapture(0)
    if use_webcam:
        pygame.init()
        mixer.music.load("C:/Users/Admin/Desktop/Gesture_Gaming/Audios/bg_sound.wav")
        if game_mode=='Snake Game':
            mixer.music.play(-1)

        cap=cv2.VideoCapture(0)
        cap.set(3,1280)
        cap.set(4,720)

        detector=HandDetector(detectionCon=0.8,maxHands=1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))

        #<--!codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)-->
        #<--!codec = cv2.VideoWriter_fourcc('V','P','0','9')-->
        #<--!out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

        #st.sidebar.text('Input Video')
        #st.sidebar.video(tfflie.name)
        fps = 0
        i = 0
        #drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)


        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Detected Faces**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)

        #mp_drawing = mp.solutions.drawing_utils
        #mp_hands = mp.solutions.hands

        class Snake:
            def __init__(self,pathFood):
                self.points=[]
                self.length=[]
                self.currentLen=0
                self.allowedLen=150
                self.previousHead=0,0

                self.imgFood=cv2.imread(pathFood,cv2.IMREAD_UNCHANGED)
                self.hfood,self.wfood,_=self.imgFood.shape
                self.foodPoint=0,0
                self.score=0
                self.gameOver=False
                self.ranFood()

            def ranFood(self):
                self.foodPoint=random.randint(100,1000),random.randint(100,600)

            def update(self,imgMain,currentHead):
                if self.gameOver:
                    cvzone.putTextRect(imgMain,"Game Over!",(300,320),scale=7,thickness=5,offset=20)
                    cvzone.putTextRect(imgMain,f"Your Score:{self.score}",(300,470),scale=7,thickness=5,offset=20)
                    cvzone.putTextRect(imgMain,"Press r to restart the game", (720,610),scale=2,)
                    cvzone.putTextRect(imgMain,"Press Esc or q to exit", (720,680),scale=2)

                else:
                    px,py=self.previousHead
                    cx,cy=currentHead

                    self.points.append([cx,cy])
                    distance=math.hypot(cx-px,cy-py)
                    self.length.append(distance)
                    self.currentLen+=distance
                    self.previousHead=cx,cy

                    #lenth rduction
                    if self.currentLen>self.allowedLen:
                        for i,length in enumerate(self.length):
                            self.currentLen-=length
                            self.length.pop(i)
                            self.points.pop(i)

                            if self.currentLen<self.allowedLen:
                                break
                    #snake eating food
                    rx,ry=self.foodPoint
                    if rx - self.wfood //2 <cx< rx + self.wfood //2 and ry-self.hfood //2  <cy< ry+self.hfood //2:
                        winsound.PlaySound("C:/Users/Admin/Desktop/Gesture_Gaming/Audios/splatter_2.wav", winsound.SND_ASYNC)
                        self.ranFood()
                        self.allowedLen+=50
                        self.score+=1
                        print(self.score)

                    
                    #drowing snake
                    if self.points:
                        for i,point in enumerate(self.points):
                            if i!=0:
                                cv2.line(imgMain,self.points[i-1],self.points[i],(0,0,255),20)
                        cv2.circle(imgMain,self.points[-1],17,(200,0,200),cv2.FILLED)

                    #draw food
                    #rx,ry=self.foodPoint
                    imgMain=cvzone.overlayPNG(imgMain,self.imgFood,(rx - self.wfood// 2, ry - self.hfood//2)) 
                    
                    #score
                    cvzone.putTextRect(imgMain,f" Score ={self.score}",(50,80),scale=3,thickness=3,offset=10)


                    #Collision
                    pts=np.array(self.points[:-2],np.int32)
                    pts=pts.reshape((-1,1,2))
                    cv2.polylines(imgMain,[pts],False,(0,200,0),3)
                    min_dist=cv2.pointPolygonTest(pts,(cx,cy),True)
                    #print(min_dist)
                    if -1<= min_dist<=1:
                        end_time = time.time() 
                        elapsed_time = end_time - start_time 
                        if elapsed_time>7:
                            print("Game Over")
                            winsound.PlaySound("C:/Users/Admin/Desktop/Gesture_Gaming/Audios/Game_over.wav", winsound.SND_ASYNC)
                            self.gameOver=True
                            self.points=[]
                            self.length=[]
                            self.currentLen=0
                            self.allowedLen=150
                            self.previousHead=0,0
                            self.ranFood()

                return imgMain

        game=Snake("C:/Users/Admin/Desktop/Gesture_Gaming/Images/Donut.png")

        start_time = time.time()

        # Use this line to capture video from the webcam
        #cap = cv2.VideoCapture(0)


        # Set the title for the Streamlit app
        #st.title("Video Capture with OpenCV") ######################################################################################

        frame_placeholder = st.empty()

        # Add a "Stop" button and store its state in a variable
        col1, col2 = st.columns(2)
        restart_button_pressed = col1.button("Restart Game")
        stop_button_pressed = col2.button("Exit Game")
        prevTime=0
        while True and not stop_button_pressed:
            success,img=cap.read()
            img=cv2.flip(img,1)
            
            if not success:
                st.write("The video capture has ended.")
                break

            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            face_count = len(faces)
            #for (x, y, w, h) in faces:
            #    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            hands,img=detector.findHands(img,flipType=False)

            if hands:
                lmList=hands[0]['lmList']
                pointIndex=lmList[8][0:2]
                img=game.update(img, pointIndex)
            
                #if lmList[8][2]>lmList[6][2]:
                    #print("two fng")
                
            #cv2.imshow("SNAKE GAME",img)
            if cv2.waitKey(1)==27 or cv2.waitKey(1)==ord('q'):
                pygame.mixer.music.stop()
                cv2.destroyAllWindows()
                cap.release()
            key=cv2.waitKey(1)

            if key == ord('r') or restart_button_pressed:
                start_time = time.time()
                game.gameOver=False
                restart_button_pressed=False
                game.score=0
            


            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Display the frame using Streamlit's st.image
                #frame_placeholder.image(image, channels="RGB")

                # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
            if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed: 
                use_webcam=False
                break

            #face_count=1  
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            #Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(img,(0,0),fx = 0.8 , fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            stframe.image(image,channels = 'BGR',use_column_width=True)

        #st.text('Video Processed')

        #output_video = open('output1.mp4','rb')
        #out_bytes = output_video.read()
        #st.video(out_bytes)

        #vid.release()
        #out. release()
elif game_mode =='Fruit Ninja':
    import threading
    import pygame
    from fruit import Fruit, fruit_names
    import calculations

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Enable Webcam and Play Now')
    #record = st.sidebar.checkbox("Record Video")
    #if record:
    #    st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )
    # max faces

    st.sidebar.write("Download the game and  run on your PC")
    with open("C:/Users/Admin/Desktop/Gesture_Gaming/Zips/Subway_Surfers.zip", "rb") as file:
        btn = st.sidebar.download_button(
            label="Download Here",
            data=file,
            file_name="Subway_Surfers.zip",
            mime="application/zip"
          )
          
    #max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Fruit Ninja')

    stframe = st.empty()

    #video_file_buffer = False#st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    if not use_webcam:   
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/fruit_ninja.png')
        #image = Image.open('ss.jpg')      
        st.image(image, caption='Gameplay of the game',width=700)
        st.subheader(':blue[Game Description]')
        st.markdown('''
            Fruit Ninja is popular game of slicing fruits.
            How about playing the same game with **Hand Gestures?** :sunglasses: \n
             \n

            Enable the webcam to begin the fun.  \n
            The objective is to slice and dice various fruits thrown into the air.\n
            Players earn points for successfully slicing fruits\n
            Bombs must be avoided as slicing them results in penalties.\n
            It offers colorful graphics, catchy music, and sound effects.\n
            Game ends if bomb is slices.\n
            ''')
        

    if use_webcam:
        FPS = 25
        SHOW_MINICAM = False # bottom right corner of screen
        PLAY_THROUGH_MIC = False # play through VB audio cable virtual mic
        GAME_WIDTH = 1280 # width of pygame window (height will be auto resized to 16:9 ratio)

        # global constant variables
        GAME_HEIGHT = round(GAME_WIDTH * 0.5625) # 16:9 ratio
        WANTED_WIDTH, WANTED_HEIGHT = 1920, 1080 # the size we expect
        CAP_WIDTH, CAP_HEIGHT = 1280, 720 # size of video capture
        MAX_FRUIT_HEIGHT = GAME_HEIGHT / 4 # from top of screen
        ROUND_COOLDOWN = 2 # seconds
        MP_POSE = mp.solutions.pose
        BACKGROUND_IMAGE = "C:/Users/Admin/Desktop/Gesture_Gaming/Images/background.jpg"

        sad_fruit = [
            "C:/Users/Admin/Desktop/Gesture_Gaming/Images/sad_apple.png",
            "C:/Users/Admin/Desktop/Gesture_Gaming/Images/sad_pear.png",
            "C:/Users/Admin/Desktop/Gesture_Gaming/Images/sad_tomato.png",
        ]
        fruit_slice_sounds = [
            "C:/Users/Admin/Desktop/Gesture_Gaming/Audios/splatter_1.wav",
            "C:/Users/Admin/Desktop/Gesture_Gaming/Audios/splatter_2.wav",
            "C:/Users/Admin/Desktop/Gesture_Gaming/Audios/splatter_3.wav",
            "C:/Users/Admin/Desktop/Gesture_Gaming/Audios/splatter_4.wav",
        ]

        # initialize all imported pygame modules
        pygame.mixer.init(
            devicename='CABLE Input (VB-Audio Virtual Cable)') if PLAY_THROUGH_MIC else pygame.mixer.init()
        pygame.font.init()



        vid = cv2.VideoCapture(0)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        #<--!codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)-->
        #<--!codec = cv2.VideoWriter_fourcc('V','P','0','9')-->
        #<--!out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

        #st.sidebar.text('Input Video')
        #st.sidebar.video(tfflie.name)
        fps = 0
        i = 0
        #drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)


        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Detected Faces**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        score=0
        #face_count=1
        
        def ratio(x):
            return round(x * (GAME_WIDTH / WANTED_WIDTH))

        # fonts
        SMALL_NUMBER_FONT = pygame.font.SysFont('Comic Sans MS', ratio(30))
        MAIN_NUMBER_FONT = pygame.font.Font("C:/Users/Admin/Desktop/Gesture_Gaming/fonts/Ninja Font.ttf", ratio(50))
        TITLE_FONT = pygame.font.Font("C:/Users/Admin/Desktop/Gesture_Gaming/fonts/Ninja Font.ttf", ratio(60))

        # colours
        LIGHT_GRAY = (160, 166, 176)
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        NINJA_RED = (227, 25, 25)

        # other
        KNIFE_WIDTH = ratio(10)
        KNIFE_TRAIL_LIFETIME = 0.25

        #create display, set window size, start clock
        #pygame.display.set_caption("Fruit Ninja Motion Tracking")
        screen = pygame.display.set_mode([GAME_WIDTH, GAME_HEIGHT], pygame.SCALED)
        #pygame.display.toggle_fullscreen()
        clock = pygame.time.Clock()

        left_knife_trail = [] # list of tuples that store coords of knife trail, and when they're drawn
        right_knife_trail = [] # right hand trail
        fruits = [] # list of fruits
        background_cv2_image = cv2.imread(BACKGROUND_IMAGE)


        def main():
            round_number = 0
            total_points = 0
            explosion_alpha = 0
            start_game = False # should game start?
            running = True # is game running?
            exploding = False
            main_music = None
            
            last_round = time.time() # time that last round started
            start_fruit = None
            dead_fruit = None
            left_circles = []
            right_circles = []

            # create pose object used to motion track the pose of the user
            with MP_POSE.Pose(
                min_detection_confidence = 0.7,
                min_tracking_confidence=0.5,
                model_complexity=0) as pose:

                # open the webcamera and set the capture's resolution
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
                main_music = play_sound("C:/Users/Admin/Desktop/Gesture_Gaming/Audios/music.wav")

                frame_placeholder = st.empty()

                # Add a "Stop" button and store its state in a variable
                stop_button_pressed = st.button("Stop")

                # while the webcam capture session is open and the game loop is running
                while cap.isOpened() and running:
                    start = time.time()
                    clock.tick(FPS) # tick the clock relative to the FPS of the game
                    success, frame = cap.read()

                    if not success: # stopped camera?
                        continue
                    
                    # process webcam to track pose, and draw it on image_to_display
                    results, image_to_display = calculations.find_and_draw_pose(
                        pose, 
                        frame,
                        background_cv2_image)
                    
                    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    face_count = len(faces)

                    # pass mutable reference of webcam feed, display on corner of image_to_display
                    if SHOW_MINICAM:
                        calculations.add_webcam_feed(image_to_display, frame)
                    
                    # transform image_to_display into a pygame surface
                    cam_img = calculations.array_img_to_pygame(image_to_display, GAME_WIDTH, GAME_HEIGHT)
                    screen.fill(RED)
                    screen.blit(cam_img, (0, 0))

                    # generate fruit if we have none, and after intermission has ended
                    if (len(fruits) == 0 and time.time() - last_round > ROUND_COOLDOWN 
                        and start_game
                        and start_fruit == None
                        and dead_fruit is None):
                        round_number += 1 # new round
                        new_fruits = random.randrange(2, 6) # number of new fruits to generate in one round
                        make_new_fruits(new_fruits, round_number) # give mutable reference to list of fruits that will be displayed

                    # pass mutable reference of knife_trail lists, add trails based on results, and get coords of hands
                    left_hand, right_hand = calculations.knife_trails_and_find_hands(
                        results, 
                        left_knife_trail, 
                        right_knife_trail, 
                        GAME_WIDTH, 
                        GAME_HEIGHT)

                    # delete cut fruit, see if we touched a bomb
                    for fruit in fruits:
                        if results.pose_landmarks and dead_fruit is None:
                            new_points, exploding = process_fruit(
                                fruit,
                                left_hand,
                                right_hand,
                                left_circles,
                                right_circles)
                            total_points += new_points
                            
                            if exploding:
                                break
                    
                    for fruit in fruits:
                        fruit.draw(screen)
                    
                    # reset list containing pos of knife trail circles for cutting
                    left_circles = []
                    right_circles = []

                    # remove old knife trail points from knife_trail lists, and draw new lines
                    if len(left_knife_trail) >= 2:
                        for i, point in enumerate(left_knife_trail):
                            coords, time_painted = point
                            
                            # if the knife trail point still has lifetime
                            if time_painted + KNIFE_TRAIL_LIFETIME > time.time(): 

                                # if there's another knife trail point after this one
                                if len(left_knife_trail) - 1 > i:                       # anti-aliased line
                                    left_circles += calculations.knife_trail(
                                        screen, 
                                        BLACK, 
                                        coords, 
                                        left_knife_trail[i + 1][0], 
                                        radius=KNIFE_WIDTH)
                            else:
                                left_knife_trail.remove(point) # remove knife trail point because its lifetime is now over

                    if len(right_knife_trail) >= 2:
                        for i, point in enumerate(right_knife_trail):
                            coords, time_painted = point

                            if time_painted + KNIFE_TRAIL_LIFETIME > time.time():
                                if len(right_knife_trail) - 1 > i:
                                    right_circles += calculations.knife_trail(
                                        screen, 
                                        BLACK, 
                                        coords, 
                                        right_knife_trail[i + 1][0], 
                                        radius=KNIFE_WIDTH)
                            else:
                                right_knife_trail.remove(point)

                    # if start fruit exists (in the beginning)
                    if type(start_fruit) is Fruit: 
                        destroy = draw_start_end_fruit(
                            start_fruit,
                            left_circles,
                            right_circles,
                            last_round) # pass reference of start_fruit

                        if destroy:
                            if main_music != None:
                                main_music.stop()

                            play_sound("C:/Users/Admin/Desktop/Gesture_Gaming/Audios/game start.wav", 0.5)
                            play_sound(random.choice(fruit_slice_sounds))
                            last_round = time.time()
                            start_fruit = None
                            start_game = True
                    elif start_game == False: # no start fruit yet, but game hasn't started
                        img_size = (ratio(320), ratio(400))
                        fruit_pos = (
                            GAME_WIDTH / 2 - img_size[0] / 2, 
                            GAME_HEIGHT / 2 - img_size[1] / 2)
                        start_fruit = Fruit( # create fruit object
                            name="Start Fruit", 
                            img_filepath="C:/Users/Admin/Desktop/Gesture_Gaming/Images/watermelon.png", 
                            starting_point=fruit_pos,
                            size=img_size,
                            velocity=0,
                            points=0)
                        last_round = time.time()
                    elif exploding or (round_number >= 3 and total_points < 0 and len(fruits) == 0):
                        # make sad fruit if it doesn't exist already
                        if dead_fruit is None:
                            img_size = (ratio(300), ratio(300))
                            fruit_pos = (
                                GAME_WIDTH / 2 - img_size[0] / 2, 
                                (GAME_HEIGHT - GAME_HEIGHT / 4) - img_size[1] / 2)
                            dead_fruit = Fruit(
                                name="sadness",
                                img_filepath=random.choice(sad_fruit),
                                starting_point=fruit_pos,
                                size=img_size,
                                velocity=0,
                                points=0
                            )
                            last_round = time.time()

                            if exploding: # if exploding, but hadn't made dead_fruit yet
                                print("Player hit a bomb, so the round ended.")
                                
                                for fruit in fruits:
                                    fruit.stop_bomb_sound()

                            play_sound("C:/Users/Admin/Desktop/Gesture_Gaming/Audios/game over.wav", 0.6)

                        # stop all fruits
                        for fruit in fruits:
                            fruit.velocity = 0
                        
                        # fade to white (explosion)
                        if explosion_alpha < 200:
                                explosion_alpha += 5

                        # YOU LOST text
                        explosion = pygame.Surface([GAME_WIDTH, GAME_HEIGHT], pygame.SRCALPHA, 32)
                        explosion = explosion.convert_alpha()
                        explosion.fill((255, 255, 255, explosion_alpha))
                        screen.blit(explosion, (0, 0))

                        lost_text = TITLE_FONT.render('You Died!' if exploding else 'You Lost!', False, NINJA_RED)
                        screen.blit(lost_text, (
                            round(GAME_WIDTH / 2) - lost_text.get_width() / 2, 
                            round(GAME_HEIGHT / 3) - lost_text.get_height() / 2))
                        
                        # sad fruit
                        destroy = draw_start_end_fruit(
                            dead_fruit, 
                            left_circles, 
                            right_circles, 
                            last_round)

                        # start new game if sad fruit is cut
                        if destroy:
                            print("Starting new game!")
                            exploding = False
                            fruits.clear()
                            round_number = 0
                            total_points = 0
                            last_round = time.time()
                            dead_fruit = None
                            explosion_alpha = 0
                            play_sound(random.choice(fruit_slice_sounds))
                            play_sound("C:/Users/Admin/Desktop/Gesture_Gaming/Audios/game start.wav")

                    # if no more fruits, set cooldown
                    if (len(fruits) == 0 
                        and time.time() - last_round > ROUND_COOLDOWN
                        and round_number > 0
                        and not dead_fruit is None):
                        last_round = time.time()
                        print(f"Round {round_number} done!")
                    
                    # update display, render menu
                    end = time.time()
                    fps = round(1 / (end - start), 1)
                    display_menu(
                        fps,
                        round_number if start_game else None, 
                        total_points if start_game else None, 
                        len(fruits) if start_game else None)
                    #pygame.display.update()

                    # keyboard press events
                    keys_pressed = pygame.key.get_pressed()

                    # escape key used to close game
                    if keys_pressed[pygame.K_ESCAPE]:
                        pygame.event.post(pygame.event.Event(pygame.QUIT))

                    # pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit() # quit pygame window, stop game loop
                            running = False

                    #frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                # Display the frame using Streamlit's st.image
                #frame_placeholder.image(image, channels="RGB")

                    # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
                    if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed: 
                        use_webcam=False
                        break

                    #fps=23
                    #currTime = time.time()
                    #fps = 1 / (currTime - prevTime)
                    #prevTime = currTime
                    

                    #Dashboard
                    kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                    kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
                    kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

                    frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
                    frame = image_resize(image = frame, width = 640)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    image = pygame.surfarray.array3d(screen).swapaxes(0, 1)
                    image = np.rot90(image, 1, (1, 0))
                    #----------------------------------------------
                    flipped_image = cv2.flip(image, -1)
                    rotated_image = cv2.rotate(flipped_image, cv2.ROTATE_90_CLOCKWISE)
                    stframe.image(rotated_image,channels = 'RGB',use_column_width=True)


        def play_sound(file, volume=1):
            sound = pygame.mixer.Sound(file)
            sound.set_volume(volume)
            sound.play()
            return sound

        def play_sound_non_blocking(file):
            thread = threading.Thread(target=play_sound, args=(file,))
            thread.start()

        def draw_start_end_fruit(
            fruit: Fruit, 
            left_circles: list, 
            right_circles: list, 
            last_round: float) -> bool:
            circle_radius = round(fruit.get_length() * 0.65)
            hit_left = False
            
            # wait so that you don't accidentaly cut the start or end fruit just by touching it
            if time.time() - last_round >= ROUND_COOLDOWN: 
                for left_circle in left_circles:
                    if calculations.distance_2D(fruit.get_centre(), left_circle) <= circle_radius:
                        hit_left = True
                        break

                if hit_left: # must hit with both hands!
                    for right_circle in right_circles:
                        if calculations.distance_2D(fruit.get_centre(), right_circle) <= circle_radius:
                            return True

            pygame.draw.circle(
                screen,
                RED,
                fruit.get_centre(),
                radius=circle_radius,
                width=round(fruit.get_length() / 15))
            fruit.rotation += 2
            fruit.draw(screen)
            return False

        def cut_fruit(fruit: Fruit, exists_in_list=True):    
            if exists_in_list: # if fruit is in the fruits list
                fruits.remove(fruit)

            # fruit cutting effects here
            play_sound(random.choice(fruit_slice_sounds))
            return fruit.points

        def display_menu(fps, round_number=None, total_points=None, fruits_left=None):
            fps_text = SMALL_NUMBER_FONT.render(f'FPS: {fps}', False, NINJA_RED)
            screen.blit(fps_text, (ratio(20), GAME_HEIGHT - fps_text.get_height() - ratio(20)))

            if type(fruits_left) is int:
                fruits_left_text = SMALL_NUMBER_FONT.render(f'Fruits left: {fruits_left}', False, NINJA_RED)
                screen.blit(
                    fruits_left_text, 
                        (GAME_WIDTH - fruits_left_text.get_width() - ratio(20), 
                        GAME_HEIGHT - fruits_left_text.get_height() - ratio(20)))

            if type(total_points) is int:
                points_text = MAIN_NUMBER_FONT.render(f'Points: {total_points}', False, WHITE)
                screen.blit(points_text, (ratio(20), ratio(20)))

            if type(round_number) is int:
                round_text = MAIN_NUMBER_FONT.render(f'Round {round_number}', False, WHITE)
                screen.blit(round_text, (GAME_WIDTH - round_text.get_width() - ratio(20), ratio(20)))
            else:
                starting_title = TITLE_FONT.render(f'Cut with both hands to begin!', False, NINJA_RED)
                screen.blit(starting_title, (
                    round(GAME_WIDTH / 2) - starting_title.get_width() / 2, 
                    GAME_HEIGHT - starting_title.get_height() - ratio(30)))

            title_text = TITLE_FONT.render(f'Fruit Ninja 3D', False, NINJA_RED)
            screen.blit(title_text, (round(GAME_WIDTH / 2) - title_text.get_width() / 2, ratio(20)))

        def fruit_hit_circles(fruit: Fruit, points: list):
            for point in points:
                if fruit.rect.collidepoint(point): # check if it's in rectangle first
                    if calculations.colliding_fruit(point, fruit):
                        return True
            return False

        def process_fruit(
            fruit: Fruit, 
            left_hand, right_hand, 
            left_circles,
            right_circles) -> tuple:
            points = 0
            bomb_touched = False

            # if fruit is under the screen and it's on its way down
            if fruit.y > GAME_HEIGHT and not fruit.going_up:
                fruit.stop_bomb_sound()
                fruits.remove(fruit)
                points -= fruit.points * 2
                return points, bomb_touched
                
            # if fruit is finally on screen and we didn't play its spawn sound yet
            elif fruit.y < GAME_HEIGHT and fruit.going_up and not fruit.spawn_sound_played:
                if fruit.name == "bomb":
                    fruit.play_bomb_sound()
                else:
                    play_sound("C:/Users/Admin/Desktop/Gesture_Gaming/Audios/fruit spawn.wav")

                fruit.spawn_sound_played = True

            if fruit.name == "bomb":
                if (fruit.rect.collidepoint(left_hand) # if either hands touch the bomb
                    or fruit.rect.collidepoint(right_hand)):
                        bomb_touched = True
                        play_sound("C:/Users/Admin/Desktop/Gesture_Gaming/Audios/explode.wav")
            else:
                if screen.get_rect().collidepoint((fruit.x, fruit.y)): # if left hand is in frame
                    if fruit_hit_circles(fruit, left_circles): # separate if's to not check hit circles twice (laggy)
                        points = cut_fruit(fruit)
                        return points, bomb_touched

                    if fruit_hit_circles(fruit, right_circles):
                        points = cut_fruit(fruit)
                        return points, bomb_touched
            
            # if reached max_fruit_height, time to go down!
            if fruit.get_centre()[1] <= MAX_FRUIT_HEIGHT and fruit.going_up:
                fruit.going_up = False
                fruit.velocity *= -1
            else:
                fruit.y -= fruit.velocity
                fruit.rotation += abs(round(fruit.velocity / 3))
            return points, bomb_touched

        def make_new_fruits(number_of_fruits, round_number):
            probabilities = []

            # bombs should not appear until round 4
            for fruit_n in fruit_names:
                if fruit_n[0] == "bomb" and round_number < 4: # name of fruit
                    probabilities.append(0)
                else:
                    probabilities.append(fruit_n[4]) # rarity of fruit

            random_fruits = random.choices(fruit_names, weights = probabilities, k = number_of_fruits)

            for fruit in random_fruits:
                fruit_name, img_path, points, velocity, _, img_size = fruit # unwrap list containing details of fruit
                print(f"Generating a new {fruit_name}")

                # ratio velocity
                velocity = ratio(velocity)
                
                # random velocity change
                velocity *= 1 + random.random() / 2

                random_x = random.randrange(1, GAME_WIDTH - img_size[0]) # random horizontal starting point
                random_y = random.randrange(GAME_HEIGHT, GAME_HEIGHT * 3) # random start point under screen
                
                new_fruit = Fruit( # create fruit object
                    name=fruit_name, 
                    img_filepath=img_path, 
                    starting_point=(random_x, random_y),
                    size=(ratio(img_size[0]), ratio(img_size[1])),
                    velocity=velocity,
                    points=points)
                fruits.append(new_fruit) # add fruit to list of fruits that will be displayed

        if __name__ == '__main__':
            main()

        # Use this line to capture video from the webcam
        #cap = cv2.VideoCapture(0)


        # Set the title for the Streamlit app
        #st.title("Video Capture with OpenCV")

        frame_placeholder = st.empty()

        # Add a "Stop" button and store its state in a variable
        stop_button_pressed = st.button("Exit Game")

        with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
            prevTime = 0
            while cap.isOpened() and not stop_button_pressed:
                ret, frame = cap.read()

                if not ret:
                    st.write("The video capture has ended.")
                if stop_button_pressed:
                    use_webcam=False                 

                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                face_count = len(faces)
                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                image = cv2.flip(image, 1)
                
                imageHeight, imageWidth, _ = image.shape
        
                results = hands.process(image)
        
        
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
                font=cv2.FONT_HERSHEY_SIMPLEX
                color=(255,0,255)
                text=cv2.putText(image,"Score",(480,30),font,1,color,4,cv2.LINE_AA)
                text=cv2.putText(image,str(score),(590,30),font,1,color,4,cv2.LINE_AA)
        
                enemy()
        
                if results.multi_hand_landmarks:
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                                )
        
        
                if results.multi_hand_landmarks != None:
                    for handLandmarks in results.multi_hand_landmarks:
                        for point in mp_hands.HandLandmark:
            
                
                            normalizedLandmark = handLandmarks.landmark[point]
                            pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
                
                            point=str(point)
                            #print(point)
                            if point=='HandLandmark.INDEX_FINGER_TIP':
                                try:
                                    cv2.circle(image, (pixelCoordinatesLandmark[0], pixelCoordinatesLandmark[1]), 25, (0, 200, 0), 5)
                                    #print(pixelCoordinatesLandmark[1])
                                    if pixelCoordinatesLandmark[0]==x_enemy or pixelCoordinatesLandmark[0]==x_enemy+10 or pixelCoordinatesLandmark[0]==x_enemy-10:
                                        #if pixelCoordinatesLandmark[1]==y_enemy or pixelCoordinatesLandmark[1]==y_enemy+10 or pixelCoordinatesLandmark[1]==y_enemy-10:
                                    #if pixelCoordinatesLandmark[1]==y_enemy or pixelCoordinatesLandmark[1]==y_enemy+10 or pixelCoordinatesLandmark[1]==y_enemy-10:
                                        print("found")
                                        x_enemy=random.randint(50,600)
                                        y_enemy=random.randint(50,400)
                                        score=score+1
                                        font=cv2.FONT_HERSHEY_SIMPLEX
                                        color=(255,0,255)
                                        text=cv2.putText(frame,"Score",(100,100),font,1,color,4,cv2.LINE_AA)
                                        enemy()
                                except:
                                    pass
                
                #cv2.imshow('Hand Tracking', image)
                #time.sleep(1)
        
        
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Display the frame using Streamlit's st.image
                #frame_placeholder.image(image, channels="RGB")

                # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
                if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed: 
                    use_webcam=False
                    break


                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                #Dashboard
                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
                kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

                frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
                frame = image_resize(image = frame, width = 640)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                stframe.image(image,channels = 'BGR',use_column_width=True)

elif game_mode =='Pong Game':
    from cvzone.HandTrackingModule import HandDetector
    import cvzone
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    #cap = cv2.VideoCapture(0)

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    #record = st.sidebar.checkbox("Record Video")
    #if record:
    #    st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )
    # max faces

    st.sidebar.write("Download the game and  run on your PC")
    with open("C:/Users/Admin/Desktop/Gesture_Gaming/Zips/Subway_Surfers.zip", "rb") as file:
        btn = st.sidebar.download_button(
            label="Download Here",
            data=file,
            file_name="Subway_Surfers.zip",
            mime="application/zip"
          )

    #max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Pong Game')

    stframe = st.empty()

    #video_file_buffer = False#st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    if not use_webcam:   
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/pong.png')
        #image = Image.open('ss.jpg')      
        st.image(image, caption='Gameplay of Dino Game')
        st.markdown('''
          ### Game Description \n 
            Pong game is a classical game right from the old days.\n
            How about playing the same game with **Hand Gestures?** :sunglasses: \n
             \n

            Enable the webcam to begin the fun.  \n
            As soon as hands are visible to the camera,the two paddels are placed at the opposide sides of the screen and the ball starts moving.
            The objective of this game is to hit the ball back by moving the hands.\n
            Keep the rally as long as possible\n
            1 Point for each hit is awarded\n
            The Game ends when the ball hits either of the horizontal edges.
            ''')

    if use_webcam:
        vid = cv2.VideoCapture(0)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        #<--!codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)-->
        #<--!codec = cv2.VideoWriter_fourcc('V','P','0','9')-->
        #<--!out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

        #st.sidebar.text('Input Video')
        #st.sidebar.video(tfflie.name)
        fps = 0
        i = 0
        #drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)


        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Detected Faces**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)
       
        # Use this line to capture video from the webcam
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        imgBackground = cv2.imread("C:/Users/Admin/Desktop/Gesture_Gaming/Images/Background.png")
        imgGameOver = cv2.imread("C:/Users/Admin/Desktop/Gesture_Gaming/Images/gameOver.png")
        imgBall = cv2.imread("C:/Users/Admin/Desktop/Gesture_Gaming/Images/Ball.png", cv2.IMREAD_UNCHANGED)
        imgBat1 = cv2.imread("C:/Users/Admin/Desktop/Gesture_Gaming/Images/bat1.png", cv2.IMREAD_UNCHANGED)
        imgBat2 = cv2.imread("C:/Users/Admin/Desktop/Gesture_Gaming/Images/bat2.png", cv2.IMREAD_UNCHANGED)

        # Set the title for the Streamlit app
        #st.title("Video Capture with OpenCV")

        frame_placeholder = st.empty()

        # Add a "Stop" button and store its state in a variable
        stop_button_pressed = st.button("Exit Game")
        prevTime=0
        ballPos = [100, 100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0, 0]

        while True or not stop_button_pressed:
            _, img = cap.read()
            img = cv2.flip(img, 1)
            imgRaw = img.copy()

            # Find the hand and its landmarks
            hands, img = detector.findHands(img, flipType=False)

            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            face_count = len(faces)

            # Overlaying the background image
            img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check for hands
            if hands:
                for hand in hands:
                    x, y, w, h = hand['bbox']
                    h1, w1, _ = imgBat1.shape
                    y1 = y - h1//2
                    y1 = np.clip(y1, 20, 415)

                    if hand['type'] == "Left":
                        img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                        if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                            speedX = -speedX # Change OX direction 
                            ballPos[0] += 30
                            #score[0] += 1

                    if hand['type'] == "Right":
                        img = cvzone.overlayPNG(img, imgBat2, (1195, y1))
                        if 1195 - 50 < ballPos[0] < 1195 + w1 and y1 < ballPos[1] < y1 + h1:
                            speedX = -speedX # Change OX direction
                            ballPos[0] -= 30
                            #score[1] += 1

             # Game Over
            if ballPos[0] < 40 :
                score[1]+=1
                img = cvzone.overlayPNG(img, imgBall, [100,100])
            if ballPos[0] > 1200:
                score[0]+=1
                img = cvzone.overlayPNG(img, imgBall, [100,100])
            if score[0]==3 or score[1]==3:
                gameOver = True
            
            if gameOver:
                img = imgGameOver
                cv2.putText(img, str(score[1] + score[1]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX, 2.5, (200, 0, 200), 5)

            # If game not over move the ball
            else:

                # Move the ball
                    # Change OY direction
                if ballPos[1] >= 500 or ballPos[1] <= 10:
                    speedY = -speedY

                ballPos[0] += speedX
                ballPos[1] += speedY

                # Draw the ball
                img = cvzone.overlayPNG(img, imgBall, ballPos)

                # Display score on the image
                    # Left player (hand)
                cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
                    # Rigth player (hand)
                cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

            # Cam show
            #img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120)) 

            #cv2.imshow("Image", img)
            key = cv2.waitKey(1)

            # Reload the game by pressing "r"
            if key == ord("r"):
                ballPos = [100, 100]
                speedX = 15
                speedY = 15
                gameOver = False
                score = [0, 0]
                imgGameOver = cv2.imread("C:/Users/Admin/Desktop/Gesture_Gaming/Images/gameOver.png")


            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            #face_count=1
            #Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(img,(0,0),fx = 0.8 , fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            stframe.image(image,channels = 'BGR',use_column_width=True)

elif game_mode =='Temple Run':
    #from time import time
    import cvzone
    import time as tp
    from cvzone.HandTrackingModule import HandDetector

  
    st.set_option('deprecation.showfileUploaderEncoding', False)

    
    use_webcam = st.sidebar.button('Enable Webcam and Play Now')
    #record = st.sidebar.checkbox("Record Video")
    #if record:
    #    st.checkbox("Recording", value=True)
    
    
    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )
    # max faces

    st.sidebar.write("Download the game and  run on your PC")
    with open("C:/Users/Admin/Desktop/Gesture_Gaming/Zips/Subway_Surfers.zip", "rb") as file:
        btn = st.sidebar.download_button(
            label="Download Here",
            data=file,
            file_name="Subway_Surfers.zip",
            mime="application/zip"
          )
    
    #max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')
    
    st.markdown(' ## Temple Run')

    stframe = st.empty()
    #video_file_buffer = False#st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    #vid = cv2.VideoCapture(0)
    if not use_webcam:
        col1, col2 = st.columns(2)
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/temple_run2.png')  
        col1.image(image,width=250)   
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/temple_run1.jpg')        
        col2.image(image,width=200)  
        
        use_webcam = st.button("Play Now!")


        st.markdown('''
          ### Game Description \n 
            Temple Run is a popular endless runner game that combines high-speed action, vibrant visuals, and exciting gameplay.\n
            How about playing the same game with **Body Gestures?** :sunglasses: \n
             \n

             Enable the webcam to begin the fun.  \n
            ''')


        st.text('''After the camera turns up Join both the hands to start the game\n
The basic 4 operations are :
    - Left
    - Right
    - Jump
    - Crouch
''')    
        col1, col2 = st.columns(2)
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/up.png')  
        col1.image(image,caption="To Jump",width=200)   
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/down.jpg')        
        col2.image(image,caption="To Crouch",width=200)
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/left.png') 
        col1.image(image,caption="To turn right",width=200)   
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/right.png')        
        col2.image(image,caption="To turn left",width=200)
        image = Image.open('C:/Users/Admin/Desktop/Gesture_Gaming/Images/horizontal_movements.png')    
        st.markdown('''### As shown in the above images,\n
    - To Move Left  :  Raise the Little finger and close other fingers\n
    - To Move Right :  Raise the Thumb finger and close other fingers\n
    - To Jump       :  Raise the Index finger and close other fingers\n
    - To Crouch     :  Raise the Index finger and **Little finger** close other fingers\n\n
        ''')

    if use_webcam:
        cap = cv2.VideoCapture(0)
        cap.set(3,1280)
        cap.set(4,960)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))

        #<--!codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)-->
        #<--!codec = cv2.VideoWriter_fourcc('V','P','0','9')-->
        #<--!out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

        #st.sidebar.text('Input Video')
        #st.sidebar.video(tfflie.name)
        fps = 0
        i = 0
        #drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)


        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Detected Faces**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)

        # Use this line to capture video from the webcam
        cap = cv2.VideoCapture(0)
        detector = HandDetector(detectionCon=0.8, maxHands=1)


        # Set the title for the Streamlit app
        #st.title("Video Capture with OpenCV")

        #cv2.namedWindow('Subway Surfers with Pose Detection', cv2.WINDOW_NORMAL)
        #tp.sleep(3)

        #pyautogui.keyDown(key='ctrl')            
        #pyautogui.typewrite('n')
        #pyautogui.keyUp(key='ctrl')  

        webbrowser.open("https://templerun.ee/")
        tp.sleep(3)

        #pyautogui.keyDown(key='win')            
        #pyautogui.press('left') 
        #pyautogui.keyUp(key='win') 

        #pyautogui.typewrite('\n')
        time1 = 0

        frame_placeholder = st.empty()

        # Add a "Stop" button and store its state in a variable
        stop_button_pressed = st.button("Exit Game")

        prevTime=0
        while True and not stop_button_pressed:
            success, img = cap.read()
            if stop_button_pressed:
                use_webcam=False

            if not success:
                continue
            
            img=cv2.flip(img,1)

            hands,img=detector.findHands(img)
            delayCounter=0

            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            face_count = len(faces)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if hands:
                hand1=hands[0]
                finger=detector.fingersUp(hand1)

                if finger[0]==1 and finger[1]==0 and finger[2]==0 and finger[3]==0 and finger[4]==0 and delayCounter==0:
                    #pyautogui.press("left")
                    cvzone.putTextRect(img,"Pointing Left!",(50,80),scale=3,thickness=3,offset=10)

                if finger[4]==1 and finger[1]==0 and finger[2]==0 and finger[3]==0 and finger[0]==0 and delayCounter==0:
                    #pyautogui.press("right")
                    cvzone.putTextRect(img,"Pointing Right!",(50,80),scale=3,thickness=3,offset=10)

                if finger[1]==1 and finger[0]==0 and finger[2]==0 and finger[3]==0 and finger[4]==0 and delayCounter==0:
                    #pyautogui.press("up")
                    cvzone.putTextRect(img,"Pointing Up!",(50,80),scale=3,thickness=3,offset=10)

                if finger[0]==1 and finger[1]==1 and finger[2]==0 and finger[3]==0 and finger[4]==1 and delayCounter==0:
                    #pyautogui.press("down")
                    cvzone.putTextRect(img,"Pointing Down!",(50,80),scale=3,thickness=3,offset=10)

                if delayCounter != 0:
                    delayCounter += 1
                    if delayCounter > 50:
                        delayCounter = 0

            #cv2.imshow("img",img)
            
            if cv2.waitKey(1)==27 or cv2.waitKey(1)==ord('q') or stop_button_pressed:
                cv2.destroyAllWindows()
                cap.release()

                                
            #cap.release()
            #cv2.destroyAllWindows()
                    


            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            
            #Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(img,(0,0),fx = 0.8 , fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            stframe.image(image,channels = 'RGB',use_column_width=True)

        #vid.release()
        #out. release()