from face_detector import YoloDetector
import numpy as np
from PIL import Image, ImageDraw
import cv2
import pafy

model = YoloDetector(target_size=720,gpu=0,min_face=90)

def detect_camera(cam_num = 0):
    # imgcp.save('DC1.jpg')
    # imgcp.show()

    # define a video capture object
    vid = cv2.VideoCapture(cam_num)
    
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        
        orgimg = np.array(frame)
        bboxes,points = model.predict(orgimg)

        # image = frame
        # imgcp = image.copy()
        # imgcp_draw = ImageDraw.Draw(imgcp)

        for i in  range(len(bboxes)):
            coord = bboxes[i]
            for j in range(len(coord)):
                det = coord[j]
                #imgcp_draw.rectangle(det, fill = None, outline = "red")
                cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), (0, 0, 255), 2)
        for pp in points:
            for kk in pp:
                for ii in kk:
                    # cv2.ellipse((ii[0], ii[1], ii[0]+5, ii[1]+5), fill = 'blue', outline ='blue')
                    cv2.circle(frame, ii, 3, (0,255,0), -1)
                    # imgcp_draw.point(ii, fill= "green")
        
        #print(bboxes)
        #print(points)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def detect_video(url = "https://www.youtube.com/watch?v=2fhBvH_-tNM", is_local = False):

    
    if(is_local):
        vid = cv2.VideoCapture(url)
    else:
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")

        vid = cv2.VideoCapture(best.url)

    
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        
        orgimg = np.array(frame)
        bboxes,points = model.predict(orgimg)

        # image = frame
        # imgcp = image.copy()
        # imgcp_draw = ImageDraw.Draw(imgcp)

        for i in  range(len(bboxes)):
            coord = bboxes[i]
            for j in range(len(coord)):
                det = coord[j]
                #imgcp_draw.rectangle(det, fill = None, outline = "red")
                cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), (0, 0, 255), 2)
        for pp in points:
            for kk in pp:
                for ii in kk:
                    # cv2.ellipse((ii[0], ii[1], ii[0]+5, ii[1]+5), fill = 'blue', outline ='blue')
                    cv2.circle(frame, ii, 3, (0,255,0), -1)
                    # imgcp_draw.point(ii, fill= "green")
        
        #print(bboxes)
        #print(points)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


detect_video(is_local= True, url="./videos/goals-worth-watching-again-3.mp4")