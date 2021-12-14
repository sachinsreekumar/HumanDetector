import cv2
import numpy as np
import argparse
import imutils

def detect(frame):
    #Human detection function
    box_cordinates, weights = hog_cv.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03) #Getting human coordinates and weights
    print(box_cordinates)
    print(weights)
    person_count = 1
    #Printing boxes around detected humans
    for x, y, w, h in box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Person {person_count}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        person_count += 1
    cv2.putText(frame, f'Total Persons : {person_count - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 2, 0), 2)
    cv2.imshow('output', frame)
    return frame

#To handle human detection from saved video
def detectByPathVideo(path, writer):
    input_video = cv2.VideoCapture(path)                            #Taking image from path
    check, frame = input_video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path')
        return
    while input_video.isOpened():
        # check is True if reading was successful
        print("test")
        check, frame = input_video.read()
        if check:
            frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            frame = detect(frame)
            if writer is not None:
                writer.write(frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    input_video.release()
    cv2.destroyAllWindows()

#To handle human detection from Cam
def detectByCamera(writer):
    print("test")
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)                  #Get webcam access
    if not video.isOpened():
        raise IOError("Unable to open webcam")
    print('Detecting people...')
    while video.isOpened():
        print("Test")
        # check is True if reading was successful
        check, frame = video.read()                             #Live video to frames
        if check:
            frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            frame = detect(frame)                                #call detect function for each frame
            if writer is not None:
                writer.write(frame)
            key = cv2.waitKey(1)
            if key == ord('e'):                                 #Press e to end the cam
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()

#To handle human detection from saved image
def detectByPathImage(path, output_path):
    image = cv2.imread(path)
    image = imutils.resize(image, width=min(800, image.shape[1]))
    result_image = detect(image)
    if output_path is not None:
        cv2.imwrite(output_path, result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#To identify input based on arguments
def humanDetect(args):
    #Gettning input parameters from command
    image_path = args["image"]
    video_path = args['video']
    camera = True if str(args["camera"]) == 'True' else False
    op_args = args['output']
    print(camera)

    op_writer = None

    if op_args is not None and image_path is None:                                     #If input is a video and has to be saved
        op_writer = cv2.VideoWriter(op_args, cv2.VideoWriter_fourcc(*'MJPG'),10,(600, 600))
    if camera:                                                                         #If input is web cam
        print('----Opening Web Cam.-----')
        detectByCamera(op_writer)
    elif video_path is not None:                                                       #If input is video taken from a path
        print('-----Opening Video from path.----')
        detectByPathVideo(video_path, op_writer)
    elif image_path is not None:                                                       #If input is an image taken from a path
        print('----Opening Image from path.----')
        detectByPathImage(image_path, args['output'])

def argParser():
    parser = argparse.ArgumentParser()                                                              #Argument Parser Object creation

    #Setting input arguments
    parser.add_argument("-v", "--video",default=None,help="Video File path")
    parser.add_argument("-c", "--camera",default=False,help="To use camera, provide True in the input")
    parser.add_argument("-i", "--image",default=None,help="Image path")
    parser.add_argument("-o", "--output",type=str,help="output location path")

    ip_args = vars(parser.parse_args())
    return ip_args

if __name__ == "__main__":
    hog_cv = cv2.HOGDescriptor()
    hog_cv.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    input_args = argParser()
    humanDetect(input_args)