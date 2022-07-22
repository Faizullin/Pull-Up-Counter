import cv2, time
import numpy as np
import tensorflow as tf
import pyttsx3
from threading import Thread


def say_t(words):
    engine=pyttsx3.init()
    engine.say(words)
    engine.runAndWait()
def talk(words):
    Thread(target=say_t,args=(words,),daemon=True).start()
    
    
cv_show={
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "bottomLeftCornerOfText":(10,400),
    "fontScale": 1,
    "fontColor":(255,255,255),
    "lineType": 5
}


def push_up(push_up_model,show= False,max_count=15,talk_every_count=5,not_do_delay=5):
    global cv_show, cap
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    repetitions = 0;prev_repetitions=0
    up = 0; down = 0; no_move = 0; current_move = 0
    initial = -1
    talk('начинаем')
    time.sleep(1)
    not_do_time = time.time()
    while True:
        if time.time() - not_do_time > not_do_delay:
            if repetitions==prev_repetitions:
                print('не остонавливайтесь')
            not_do_time=time.time()
        ret, frame2 = cap.read()
        #if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            #break
        nextt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,nextt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        image = cv2.resize(rgb, (64, 64))
        image = image.reshape((1,) + image.shape)
        image = image/255.0
        prediction = np.argmax(push_up_model.predict(image), axis=-1)[0]
        if prediction == 0:
            down +=1
            if down == 3:
                if initial == -1:
                    initial = 0
                if current_move == 2:
                    repetitions+=1
                    if repetitions%talk_every_count==0:
                        prev_repetitions=repetitions
                        talk(repetitions)
                        if repetitions==max_count:
                            time.sleep(1.5)
                            break
                current_move = 0
            elif down > 0:
                up = 0
                no_move = 0
        elif prediction == 2:
            up += 1
            if up == 3 and initial != -1:
                current_move = 2
            elif up > 1:
                down = 0
                no_move = 0
        else:
            no_move += 1
            if no_move == 15:
                current_move = 1
            elif no_move > 10:
                up = 0
                down = 0
        if show:
            cv2.putText(frame2, "Prediction:"+str(prediction)+"  Repetitions:"+ str(repetitions),
                       cv_show['bottomLeftCornerOfText'],cv_show['font'], cv_show['fontScale'],cv_show['fontColor'],cv_show['lineType'])
            prvs = nextt
            cv2.imshow('P',frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    talk('конец отдохните')
    time.sleep(1.3)

def wait(delay=30):
    talk('отдых '+str(delay)+' секунд')
    if delay%2==0:
        d2=delay/2
    else:
        d2=(delay//2)+1
    time.sleep(d2)
    talk('отдых '+str(int(delay-d2))+' секунд')
    time.sleep(delay-d2)
    talk('конец отдыха')
    time.sleep(1)

if __name__=='__main__':
    cap = cv2.VideoCapture(1)#index
    push_up_model = tf.keras.models.load_model('models/model.h5')
    push_up(push_up_model,True)
    wait()
    push_up(push_up_model)
    cap.release()
