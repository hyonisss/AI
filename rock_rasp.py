import tflite_runtime.interpreter as tflite
import numpy as np
import cv2 

# Loading Model Information
interpreter = tflite.Interpreter(model_path = "/home/pi/rock/rock.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

cap=cv2.VideoCapture(0)     #  0th camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
r, g, b = 255,0,0

labels = ['paper', 'rock', 'scissors']

while(cap.isOpened()):
    ret,frame=cap.read()

    if(ret):
        import cv2

        img = frame[:,:,0] * 0.2989 + frame[:,:,1] * 0.5870 +  frame[:,:,2]*0.1140
        gray = cv2.resize(img,(height,width))
        #gray = cv2.resize(255-img,(height,width))
        #gray = gray//(np.max(gray)*0.7)
        if floating_model == True :     # 트레이닝된 모델이 float32면 해당 타입으로 캐스팅
            gray = (np.float32(gray))
        cv2.imshow('cam1',gray)

        #cv2.imshow('result',gray)  # 전처리된 이미지파일 출력

        test_num = gray.flatten()
        test_num = test_num.reshape((-1,height,width,1))

        interpreter.set_tensor(input_details[0]['index'], test_num)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # 정답 출력을 위한 코드
        ans = np.argmax(output_data)
        print('The Answer is ', ans)
        #text = str(ans)
        r,g,b = g,b,r
        text = labels[ans]
        cv2.putText(frame, text, (50,100), cv2.FONT_ITALIC, 5, (r, g, b),3)
        cv2.namedWindow('cam')
        cv2.moveWindow('cam',50,50)
        cv2.imshow('cam',frame)
        
    cv2.waitKey(200)   # 200ms  대기

cap.release()
cv2.destroyAllWindows()
