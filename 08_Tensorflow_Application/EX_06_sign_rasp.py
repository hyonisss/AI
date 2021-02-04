# 모듈 로딩
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2 

# TFLite 로딩
interpreter = tflite.Interpreter(model_path = "/home/pi/sign/sign.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# 카메라 설정
cap=cv2.VideoCapture(0) # 0th camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
r,g,b = 255,0,0

while(cap.isOpened()):

	ret,frame=cap.read() # 사진 찍기: (480,640,3)

	if(ret):

		# 사이즈 축소: (480,640,3) -> (64,64,3)
		img = cv2.resize(frame,(height,width))

		# 학습된 모델이 float32면 해당 타입으로 캐스팅
		if floating_model == True :     
			img = (np.float32(img))

		# 0 ~ 1 로 정규화
		img = img / 255.0

		# 전처리된 이미지파일 출력
		#cv2.imshow('result',img)  

		# 모델의 입력 형태로 수정: (1,64,64,3)
		test_num = img.reshape((-1,height,width,3))

		# 모델에 입력하여 결과 얻기
		interpreter.set_tensor(input_details[0]['index'],test_num)
		interpreter.invoke()
		output_data = interpreter.get_tensor(output_details[0]['index'])

		# 예측 결과 출력
		ans = np.argmax(output_data)
		print('The Answer is ',ans)
		text = str(ans)
		r,g,b = g,b,r
		cv2.putText(frame,text,(50,200),cv2.FONT_ITALIC,5,(r,g,b),3)
		cv2.namedWindow('cam')
		cv2.moveWindow('cam',50,50)
		cv2.imshow('cam',frame)
		
	cv2.waitKey(200) # 200ms 대기

cap.release()
cv2.destroyAllWindows()
