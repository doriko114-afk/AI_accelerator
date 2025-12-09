ai 가속기 딥러닝 알고리즘

<img width="865" height="368" alt="image" src="https://github.com/user-attachments/assets/2ae6593a-ef91-483a-9b49-056b236cf8d2" />


학습률의 경우 일반적으로 너무 큰값이 아닌 0.001을 사용하여 조금씩 증감하여 사용

가중치와 편향치를 구하기위해 학습을 진행

에포크 반복학습

학습 코드 
```
x=2
w=3
b=1
yT=10
lr = 0.01

for epoch in range(2):
    
    y = x*w + 1*b
    E = (y -yT)**2/2
    yE = y-yT
    wE = yE*x
    bE = yE*1
    w -= lr*wE
    b -= lr*bE
    
    
    print(f'epoch = {epoch}')
    print(f' y : {y: .3f}')
    print(f' w :{w: .3f}')
    print(f' b :{b: .3f}')
    
```

<img width="460" height="186" alt="image" src="https://github.com/user-attachments/assets/547395ff-f67a-437f-a540-11b098aead0d" />

(2회 학습)

<img width="178" height="143" alt="image" src="https://github.com/user-attachments/assets/d0016525-fa35-49f6-9937-70620d1cef1b" />

(100회 학습)

<img width="184" height="276" alt="image" src="https://github.com/user-attachments/assets/4329a675-a7df-404c-87b6-5859fc8ed1c4" />

(200회 학습)

200회까지 진행결과 가중치 w가 4.2에  편향치 b가 1.6에 수렴함을 알수있음 y  또한 10에 수렴

```
x=2
w=3
b=1
yT=10
lr = 0.01

for epoch in range(200):
    
    y = x*w + 1*b
    E = (y -yT)**2/2
    yE = y-yT
    wE = yE*x
    bE = yE*1
    w -= lr*wE
    b -= lr*bE
    
    
    print(f'epoch = {epoch}')
    print(f' y : {y: .3f}')
    print(f' w :{w: .3f}')
    print(f' b :{b: .3f}')
    
    if E < 0.0000001 :
        break
```
if 문을 추가하여 오차값 E가 매우 작아지는 지점에서 학습을 멈추는 기능 추가 

<img width="156" height="270" alt="image" src="https://github.com/user-attachments/assets/0f3a788b-5d33-445a-b420-e2b40b26bf62" />




<img width="1044" height="231" alt="image" src="https://github.com/user-attachments/assets/d4ca486e-b731-4ed0-9d1a-0a59277b37d7" />

순전파와 역전파의 동작 표현



<img width="658" height="268" alt="image" src="https://github.com/user-attachments/assets/dd965374-7f16-4101-89c1-cca9994c03a4" />

목표값이 27로 변경  x2 추가한 예시


```
x1,x2=2,3
w1,w2=3,4
b=1
yT=27
lr = 0.01

for epoch in range(200):
    
    y = x1*w1 + x2*w2 + 1*b
    E = (y -yT)**2/2
    yE = y-yT
    w1E = yE*x1
    w2E = yE*x2
    bE = yE*1
    w1 -= lr*w1E
    w2 -= lr*w2E
    b -= lr*bE
    
    
    print(f'epoch = {epoch}')
    print(f' y : {y: .3f}')
    print(f' w1 :{w1: .3f}')
    print(f' w2 :{w2: .3f}')
    print(f' b :{b: .3f}')
    
    if E < 0.0000001 :
        break
```

실행결과 

<img width="195" height="253" alt="image" src="https://github.com/user-attachments/assets/76b6415d-5a34-481f-894d-466c07113700" />

실행결과 65번 에포크에서 학습이 종료됨 

2입력 2출력 인공 신경망 학습

<img width="671" height="276" alt="image" src="https://github.com/user-attachments/assets/b87bc526-e2c8-4537-9189-92ced61be139" />


```
x1,x2=2,3
w1,w2=3,4
w3,w4=5,6
b1,b2=1,2
y1T=27
y2T=-30
lr = 0.01

for epoch in range(200):
    
    y1 = x1*w1 + x2*w2 + 1*b1
    y2 = x1*w3 + x2*w4 + 1*b2
    E = ((y1 -y1T)**2 + (y2-y2T)**2)/2 
    y1E = y1-y1T
    y2E = y2-y2T
    w1E = y1E*x1
    w2E = y1E*x2
    w3E = y2E*x1
    w4E = y2E*x2
    b1E = y1E*1
    b2E = y2E*1
    w1 = w1 - lr*w1E
    w2 = w2 - lr*w2E
    w3 = w3 - lr*w3E
    w4 = w4 - lr*w4E
    b1 = b1 - lr*b1E
    b2 = b2 - lr*b2E
    
    
    print(f'epoch = {epoch}')
    print(f' y1 : {y1: .3f}')
    print(f' y2 : {y2: .3f}')
    print(f' w1 :{w1: .3f}')
    print(f' w2 :{w2: .3f}')
    print(f' w3 :{w3: .3f}')
    print(f' w4 :{w4: .3f}')
    print(f' b1 :{b1: .3f}')
    print(f' b2 :{b2: .3f}')
    
    if E < 0.0000001 :
        break
```
<img width="259" height="309" alt="image" src="https://github.com/user-attachments/assets/72174721-026a-494d-9565-57a9bb99fc23" />

실행결과 에포크 79회에서 수렴함을 알수잇음


연습문제 1

<img width="943" height="515" alt="image" src="https://github.com/user-attachments/assets/63eaf83c-105e-4452-82ed-cdfe17d24d85" />

구현해보기

```
x1,x2=0.05,0.10
w1,w2=0.15,0.20
w3,w4=0.25,0.30
w5,w6=0.40,0.55 
b1,b2,b3=0.35,0.45,0.60
y1T=0.01
y2T=0.99
y3T= 0.50
lr = 0.01

for epoch in range(2000):
    
    y1 = x1*w1 + x2*w2 + 1*b1
    y2 = x1*w3 + x2*w4 + 1*b2
    y3 = x1*w5 + x2*w6 + 1*b3
    E = ((y1 -y1T)**2 + (y2-y2T)**2 + (y3-y3T)**2)/2 
    y1E = y1-y1T
    y2E = y2-y2T
    y3E = y3-y3T
    w1E = y1E*x1
    w2E = y1E*x2
    w3E = y2E*x1
    w4E = y2E*x2
    w5E = y3E*x1
    w6E = y3E*x2
    b1E = y1E*1
    b2E = y2E*1
    b3E = y3E*1
    w1 = w1 - lr*w1E
    w2 = w2 - lr*w2E
    w3 = w3 - lr*w3E
    w4 = w4 - lr*w4E
    w5 = w5 - lr*w5E
    w6 = w6 - lr*w6E
    b1 = b1 - lr*b1E
    b2 = b2 - lr*b2E
    b3 = b3 - lr*b3E
    
    print(f'epoch = {epoch}')
    print(f' y1 : {y1: .3f}')
    print(f' y2 : {y2: .3f}')
    print(f' y3 : {y3: .3f}')
    print(f' w1 :{w1: .3f}')
    print(f' w2 :{w2: .3f}')
    print(f' w3 :{w3: .3f}')
    print(f' w4 :{w4: .3f}')
    print(f' w5 :{w5: .3f}')
    print(f' w6 :{w6: .3f}')
    print(f' b1 :{b1: .3f}')
    print(f' b2 :{b2: .3f}')
    print(f' b3 :{b3: .3f}')
    
    if E < 0.0000001 :
        break
```



<img width="288" height="224" alt="image" src="https://github.com/user-attachments/assets/ed7d7b89-0153-4728-9701-b8cf2d62ce48" />

에포크 715에서 수렴함


연습문제 2

<img width="1008" height="671" alt="image" src="https://github.com/user-attachments/assets/37bf2089-0a5b-4a89-8ad8-5dc58d19b1bc" />





```
x1,x2,x3=0.02,0.05,0.12
w1,w2=0.15,0.20
w3,w4=0.02,0.27
w5,w6 = 0.37,0.52
b1,b2=0.12,0.39
y1T=0.02
y2T=0.98
lr = 0.01

for epoch in range(2000):
    
    y1 = x1*w1 + x2*w2 + x3*w3 + 1*b1
    y2 = x1*w4 + x2*w5 + x3*w6 + 1*b2
    E = ((y1 -y1T)**2 + (y2-y2T)**2)/2 
    y1E = y1-y1T
    y2E = y2-y2T
    w1E = y1E*x1
    w2E = y1E*x2
    w3E = y1E*x3
    w4E = y2E*x1
    w5E = y2E*x2
    w6E = y2E*x3
    b1E = y1E*1
    b2E = y2E*1
    w1 = w1 - lr*w1E
    w2 = w2 - lr*w2E
    w3 = w3 - lr*w3E
    w4 = w4 - lr*w4E
    w5 = w5 - lr*w5E
    w6 = w6 - lr*w6E
    b1 = b1 - lr*b1E
    b2 = b2 - lr*b2E
    
    
    print(f'epoch = {epoch}')
    print(f' y1 : {y1: .3f}')
    print(f' y2 : {y2: .3f}')
    print(f' w1 :{w1: .3f}')
    print(f' w2 :{w2: .3f}')
    print(f' w3 :{w3: .3f}')
    print(f' w4 :{w4: .3f}')
    print(f' w5 :{w5: .3f}')
    print(f' w6 :{w6: .3f}')
    print(f' b1 :{b1: .3f}')
    print(f' b2 :{b2: .3f}')
    
    if E < 0.0000001 :
        break
```




<img width="183" height="188" alt="image" src="https://github.com/user-attachments/assets/c3166d0e-e1b4-4138-b967-2b3749559646" />

에포크 690번에서 수렴함

