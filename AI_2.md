
Tensorflow를 활용하여 딥러닝 7 공식 구현

```
pip3 install matplotlib
```
matplotlib 라이브러리 설치

1입력 1출력 인공 신경 

<img width="866" height="252" alt="image" src="https://github.com/user-attachments/assets/59f78194-2857-44d8-8be9-3b76d07615c6" />



코드상에서 shape 의 경우 행렬의 차원 역할 이는 추후 좀더 보완하여 정리

model fit또한 추가설명 및 보완 정리 

```
import tensorflow as tf
import numpy as np

X=np.array([[2]])
YT = np.array([[10]])
W= np.array([[3]])
B=np.array([1])

model=tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.layers[0].set_weights([W,B])

model.compile(optimizer='sgd',loss='mse')

Y=model.predict(X)
print(Y)

model.fit(X,YT,epochs=999)

print('W=',model.layers[0].get_weights()[0])
print('B=',model.layers[0].get_weights()[1])

Y=model.predict(X)
print(Y)

```
텐서플로우를 활용한 코드 

결과

<img width="698" height="286" alt="image" src="https://github.com/user-attachments/assets/165fbbda-175e-4056-aacc-dc3011ffbf78" />


2입력 1출력 인공 신경 구현

<img width="1009" height="314" alt="image" src="https://github.com/user-attachments/assets/6cc5cbc0-08c6-4945-97a4-fc21c3ba0329" />


입력을 2개로 늘려줌

```

import tensorflow as tf
import numpy as np

X=np.array([[2,3]]) #input data
YT=np.array([[27]]) #Target data(label)
W=np.array([[3],[4]])
B=np.array([1])

model=tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(1)
])


model.layers[0].set_weights([W,B])

model.compile(optimizer='sgd',loss='mse')

Y=model.predict(X)
print(Y)

model.fit(X,YT,epochs=999,verbose=0)

print('W=',model.layers[0].get_weights()[0])
print('B=',model.layers[0].get_weights()[1])

Y=model.predict(X)
print(Y)
```

결과

<img width="485" height="90" alt="image" src="https://github.com/user-attachments/assets/a01ff740-c3de-496b-a9f1-39d555cf0861" />



```
verbose=0
```
verbose를 통해 중간 과정을 생략할수있음


2입력 2출력 인공 신경망

<img width="1077" height="311" alt="image" src="https://github.com/user-attachments/assets/55229089-4b0b-4945-bfb7-cad7abada12d" />


결과

<img width="506" height="132" alt="image" src="https://github.com/user-attachments/assets/1e6600c4-f25f-4d3d-a8f1-aad9d5d6f848" />




2입력 2은닉 2출력 인공 신경망

<img width="881" height="355" alt="image" src="https://github.com/user-attachments/assets/641b771b-2835-4e7f-a808-46c1c260230c" />

결과 

<img width="496" height="144" alt="image" src="https://github.com/user-attachments/assets/9a94b9d0-a875-4674-9850-bed97c2234b7" />



딥러닝 학습과정 시각화

```
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection='3d')
ax.set_title("wbE",size = 20)

ax.set_xlabel("w",size=14)
ax.set_ylabel("b",size=14)
ax.set_zlabel("E",size=14)

x=2
yT=10

w= np.random.uniform(-200,200,10000)
b= np.random.uniform(-200,200,10000)

y=x*w+b
E=(y-yT)**2/2



ax.plot(w,b,E,'g')
plt.show()

```
결과 

<img width="712" height="714" alt="image" src="https://github.com/user-attachments/assets/e317c88a-0903-4139-9514-09b2d58df363" />

위의 코드를 통해 w,b,E의 관계를 살필수 있음 


애니메이션 기능 추가


```
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize= (8,8))
ax =  fig.add_subplot(projection = '3d')
ax.set_title("wbE",size=20)

ax.set_xlabel("w", size = 14)
ax.set_ylabel("b", size = 14)
ax.set_zlabel("E", size = 14)

x=2
yT=10

w= np.random.uniform(2,7,10000)
b= np.random.uniform(0,4,10000)


y=x*w+b
E=(y-yT)**2/2

ax.plot(w,b,E,'g')

x=2
w=3
b=1
yT= 10
lr = 0.01

wbEs = []
EPOCHS = 200

for epoch in range(EPOCHS):
        y = x*w +  1*b
        E = (y-yT)**2 /2
        yE = y-yT
        wE = yE*x
        bE = yE*1
        w  -= lr*wE
        b  -= lr*bE
        
        wbEs.append(np.array([w,b,E]))


data = np.array(wbEs).T
line, = ax.plot([],[],[],'r.')

def animate(epoch, data, line):
        print(epoch, data[2, epoch])
        line.set_data(data[:2, :epoch])
        line.set_3d_properties(data[2, :epoch])
        
from matplotlib.animation import FuncAnimation

ani = FuncAnimation(fig, animate, EPOCHS, fargs=(data,line), interval= 20000/EPOCHS)

plt.show()


```

결과 

<img width="731" height="700" alt="image" src="https://github.com/user-attachments/assets/664b1ce0-7369-492b-8aeb-b526ecc2d649" />


<img width="692" height="634" alt="image" src="https://github.com/user-attachments/assets/02d22a07-c08b-4436-ae2a-9c059e300451" />

w,b 값을 조정하는것에 따라 다른 지점에서 시작함을 알수있음 


<img width="703" height="620" alt="image" src="https://github.com/user-attachments/assets/f99b7f91-eeb9-4449-89c2-b1db615f2793" />

코드를 수정하여 애니메이션을 다른 모습으로도 수정 가능

활성화함수

좀더 복잡한 딥러닝을 구현

7 segment, 손글씨, FASHION MNIST, CIFAR10, CIFAR100 순으로 구현 예정


1.시그모이드

<img width="996" height="211" alt="image" src="https://github.com/user-attachments/assets/5264a480-a448-43bc-ac3a-d20c8b230c04" />

시그모이드 함수의 그래프와 수식 


2.RelU

<img width="1069" height="247" alt="image" src="https://github.com/user-attachments/assets/f9fb6dbb-5f0e-4d4c-8d6c-f1212d2ee4e7" />

RelU 함수의 그래프와 수식

은닉층에서 자주 사용됨

3.소프트맥스

<img width="565" height="274" alt="image" src="https://github.com/user-attachments/assets/d5db2156-d230-40c2-aae1-31d63e70b64b" />


소프트맥스 함수의 모습

출력단 전용 함수



활성화 함수 그려보기

시그모이드
```
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    
x= np.random.uniform(-10,10,1000)
y=sigmoid(x)

plt.plot(x,y,'r.')
plt.show()

```

결과

<img width="599" height="444" alt="image" src="https://github.com/user-attachments/assets/7a46ce96-ac49-4128-98c9-b978da1f926b" />


ReLU


```
import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
        return x*(x>0)
    
x = np.random.uniform(-10,10,1000)
y=ReLU(x)

plt.plot(x,y,'g.')
plt.show()

```

결과

<img width="583" height="440" alt="image" src="https://github.com/user-attachments/assets/b145ea23-5a27-47b0-80df-bb50289e9b5d" />



계단 함수

```
import numpy as np
import matplotlib.pyplot as plt

def step(x):
        return x>0
    
x=np.random.uniform(-10,10,1000)
y=step(x)

plt.plot(x,y,'b.')
plt.show()

```

결과

<img width="585" height="429" alt="image" src="https://github.com/user-attachments/assets/36e30b6d-a515-433c-b91e-7069ee5a9076" />



계단함수 구조 애니메이션

![Video Project](https://github.com/user-attachments/assets/46faea12-49f5-47b4-a2a6-0ef62074b894)


활성화 함수 적용

sigmoid ReLU 활성함수 적용

```

from math import exp

x1,x2 = 0.05,0.10
w1,w2 = 0.15,0.20
w3, w4= 0.25,0.30
b1,b2 = 0.35,0.35

w5,w6 = 0.40,0.45
w7,w8 = 0.50,0.55
b3,b4 = 0.60,0.60

y1T, y2T = 0.01,0.99

lr = 0.01

EPOCH = 1000

for epoch in range(EPOCH):
        
        h1 = x1*w1  + x2*w2 + 1*b1
        h2 = x1*w3  + x2*w4 + 1*b2
        
        h1 = h1 if h1>0 else 0
        h2 = h2 if h2>0 else 0
        
        y1 = h1*w5 + h2*w6 + 1*b3
        y2 = h1*w7 + h2*w8 + 1*b4
        
        y1=1/(1+exp(-y1))
        y2=1/(1+exp(-y2))
        
        E = (y1-y1T)**2/2 + (y2-y2T)**2/2
        
        y1E = y1 -y1T
        y2E = y2- y2T
        
        y1E = y1*(1-y1)*y1E
        y2E = y2*(1-y2)*y2E
        
        w5E = y1E*h1
        w6E = y1E*h2
        w7E = y2E*h1
        w8E = y2E*h2
        b3E = y1E*1
        b4E = y2E*1
        
        h1E = y1E*w5 + y2E*w7
        h2E = y1E*w6 + y2E*w8
        
        h1E = h1E if h1>0 else 0
        h2E = h2E if h2>0 else 0
        
        w1E = h1E*x1
        w2E = h1E*x2
        w3E = h2E*x1
        w4E = h2E*x2
        b1E = h1E*1
        b2E = h2E*1
        
        w5 -= lr*w5E
        w6 -= lr*w6E
        w7 -= lr*w7E
        w8 -= lr*w8E
        b3 -= lr*b3E
        b4 -= lr*b4E
        
        w1 -= lr*w1E
        w2 -= lr*w2E
        w3 -= lr*w3E
        w4 -= lr*w4E
        b1 -= lr*b1E
        b2 -= lr*b2E
        
        
        if epoch % 100 == 99:
                print(f'epcoh =  {epoch}')
                print(f' y1 :  {y1: .6f}')
                print(f' y2 :  {y2: .6f}')
                
        if E<0.0000001:
                break
            
print(f'w1.w3 = {w1:.6f},{w3:.6f}')
print(f'w2.w4 = {w2:.6f},{w4:.6f}')
print(f'b1.b2 = {b1:.6f},{b2:.6f}')
print(f'w5.w7 = {w5:.6f},{w7:.6f}')
print(f'w6.w8 = {w6:.6f},{w8:.6f}')
print(f'b3.b4 = {b3:.6f},{b4:.6f}')



```

결과

<img width="281" height="158" alt="image" src="https://github.com/user-attachments/assets/5cbe9333-2b2b-4f20-b736-ccea7d336a25" />

텐서플로에 적용

```
import tensorflow as tf
import numpy as np

X  = np.array([[.05,.10]]) # input data
YT = np.array([[.01,.99]]) # target data
W  = np.array([[.15,.25],[.20,.30]])
B  = np.array([.35,.35])
W2 = np.array([[.40,.50],[.45,.55]])
B2 = np.array([.60,.60])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
        
])

model.layers[0].set_weights([W,B])
model.layers[1].set_weights([W2,B2])

model.compile(optimizer = 'sgd',loss = 'mse')

Y=model.predict(X)
print(Y)

model.fit(X,YT,epochs=999)

print('W=',model.layers[0].get_weights()[0])
print('B=',model.layers[0].get_weights()[1])
print('W2=',model.layers[1].get_weights()[0])
print('B2=',model.layers[1].get_weights()[1])

Y= model.predict(X)
print(Y)

```

결과

<img width="505" height="147" alt="image" src="https://github.com/user-attachments/assets/71a07d15-0278-4db8-b7af-33ba241a6e45" />


출력층에 linear 함수 적용

```
import tensorflow as tf
import numpy as np

X  = np.array([[.05,.10]]) # input data
YT = np.array([[.01,.99]]) # target data
W  = np.array([[.15,.25],[.20,.30]])
B  = np.array([.35,.35])
W2 = np.array([[.40,.50],[.45,.55]])
B2 = np.array([.60,.60])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
        
])

model.layers[0].set_weights([W,B])
model.layers[1].set_weights([W2,B2])

model.compile(optimizer = 'sgd',loss = 'mse')

Y=model.predict(X)
print(Y)

model.fit(X,YT,epochs=599)

print('W=',model.layers[0].get_weights()[0])
print('B=',model.layers[0].get_weights()[1])
print('W2=',model.layers[1].get_weights()[0])
print('B2=',model.layers[1].get_weights()[1])

Y= model.predict(X)
print(Y)



```
결과 

<img width="603" height="141" alt="image" src="https://github.com/user-attachments/assets/30e9d0d5-2798-4b65-8a54-c2946e788678" />


softmax 활성함수 cross entropy 오차함수 

softmax 함수의 경우 크로스 엔트로피 함수와 같이 사용될때만 역전파 시 softmax 함수를 역으로 거쳐 전파되는 오차가 
다음과 같이 예측값과 목표값의 차가됨

공식

<img width="322" height="107" alt="image" src="https://github.com/user-attachments/assets/71b7bb8f-c814-447e-a0fc-d9e395febb66" />



softmax 활성화 함수 cross entropy 오차 함수 적용하기


```
from math import exp,log

x1,x2 = 0.05,0.10
w1,w2 = 0.15,0.20
w3, w4= 0.25,0.30
b1,b2 = 0.35,0.35

w5,w6 = 0.40,0.45
w7,w8 = 0.50,0.55
b3,b4 = 0.60,0.60

y1T, y2T = 0.,1.

lr = 0.01

EPOCH = 10000

for epoch in range(EPOCH):
        
        h1 = x1*w1  + x2*w2 + 1*b1
        h2 = x1*w3  + x2*w4 + 1*b2
        
        h1 = h1 if h1>0 else 0
        h2 = h2 if h2>0 else 0
        
        y1 = h1*w5 + h2*w6 + 1*b3
        y2 = h1*w7 + h2*w8 + 1*b4
        
        #softmax feed forward
        yMax= y1 if y1>y2 else y2
        y1 -=yMax
        y2 -=yMax
        sumY = exp(y1)+exp(y2)
        
        
        y1 = exp(y1)/sumY
        y2 = exp(y2)/sumY
        
        E =  -(y1T*log(y1)+y2T*log(y2))
        
        
        y1E = y1 -y1T
        y2E = y2- y2T
        
        
        w5E = y1E*h1
        w6E = y1E*h2
        w7E = y2E*h1
        w8E = y2E*h2
        b3E = y1E*1
        b4E = y2E*1
        
        h1E = y1E*w5 + y2E*w7
        h2E = y1E*w6 + y2E*w8
        
        h1E = h1E if h1>0 else 0
        h2E = h2E if h2>0 else 0
        
        w1E = h1E*x1
        w2E = h1E*x2
        w3E = h2E*x1
        w4E = h2E*x2
        b1E = h1E*1
        b2E = h2E*1
        
        w5 -= lr*w5E
        w6 -= lr*w6E
        w7 -= lr*w7E
        w8 -= lr*w8E
        b3 -= lr*b3E
        b4 -= lr*b4E
        
        w1 -= lr*w1E
        w2 -= lr*w2E
        w3 -= lr*w3E
        w4 -= lr*w4E
        b1 -= lr*b1E
        b2 -= lr*b2E
        
        
        if epoch % 100 == 99:
                print(f'epcoh =  {epoch}')
                print(f' y1 :  {y1: .6f}')
                print(f' y2 :  {y2: .6f}')
                
        if E<0.0000001:
                break
            
print(f'w1.w3 = {w1:.6f},{w3:.6f}')
print(f'w2.w4 = {w2:.6f},{w4:.6f}')
print(f'b1.b2 = {b1:.6f},{b2:.6f}')
print(f'w5.w7 = {w5:.6f},{w7:.6f}')
print(f'w6.w8 = {w6:.6f},{w8:.6f}')
print(f'b3.b4 = {b3:.6f},{b4:.6f}')


```

결과 

<img width="291" height="160" alt="image" src="https://github.com/user-attachments/assets/70f7380f-d1dd-4b34-8910-f0d80b91c8bc" />


텐서플로우의 경우

```
import tensorflow as tf
import numpy as np

X  = np.array([[.05,.10]]) # input data
YT = np.array([[0.,1.]]) # target data
W  = np.array([[.15,.25],[.20,.30]])
B  = np.array([.35,.35])
W2 = np.array([[.40,.50],[.45,.55]])
B2 = np.array([.60,.60])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
        
])

model.layers[0].set_weights([W,B])
model.layers[1].set_weights([W2,B2])

model.compile(optimizer = 'sgd',loss = 'categorical_crossentropy')

Y=model.predict(X)
print(Y)

model.fit(X,YT,epochs=9999)

print('W=',model.layers[0].get_weights()[0])
print('B=',model.layers[0].get_weights()[1])
print('W2=',model.layers[1].get_weights()[0])
print('B2=',model.layers[1].get_weights()[1])

Y= model.predict(X)
print(Y)



```

결과




