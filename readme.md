
## 1. 프로젝트 개요

AI 가속기(NPU)를 하드웨어로 설계하기 위해서는 내부에서 구동되는  소프트웨어 알고리즘의 

동작 원리를 정확히 알아야 한다고 판단했습니다.

먼저 파이썬으로 딥러닝의 기초(순전파, 역전파)를 구현해보며 **데이터의 흐름과 연산 과정**을 직접 확인했습니다. 

이를 통해 추후 하드웨어 설계 시 필요한 MAC 연산 구조와 메모리 접근 방식에 대한 기초적인 이해를 다지는 것을 목표로 했습니다.

---

## 2. 경사 하강법 (Gradient Descent) 기초 이해
가장 기본적인 선형 회귀 수식($y=wx+b$)을 코드로 옮겨, 가중치($w$)가 오차($E$)에 의해 어떻게 업데이트되는지 시뮬레이션했습니다.

### 2.1 학습 목표 및 코드
*   **목표:** 입력 2를 넣었을 때 10이 나오도록 가중치 학습시키기
*   **설정:** 학습률(learning rate)은 0.01로 설정하여 발산하지 않고 조금씩 수렴하도록 함

![image](https://github.com/user-attachments/assets/2ae6593a-ef91-483a-9b49-056b236cf8d2)

```python
x = 2
w = 3
b = 1
yT = 10
lr = 0.01

for epoch in range(200):
    # 순전파 (Forward): y = wx + b
    y = x*w + 1*b
    
    # 오차 계산 및 역전파 (Backward)
    E = (y -yT)**2/2
    yE = y-yT
    wE = yE*x
    bE = yE*1
    
    # 가중치 업데이트
    w -= lr*wE
    b -= lr*bE
    
    print(f'epoch = {epoch}, y : {y:.3f}, w :{w:.3f}, b :{b:.3f}')
```

### 2.2 학습 과정 확인
**[초기]** 오차가 커서 정답과 거리가 먼 상태

![image](https://github.com/user-attachments/assets/547395ff-f67a-437f-a540-11b098aead0d)

**[중간]** 100회 반복 시 꽤 근접함

![image](https://github.com/user-attachments/assets/d0016525-fa35-49f6-9937-70620d1cef1b)

**[최종]** 200회 반복 결과 가중치가 특정 값($w=4.2$)으로 수렴하는 것을 확인

![image](https://github.com/user-attachments/assets/4329a675-a7df-404c-87b6-5859fc8ed1c4)

---

## 3. 학습 효율화 (Early Stopping)
하드웨어 관점에서 불필요한 전력 소모와 연산 시간을 줄이는 것은 중요합니다. 오차가 일정 수준 이하로 떨어지면 학습을 멈추는 기능을 추가해 보았습니다.

```python
if E < 0.0000001 :
    break
```

**결과:** 목표 오차에 도달하자 루프가 자동으로 멈춤을 확인했습니다.

![image](https://github.com/user-attachments/assets/0f3a788b-5d33-445a-b420-e2b40b26bf62)

---

## 4. 다층 퍼셉트론 구조로 확장 (MIMO)
실제 딥러닝 모델처럼 입력과 출력이 여러 개인 경우를 구현했습니다. 이 과정에서 **행렬 연산이 왜 필요하고, 하드웨어의 병렬 처리가 왜 중요한지** 이해할 수 있었습니다.

### 4.1 데이터 흐름 파악
입력부터 출력, 다시 역전파로 이어지는 데이터 경로(Data Path)를 도식화했습니다.

![image](https://github.com/user-attachments/assets/d4ca486e-b731-4ed0-9d1a-0a59277b37d7)

### 4.2 2입력 1출력 구현

![image](https://github.com/user-attachments/assets/dd965374-7f16-4101-89c1-cca9994c03a4)

**결과:** 65 epoch 만에 수렴. 입력이 늘어나도 원리는 같음을 확인.

![image](https://github.com/user-attachments/assets/76b6415d-5a34-481f-894d-466c07113700)

### 4.3 2입력 2출력 (병렬 연산 구조)

![image](https://github.com/user-attachments/assets/b87bc526-e2c8-4537-9189-92ced61be139)

두 개의 출력을 계산하기 위해 곱셈과 덧셈 연산량이 2배로 늘어났습니다.

**결과:**

![image](https://github.com/user-attachments/assets/72174721-026a-494d-9565-57a9bb99fc23)

### 4.4 더 복잡한 구조 실습 (연습문제)
노드가 늘어날 때마다 코드가 길어지는 것을 보며, 이를 `for`문으로 처리하면 속도가 느려질 것 같다는 생각이 들었습니다.

**[2 Input - 3 Output]**

![image](https://github.com/user-attachments/assets/63eaf83c-105e-4452-82ed-cdfe17d24d85)
![image](https://github.com/user-attachments/assets/ed7d7b89-0153-4728-9701-b8cf2d62ce48)

**[3 Input - 2 Output (Fully Connected)]**

![image](https://github.com/user-attachments/assets/37bf2089-0a5b-4a89-8ad8-5dc58d19b1bc)
![image](https://github.com/user-attachments/assets/c3166d0e-e1b4-4138-b967-2b3749559646)

---

## 5. 하드웨어 설계 관점에서의 배운 점 (Insights)

이 스터디를 통해 소프트웨어 코드를 하드웨어로 옮길 때 고려해야 할 점들을 정리해보았습니다.

1.  **MAC 연산기(Multiplier-Accumulator)의 필요성**
    *   코드에서 `y = x1*w1 + x2*w2...` 부분은 곱하고 더하는 연산의 반복이었습니다.
    *   입출력 노드가 늘어날수록 연산량이 급격히 많아지기 때문에, CPU처럼 순차적으로 처리하는 것보다 **병렬로 동작하는 MAC 어레이**가 필수적이라는 것을 느꼈습니다.

2.  **메모리 설계의 중요성**
    *   역전파 코드를 짜면서 가장 까다로웠던 점은, 순전파 때 썼던 입력값 $x$를 역전파 때 다시 써야 한다는 점이었습니다.
    *   하드웨어로 구현할 때도 **중간 계산 결과(Activation)를 버리지 않고 저장해둘 내부 버퍼(SRAM)** 용량을 잘 산정해야 병목 현상이 없을 것 같습니다.

3.  **데이터 의존성 (Dependency)**
    *   $E$(오차)를 구하기 전까지는 가중치 $w$를 업데이트할 수 없었습니다.
    *   이는 하드웨어 파이프라인 설계 시 **앞 단계가 끝날 때까지 기다려야 하는 상황(Stall)**이 발생할 수 있음을 의미하므로, 이를 최적화하는 스케줄링이 중요하다고 생각했습니다.
---
