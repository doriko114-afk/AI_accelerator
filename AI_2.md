---
# [SoC 과정] TensorFlow를 활용한 딥러닝 모델 구현 및 하드웨어 분석

## 1. 개요
이전 스터디에서 파이썬으로 직접 구현했던 순전파/역전파 알고리즘을, 산업 표준 프레임워크인 **TensorFlow (Keras)**를 사용하여 재구현했습니다.
이 과정에서 텐서(Tensor)의 차원(Shape)이 하드웨어 메모리 구조에 어떻게 매핑되는지, 그리고 다양한 활성화 함수(Activation Function)가 하드웨어 설계 관점에서 어떤 비용(Cost)을 가지는지 분석했습니다.

---

## 2. TensorFlow를 활용한 신경망 구현
### 2.1 1입력 1출력 단순 신경망 (Single Neuron)
가장 기본적인 $y=wx+b$ 구조를 텐서플로우 레이어로 구현하며 프레임워크의 동작 방식을 익혔습니다.

*   **Input Shape `(1,)`:** 하드웨어 관점에서 **입력 데이터 버스의 폭(Width)**이 1이라는 의미입니다. 1차원 벡터 형태의 스칼라 값을 입력으로 받습니다.
*   **Dense Layer:** 완전 연결 계층(Fully Connected Layer)을 의미하며, 내부적으로 가중치 곱셈과 편향 덧셈(**MAC 연산**)을 수행합니다.
*   **`model.fit()`:** 학습 루프(Epoch Loop)를 실행하는 함수입니다. 하드웨어의 **Control Unit**이 반복적으로 연산 유닛에 데이터를 공급하고, 계산된 기울기(Gradient)를 바탕으로 레지스터(Weight)를 갱신하는 과정을 추상화한 것입니다.

![image](https://github.com/user-attachments/assets/165fbbda-175e-4056-aacc-dc3011ffbf78)

```python
import tensorflow as tf
import numpy as np

# 데이터셋 준비
X = np.array([[2]])
YT = np.array([[10]])  # Target
W = np.array([[3]])    # 초기 가중치
B = np.array([1])      # 초기 편향

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)), # 입력 차원 정의
    tf.keras.layers.Dense(1)    # 출력 노드 1개
])

# 가중치 강제 설정 (실험용)
model.layers[0].set_weights([W, B])

# 컴파일: 최적화 도구(SGD)와 손실 함수(MSE) 설정
model.compile(optimizer='sgd', loss='mse')

# 학습 진행 (999 Epoch)
model.fit(X, YT, epochs=999)

# 결과 확인
print('Learned W:', model.layers[0].get_weights()[0])
print('Learned B:', model.layers[0].get_weights()[1])
```

### 2.2 다입력 다출력 모델 (MIMO)
입력과 출력이 늘어남에 따라 행렬의 차원이 확장되는 구조입니다.

*   **2입력 1출력:** 입력 핀이 2개로 늘어나고, 내부적으로 2번의 곱셈 후 합산(Reduce Sum)이 일어납니다.
*   **2입력 2출력:** $(2 \times 2)$ 가중치 행렬이 필요하며, 병렬 연산 구조(Systolic Array 등)의 효율성이 중요해지는 시점입니다.
*   **`verbose=0`:** 불필요한 로그 출력을 생략하여 학습 속도를 높입니다.

![image](https://github.com/user-attachments/assets/a01ff740-c3de-496b-a9f1-39d555cf0861)
![image](https://github.com/user-attachments/assets/1e6600c4-f25f-4d3d-a8f1-aad9d5d6f848)

---

### 2.3 은닉층(Hidden Layer)의 도입
단순 선형 연산만으로는 풀 수 없는 복잡한 문제(XOR 등)를 해결하기 위해 **은닉층**을 추가했습니다.

*   **역할:** 입력 데이터를 고차원 공간으로 매핑하거나 비선형적 특징(Feature)을 추출합니다.
*   **HW 관점:** 입력층의 연산 결과를 바로 출력하지 않고, **중간 버퍼(Intermediate Buffer / SRAM)**에 저장했다가 다음 레이어의 입력으로 다시 사용해야 합니다. 레이어가 깊어질수록 온칩 메모리 사용량과 지연 시간(Latency)이 증가합니다.

![image](https://github.com/user-attachments/assets/9a94b9d0-a875-4674-9850-bed97c2234b7)

---

## 3. 학습 과정 시각화 (Visualization)
하드웨어가 최적해를 찾아가는 과정을 시각적으로 분석하기 위해 Matplotlib을 활용했습니다.

### 3.1 Loss Surface & Trajectory
$w$와 $b$의 변화에 따른 오차($E$) 표면을 3D로 시각화했습니다. 초기값(Initialization)에 따라 학습 수렴 속도가 달라짐을 확인했습니다.

![image](https://github.com/user-attachments/assets/e317c88a-0903-4139-9514-09b2d58df363)
![image](https://github.com/user-attachments/assets/664b1ce0-7369-492b-8aeb-b526ecc2d649)

---

## 4. 활성화 함수 (Activation Function) 하드웨어 분석
신경망에 비선형성을 부여하는 활성화 함수들을 구현하고, 하드웨어 구현 비용(Cost) 관점에서 비교했습니다.

### 4.1 Sigmoid
*   **수식:** $y = \frac{1}{1 + e^{-x}}$
*   **HW 분석:** 지수 연산($e^{-x}$)과 나눗셈이 포함되어 있어 **하드웨어 구현 비용이 매우 큽니다.** 실제 설계 시에는 **LUT(Look-Up Table)** 방식이나 Taylor 급수 근사를 사용하여 면적을 줄입니다.

![image](https://github.com/user-attachments/assets/5264a480-a448-43bc-ac3a-d20c8b230c04)
![image](https://github.com/user-attachments/assets/7a46ce96-ac49-4128-98c9-b978da1f926b)

### 4.2 ReLU (Rectified Linear Unit)
*   **수식:** $y = \max(0, x)$
*   **HW 분석:** 단순히 0과 비교하는 **비교기(Comparator)**와 멀티플렉서(MUX)만 있으면 구현 가능합니다. 하드웨어 면적과 전력 소모가 매우 적어 딥러닝 가속기에 가장 적합한 함수입니다.

![image](https://github.com/user-attachments/assets/f9fb6dbb-5f0e-4d4c-8d6c-f1212d2ee4e7)
![image](https://github.com/user-attachments/assets/b145ea23-5a27-47b0-80df-bb50289e9b5d)

### 4.3 Step Function
*   **특징:** 미분이 불가능하여 역전파 학습용으로는 사용할 수 없으나, 추론 전용 칩에서의 간단한 이진 분류 로직에는 활용 가능합니다.

![image](https://github.com/user-attachments/assets/36e30b6d-a515-433c-b91e-7069ee5a9076)

---

## 5. Softmax와 Cross Entropy (고급 구현)
다중 클래스 분류(Multi-class Classification)를 위해 Softmax 함수와 Cross Entropy 오차 함수를 결합하여 구현했습니다.

### 5.1 수식적 이점과 하드웨어 최적화
Softmax 함수를 단독으로 미분하면 수식이 매우 복잡하지만, **Cross Entropy Loss와 결합하면 역전파 수식이 매우 단순해집니다.**

*   **최종 역전파 오차:** $Prediction - Target$ ($y - t$)
*   **HW 분석:** 복잡한 미분 연산 없이 **단순 뺄셈기(Subtractor)** 하나로 출력층의 오차를 계산할 수 있어 하드웨어 복잡도를 획기적으로 낮출 수 있습니다.

![image](https://github.com/user-attachments/assets/71b7bb8f-c814-447e-a0fc-d9e395febb66)

### 5.2 구현 비교 (Python Scratch vs TensorFlow)
수식을 직접 구현한 코드와 TensorFlow 라이브러리를 사용한 코드를 비교하며, 프레임워크가 내부적으로 `categorical_crossentropy`를 통해 이러한 최적화를 수행함을 확인했습니다.

**[Python Scratch 구현 결과]**
![image](https://github.com/user-attachments/assets/70f7380f-d1dd-4b34-8910-f0d80b91c8bc)

**[TensorFlow 구현 코드]**
```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax') # 출력층 Softmax
])

# Loss 함수로 Categorical Crossentropy 사용
model.compile(optimizer='sgd', loss='categorical_crossentropy')
```

