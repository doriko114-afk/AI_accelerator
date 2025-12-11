

***

---
# [SoC 과정] 다양한 데이터셋을 활용한 DNN 구현 및 하드웨어적 한계 분석

## 1. 개요
본 프로젝트는 텐서플로우(TensorFlow)를 활용하여 논리 회로(7-Segment)부터 이미지 처리(MNIST, CIFAR)까지 다양한 데이터셋을 학습시켜 봅니다.
이를 통해 **데이터 차원 변환(Reshape)**, **국소해(Local Minima)** 문제, 그리고 이미지 처리 시 **완전 연결 계층(Dense Layer)의 한계점**을 분석하여, 추후 CNN(Convolutional Neural Network) 가속기 설계의 필요성을 도출합니다.

---

## 2. 7-Segment Display (논리 회로 근사)
디지털 회로에서 흔히 쓰이는 7-세그먼트 디코더의 진리표를 신경망으로 학습시켰습니다.

### 2.1 데이터 및 초기 모델
*   **Input:** 7개의 세그먼트 신호 (a, b, c, d, e, f, g)
*   **Target:** 4비트 2진수 (BCD Code)
*   **문제 발생:** 초기 모델(은닉층 8개, Sigmoid)에서 학습이 제대로 되지 않고 오차가 줄어들지 않는 현상 발생.

![image](https://github.com/user-attachments/assets/8400626c-2999-4b2b-8f02-56a99b0e60fb)

### 2.2 국소해 (Local Minima) 문제와 해결
*   **정의:** 오차 함수(Loss Function)의 그래프에서 전역 최소값(Global Minimum)이 아닌, 주변보다 약간 낮은 골짜기(Local Minimum)에 갇혀 가중치 업데이트가 멈추는 현상.
*   **해결:**
    1.  **모델 용량(Capacity) 증대:** 은닉층 노드 수를 8개 $\to$ 16개로 증가.
    2.  **활성화 함수 변경:** 출력층을 `Sigmoid` $\to$ `Linear`로 변경하여 기울기 소실(Vanishing Gradient) 완화.

![image](https://github.com/user-attachments/assets/fe6ec071-4932-4c7d-a4f2-86e2b8cc5243)
![image](https://github.com/user-attachments/assets/71e77439-f78a-4729-99b0-d05e6fb4c343)
> **Result:** 오차가 획기적으로 줄어들며 정상적으로 학습됨.

### 2.3 One-Hot Encoding 및 역방향 학습
*   **One-Hot Encoding:** 10진수 출력을 위해 출력 노드를 1개(Linear)가 아닌, 10개(각 숫자별 확률)로 확장하거나 매핑 방식을 변경.
*   **Inverse Mapping:** 입력과 출력을 바꿔서(숫자 $\to$ 세그먼트 점등 패턴) 학습 진행. 인공신경망이 **Encoder와 Decoder 역할을 모두 수행**할 수 있음을 확인.

![image](https://github.com/user-attachments/assets/faa94b90-a56d-4fc0-9895-68055084761f)

---

## 3. MNIST 손글씨 인식 (Image Processing 기초)
$28 \times 28$ 크기의 흑백 손글씨 이미지를 분류하는 모델입니다.

### 3.1 데이터 전처리 (Reshape & Normalization)
하드웨어(MLP 가속기)에 2차원 이미지를 입력하기 위해 데이터 형태를 변환해야 합니다.

*   **Normalization:** $0 \sim 255$의 픽셀 값을 $0 \sim 1$ 실수로 변환 (연산 오버플로우 방지 및 수렴 속도 향상).
*   **Reshape (`28x28` $\to$ `784`):**
    *   **SW 관점:** 2차원 배열을 1차원 벡터로 평탄화(Flatten).
    *   **HW 관점:** 메모리에 저장된 이미지 데이터를 순차적인 주소(Sequential Address)로 읽어와 **입력 버퍼에 직렬로 공급**하는 과정. 이 과정에서 **공간적 지역성(Spatial Locality) 정보가 손실**되는 단점이 있음.

![image](https://github.com/user-attachments/assets/14e958de-3b90-4115-9b92-0246933dca5b)

```python
# 2D 이미지를 1D 벡터로 변환 (Linearization)
X, x = X.reshape((60000, 784)), x.reshape((10000, 784))
```

### 3.2 학습 결과 및 모델 저장
*   **모델 구조:** Input(784) $\to$ Hidden(128, ReLU) $\to$ Output(10, Softmax)
*   **정확도:** 약 97% 이상 달성.
*   **Model Save:** `model.save('model.keras')`를 통해 학습된 가중치(Weight)를 파일로 저장. 이는 하드웨어의 **NVM(Non-Volatile Memory)에 펌웨어를 굽는 과정**과 유사함.

![image](https://github.com/user-attachments/assets/f32a0c92-d080-40d1-8b59-1b240e8947c6)

### 3.3 오류 분석 (Error Analysis)
모델이 예측에 실패한 케이스를 시각화하여 분석했습니다. 육안으로도 구별하기 힘든 악필 데이터에서 주로 오류가 발생함을 확인했습니다.

![image](https://github.com/user-attachments/assets/39a67d55-be4b-4446-8de7-e75db3384419)
![image](https://github.com/user-attachments/assets/0ccbb807-6be5-4f1d-8614-b8099078001f)

---

## 4. 복잡한 데이터셋과 MLP의 한계 (Fashion MNIST, CIFAR)
단순한 숫자 인식을 넘어, 복잡한 패턴과 컬러 이미지를 MLP로 학습시켰을 때의 한계를 분석했습니다.

### 4.1 Fashion MNIST
*   데이터: 의류 10종류 (흑백 28x28)
*   결과: 숫자 MNIST보다 정확도가 하락함. 단순한 형상 매칭보다 더 복잡한 특징 추출이 필요함을 시사.

![image](https://github.com/user-attachments/assets/d7bfbba2-f1e7-4346-b2bf-0c1298885fa6)

### 4.2 CIFAR-10 (Color Image)
*   **데이터:** 비행기, 자동차 등 10종류 (컬러 $32 \times 32 \times 3$ 채널)
*   **Input Dimension:** $32 \times 32 \times 3 = 3072$개의 입력 노드 필요.
*   **문제점:**
    1.  **차원의 저주:** 입력 차원이 커지면서 가중치 파라미터 수가 급증함.
    2.  **정보 손실:** `Reshape` 과정에서 RGB 채널 간의 연관성과 픽셀 간의 위치 정보가 사라짐.
    3.  **결과:** 정확도가 매우 낮음 (약 40~50% 수준).

![image](https://github.com/user-attachments/assets/b14d3422-5919-470e-8357-c07730e39123)
![image](https://github.com/user-attachments/assets/4b4d90c7-0df6-4146-9f81-16f925df3ca3)

### 4.3 CIFAR-100 (High Complexity)
*   **데이터:** 100종류의 사물 클래스.
*   **결과:** MLP 구조로는 거의 분류가 불가능함. (매우 낮은 정확도)

![image](https://github.com/user-attachments/assets/8591f685-f27f-416a-93c9-c42c4c425ee4)
![image](https://github.com/user-attachments/assets/ad9af4a5-28d0-4b8f-85fd-e373d5bbc443)

---

## 5. Noise Data Test (Baseline Check)
랜덤하게 생성한 노이즈 이미지를 학습시켰을 때, 모델이 패턴을 찾지 못하는지 검증했습니다.
*   **결과:** 학습 데이터와 평가 데이터 간의 연관성이 없으므로, 정확도가 찍기 확률(10%) 수준에 머무름을 확인.

![image](https://github.com/user-attachments/assets/8cca4927-ffed-40b2-b17b-be37badfc3a3)

---

## 6. 결론 및 하드웨어 설계 인사이트 (Insights)

1.  **Reshape의 비용과 한계:**
    *   MLP(Dense Layer)를 사용하기 위해 2D 이미지를 1D로 펼치는 과정(`Reshape`)은 하드웨어적으로는 단순한 주소 매핑이지만, 알고리즘적으로는 **공간 정보(위, 아래, 좌, 우 픽셀 관계)를 파괴**합니다.
    *   이로 인해 CIFAR-10 같은 복잡한 이미지 인식률이 현저히 떨어집니다.

2.  **CNN 가속기의 필요성:**
    *   이미지의 공간적 특징을 유지하면서 학습하기 위해서는 **합성곱 신경망(CNN)**이 필수적입니다.
    *   따라서 차기 하드웨어 설계 스터디는 단순 MAC 연산기(Matrix Multiplier)를 넘어, **필터 연산(Sliding Window)과 채널 연산을 지원하는 가속기 구조**로 나아가야 합니다.

3.  **파라미터 메모리 병목:**
    *   CIFAR-10의 경우 입력층만 해도 가중치가 $3072 \times N$개로 급증합니다. MLP는 파라미터 수가 너무 많아 온칩 메모리(SRAM) 용량에 큰 부담을 줍니다. Weight Sharing을 하는 CNN 구조가 하드웨어 리소스 측면에서도 효율적일 것으로 예상됩니다.
---
