
---
# [SoC 과정] Advanced CNN Architectures & Optimization Techniques

## 1. 개요
기초적인 CNN 구조를 넘어, 자율주행 및 이미지 분류 분야에서 검증된 심층 신경망(Deep Neural Networks) 모델인 **NVIDIA Autopilot CNN**과 **AlexNet**을 구현했습니다.
또한, 신경망이 깊어질수록 발생하는 학습 불안정 문제를 해결하기 위한 **Batch Normalization**과 **Dropout** 기법을 적용해보고, 이를 하드웨어 가속기 설계 관점에서 해석했습니다.

---

## 2. Feature Map 시각화
합성곱(Convolution) 연산 후 생성되는 다양한 특징 맵(Feature Map)을 시각화하여, 필터들이 이미지의 어떤 특징(Edge, Texture 등)을 추출하는지 확인했습니다.
하드웨어 관점에서는, 이 **24개의 이미지를 동시에 생성하기 위해 24개의 병렬 연산 유닛(Processing Element)**이 필요함을 의미합니다.

![image](https://github.com/user-attachments/assets/597fffb0-18ab-471b-a7db-4ab958d6e4ea)

```python
for i in range(24):
    plt.subplot(4,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i], cmap=plt.cm.gray)
plt.show()
```

---

## 3. NVIDIA 자율주행 CNN 구현
NVIDIA에서 자율주행 자동차의 스티어링 휠 제어를 위해 제안한 CNN 모델을 CIFAR-10 데이터셋에 맞게 수정하여 구현했습니다.

### 3.1 아키텍처 특징 및 HW 분석
*   **Deep Convolution:** 5개의 Conv Layer를 연속으로 배치하여 매우 추상화된 특징을 추출합니다.
*   **HW 관점:** 레이어가 깊어질수록 **Latency(지연 시간)**가 증가합니다. 실시간 자율주행을 위해서는 각 레이어 간의 데이터를 즉시 넘겨주는 **Pipelining** 구조 설계가 필수적입니다.

![image](https://github.com/user-attachments/assets/ac189e08-0594-49c4-8e1c-074bf5ccc03c)

```python
import tensorflow as tf
mnist = tf.keras.datasets.cifar10
(X,YT),(x,yt) = mnist.load_data()
X, x = X/255, x/255

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32,32,3)),
    # Feature Extraction
    tf.keras.layers.Conv2D(24,(3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(36,(3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(48,(3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64,(3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64,(3,3), padding='same', activation='relu'),
    # Classification
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
# ... (Compile & Fit 생략)
```
**결과:** 5 Epoch 학습으로 기본적인 정확도 확보.
![image](https://github.com/user-attachments/assets/4d10408a-c2d0-4e41-92b5-110dd6960603)

---

## 4. AlexNet 구현 (CNN의 시초)
딥러닝 붐을 일으킨 AlexNet 구조를 CIFAR-10 이미지 크기(32x32)에 맞춰 축소 구현했습니다.

### 4.1 아키텍처 특징 및 HW 분석
*   **Large Kernels:** 원본은 11x11 필터를 사용하지만, 여기서는 5x5를 사용했습니다. 3x3 필터보다 **Line Buffer의 크기가 더 커야 함**을 의미합니다.
*   **Deep & Wide:** 파라미터 수가 많아 **메모리 대역폭(Memory Bandwidth)**이 성능의 병목이 될 가능성이 높습니다.

![image](https://github.com/user-attachments/assets/f3fb4dad-6b7f-4c4d-869e-5b07a2cff54d)

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32,32,3)),
    tf.keras.layers.Conv2D(96,(5,5), activation='relu'),
    tf.keras.layers.MaxPooling2D((3,3),(2,2)),
    # ... (중략: 복잡한 Conv-Pool 구조) ...
    tf.keras.layers.Dense(496, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
```
**결과:** 복잡한 모델임에도 안정적으로 학습됨.
![image](https://github.com/user-attachments/assets/d9b0b026-d7bd-4b4c-ae02-3da97568b6ae)

---

## 5. 신경망 최적화 기법과 하드웨어 구현
딥러닝 모델의 성능을 높이기 위해 사용되는 기법들이 하드웨어 설계에는 어떤 영향을 미치는지 분석했습니다.

### 5.1 배치 정규화 (Batch Normalization)
각 레이어의 입력을 정규화(평균 0, 분산 1)하여 학습 속도를 높이고 초기값 민감도를 낮추는 기법입니다.

*   **SW 원리:** $\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$
*   **HW 분석:**
    *   **학습(Training) 시:** 평균과 분산을 구하기 위한 나눗셈/제곱근 연산기가 필요하여 하드웨어 비용이 매우 큽니다.
    *   **추론(Inference) 시:** 학습이 끝난 후에는 상수값으로 고정되므로, **가중치(Weight)와 편향(Bias)에 미리 융합(Folding)**시켜 하드웨어 추가 비용 없이 구현 가능합니다. (SoC 설계의 핵심 테크닉)

![image](https://github.com/user-attachments/assets/cd0955ca-14e3-4a2f-a5d1-e36e61e4f54e)

### 5.2 드롭아웃 (Dropout)
학습 과정에서 무작위로 일부 뉴런을 비활성화(0으로 만듦)하여 과적합(Overfitting)을 방지하는 기법입니다.

*   **SW 원리:** 지정된 확률(예: 50%)로 뉴런의 출력을 0으로 마스킹(Masking).
*   **HW 분석:**
    *   **학습용 칩:** 난수 생성기(PRNG, LFSR 등)가 필요합니다.
    *   **추론용 칩:** 모든 뉴런을 사용하되 출력값을 스케일링하거나, Dropout을 아예 제거(Identity)하므로 별도의 하드웨어 로직이 필요 없습니다.

![image](https://github.com/user-attachments/assets/baeda6d6-d0df-4fda-b0ef-5926e75db35d)

---

## 6. 결론 (Insights)
1.  **모델의 깊이와 파이프라인:** NVIDIA CNN과 같이 깊은 모델을 처리하려면, 각 레이어의 연산 속도를 균일하게 맞춰 파이프라인 효율(Utilization)을 높이는 것이 중요합니다.
2.  **커널 크기와 버퍼:** AlexNet과 같이 큰 커널(5x5 이상)을 사용할 경우, On-chip SRAM (Line Buffer) 요구량이 급증하므로 메모리 설계에 유의해야 합니다.
3.  **Inference Optimization:** Batch Norm과 Dropout은 학습 시에는 복잡하지만, 추론 가속기 설계 시에는 **수식 정리(Folding) 및 제거**를 통해 하드웨어 오버헤드를 최소화할 수 있습니다.
---
