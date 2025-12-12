

---
# [SoC 과정] 합성곱 신경망(CNN)의 연산 구조와 하드웨어 분석

## 1. 개요
이전 스터디에서 확인한 MLP(완전 연결 계층)의 한계(공간 정보 손실, 파라미터 폭증)를 극복하기 위해 **합성곱 신경망(Convolutional Neural Network)**을 학습하고 구현했습니다.
이미지 처리의 핵심인 필터 연산(Convolution), 풀링(Pooling), 그리고 채널(Channel)의 개념을 이해하고, 이를 하드웨어 가속기(Accelerator) 관점에서 해석했습니다.

---

## 2. CNN의 핵심 연산 구조
### 2.1 합성곱 연산 (Convolution)
입력 이미지에 작은 크기의 필터(Kernel)를 슬라이딩하며 곱셈-합(MAC) 연산을 수행합니다.
*   **SW 관점:** 이미지의 특징(Feature)을 추출하는 과정.
*   **HW 관점:** **3x3 MAC Array**가 이동하며 연산을 수행. MLP와 달리 **가중치(필터) 공유(Weight Sharing)**를 통해 메모리 사용량을 획기적으로 줄일 수 있습니다.

![image](https://github.com/user-attachments/assets/793b3656-20ff-4efb-baaa-728f54121650)

```python
# 3x3 Convolution 연산 예시
image_x_filter = image * filter
convolution = np.sum(image_x_filter)
```

### 2.2 Stride와 Padding
*   **Stride (보폭):** 필터가 이동하는 간격. Stride가 클수록 출력 데이터(Feature Map)의 크기가 줄어듭니다.
*   **Padding:** 이미지 가장자리의 정보 손실을 막고 출력 크기를 유지하기 위해 외곽에 0을 채우는 기법.
*   **HW 관점:**
    *   **Padding:** 하드웨어 입력 버퍼 제어 로직에서 경계 검사(Boundary Check)를 통해 0을 주입하는 **Zero Padding Logic**이 필요합니다.
    *   **Stride:** 메모리 주소 생성기(Address Generator)의 증가폭(Offset)을 조절하여 구현합니다.

![image](https://github.com/user-attachments/assets/62549f26-bcef-409d-bb7c-a44b77e3c376)
![image](https://github.com/user-attachments/assets/14294ebf-3c82-4596-8930-d16c44b9a67b)

### 2.3 Pooling (Sub-sampling)
*   **역할:** 중요 정보(최대값 등)만 남기고 데이터 크기를 줄여 연산량을 감소시키고, 위치 변화에 대한 불변성(Invariance)을 확보합니다.
*   **Max Pooling:** 윈도우 내에서 가장 큰 값을 선택.
*   **HW 관점:** 비교기(Comparator)와 레지스터만으로 구현 가능한 **저비용 연산**입니다.

![image](https://github.com/user-attachments/assets/bf3cac16-4b85-4417-8e5d-589921847548)
![image](https://github.com/user-attachments/assets/60158194-56a8-427a-a04f-071a5ea6156d)

---

## 3. 필터(Filter)의 역할과 이미지 처리
다양한 커널 값을 적용하여 이미지에서 특정 특징(Feature)을 추출하는 과정을 시각화했습니다.

### 3.1 Blurring (부드럽게)
모든 픽셀 값을 평균내어 노이즈를 제거합니다. (Low-pass Filter)
```python
filter = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]]) / 9
```
![image](https://github.com/user-attachments/assets/1bc68e22-c78b-4b9b-8a44-3c76490e3f77)

### 3.2 Sharpening (선명하게)
중앙 픽셀을 강조하고 주변 픽셀을 빼서 경계를 뚜렷하게 만듭니다. (High-pass Filter)
```python
filter = np.array([[-1,-1,-1],
                   [-1,9,-1],
                   [-1,-1,-1]])
```
![image](https://github.com/user-attachments/assets/5a17eb60-18a6-45c0-a966-3bf53b9ce049)

### 3.3 Edge Detection (경계선 검출)
이미지의 명암이 급격히 변하는 부분(Edge)만 남깁니다. CNN의 초기 레이어가 주로 학습하는 특징입니다.
```python
filter = np.array([[-1,-1,-1],
                   [-1,8,-1],
                   [-1,-1,-1]])
```
![image](https://github.com/user-attachments/assets/b1079d84-4fce-4502-b591-f8e6e0cd035a)

---

## 4. CNN 구조의 차원(Dimension) 해석
CNN 설계 시 가장 헷갈리는 '채널(Channel)'의 개념을 정리했습니다.

*   **입력 채널 (Input Depth):** 이전 층에서 넘어온 Feature Map의 개수. 이는 **필터의 깊이(Depth)**를 결정합니다. (필터는 항상 입력 채널 전체를 관통하여 연산합니다.)
*   **출력 채널 (Output Depth):** 현재 층에서 사용하는 **필터의 개수(Number of Filters)**가 곧 출력 Feature Map의 개수가 됩니다.
*   **HW 설계 포인트:**
    *   `Conv2D(32, (3,3))`은 32개의 서로 다른 필터 세트가 존재한다는 의미입니다.
    *   하드웨어는 이 32개의 필터 연산을 **병렬(Parallel)**로 처리할지, **순차(Sequential)**로 처리할지에 따라 면적과 성능(Throughput)이 결정됩니다.

![image](https://github.com/user-attachments/assets/fe6df46f-d102-4dcd-926d-454271bc52fc)

---

## 5. TensorFlow CNN 모델 구현 및 성능 평가
### 5.1 Fashion MNIST (CNN 적용)
단순 MLP로는 한계가 있었던 Fashion MNIST에 CNN을 적용했습니다.

*   **모델 구조:** `Conv2D` $\to$ `Conv2D` $\to$ `MaxPooling` $\to$ `Flatten` $\to$ `Dense`
*   **Flatten Layer:** 3차원 Feature Map $(H, W, C)$을 1차원 벡터 $(H \times W \times C)$로 펼쳐서 완전 연결 계층(Dense)에 전달하는 연결 고리입니다.

![image](https://github.com/user-attachments/assets/6fab4874-bbbd-4457-82d0-9985170c34cf)

### 5.2 모델 경량화 실험 (Optimization)
하드웨어 리소스 제한을 가정하여 모델 크기를 줄여보았습니다.
1.  **레이어 축소:** Conv 레이어 하나 제거 $\to$ 정확도 소폭 하락, 연산량 대폭 감소.
2.  **필터 개수 축소:** 채널 수(32, 64)를 절반으로 감소 $\to$ 파라미터 메모리 사용량 감소.

![image](https://github.com/user-attachments/assets/b7007063-081b-4e4f-b3d2-f6487b3f7b75)
![image](https://github.com/user-attachments/assets/ab1ad1d0-4c48-4bcd-8e1d-119592e6b7ed)

### 5.3 CIFAR-100 (고난이도 데이터셋)
100개의 클래스를 가진 CIFAR-100 데이터셋을 학습시켰습니다. MLP에서는 학습이 거의 불가능했으나, CNN을 통해 유의미한 정확도 상승을 확인했습니다. 하지만 여전히 더 깊은 모델(ResNet 등)이 필요함을 알 수 있었습니다.

![image](https://github.com/user-attachments/assets/c53c0557-ee2a-4ba2-83ec-341f21f4b5b6)

---

## 6. 결론 및 하드웨어 설계 인사이트
이번 스터디를 통해 CNN 가속기 설계의 핵심 요구사항을 도출했습니다.

1.  **Line Buffer의 필요성:**
    *   Convolution 연산은 `3x3` 같은 윈도우가 이미지를 훑고 지나갑니다. 전체 이미지를 저장할 필요 없이, 연산에 필요한 **3줄(Line) 분량의 픽셀만 버퍼링**하면 실시간 처리가 가능합니다.
2.  **병렬 MAC Array:**
    *   다수의 필터(32개, 64개 등)를 동시에 계산하기 위해 대규모 병렬 곱셈기가 필요합니다.
3.  **메모리 대역폭 최적화:**
    *   Feature Map 데이터가 레이어를 거칠 때마다 DRAM을 왔다 갔다 하면 성능이 저하됩니다. **On-chip Memory**를 활용한 데이터 재사용(Data Reuse) 전략이 필수적입니다.
---
