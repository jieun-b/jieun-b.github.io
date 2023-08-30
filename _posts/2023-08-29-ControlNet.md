---
title: "[논문 리뷰] Adding Conditional Control to Text-to-Image Diffusion Models"

categories:
  - 논문 리뷰

date: 2023-08-29 
tag: [AI, Diffusion, ControlNet]
---

# Adding Conditional Control to Text-to-Image Diffusion Models

> arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05543)] [[Github](https://github.com/lllyasviel/ControlNet)]
> 

![figure1.png](/assets/img/2023-08-29/figure1.png)

# Introduction

ControlNet은 large image diffusion model을 제어하는 end-to-end 신경망 구조로, task-specific한 입력 조건을 학습하는 역할을 한다. ControlNet은 large diffusion model의 가중치들을 trainable copy와 locked copy로 복제한다. 

locked copy는 수십억 개의 이미지로부터 학습한 네트워크 capability를 보존하며, trainable copy는 task-specific 데이터셋에서 조건부 컨트롤을 학습한다. trainable, locked 블록은 zero convolution이라 불리는 고유한 유형의 컨볼루션 레이어로 연결되는데, 컨볼루션 가중치들은 점진적으로 0에서부터 최적화된 파라미터로 학습에 따라 성장한다. 

production-ready weight들이 보존되므로 학습은 서로 다른 규모의 데이터셋에서 강건하게 이루어진다. 또한 zero convolution은 deep feature에 새로운 노이즈를 추가하지 않기 때문에 새로운 레이어를 처음부터 훈련하는 것에 비해 학습 속도가 빠르다.

# Related Work

- HyperNetwork는 작은 재귀 신경망을 훈련시켜 더 큰 신경망의 가중치를 조정하기 위한 방법으로, ControlNet과 HyperNetwork는 신경망의 동작을 조정하는 방식에서 유사점을 갖는다.
- Image-to-image translation은 다른 도메인의 이미지 사이 매핑을 학습하는 반면 ControlNet은 task-specific condition으로 diffusion 모델을 컨트롤한다는 점에서 다르다.

# Method

## ControlNet

ControlNet은 전체 신경망의 전반적인 동작을 컨트롤 하기 위해 신경망 블록의 입력 조건을 조작한다. 이때 네트워크 블록은 신경망을 구성하는 레이어의 집합을 의미한다.

2D feature를 예로, feature map $\boldsymbol{x} ∈ \mathbb{R} ^{h×w×c}$이 주어질 때, 신경망 블록 $\cal F(\cdot;\Theta)$은 $\boldsymbol{x}$를 다른 feature map $\boldsymbol{y}$로 변환한다. 아래의 식은 그림(a)와 같이 시각화 할 수 있다.

![Untitled](/assets/img/2023-08-29/Untitled.png)

논문에서는 $\Theta$의 모든 매개변수를 잠그고 이를 trainable copy $\Theta_c$로 복사한다. 복제된 $\Theta_c$는 외부 조건 벡터 $c$와 함께 학습된다. 원래의 매개변수와 새로운 매개변수들은 각각 locked copy, trainable copy로 부른다. 원래의 가중치들을 직접 학습시키는 대신 이러한 복사본을 만드는 것은 데이터셋이 작을 때 과적합을 피하고 학습된 대규모 모델의 품질을 유지하기 위하는 데 있다.

신경망 블록들은 zero convolution이라 불리는 weight와 bias가 모두 0으로 초기화된 1$×$1 컨볼루션 레이어로 연결된다. zero convolution 연산은 $\cal Z(\cdot;\cdot)$으로 표기되며 두 개의 매개변수 인스턴스  $\\{\Theta_{z1}, \Theta_{z2}\\}$를 사용해 ControlNet 구조를 아래 식으로 나타낼 수 있다. 그림(b)는 해당 구조를 시각화하여 나타낸 것이다. 

![Untitled](/assets/img/2023-08-29/Untitled%201.png)

![Untitled](/assets/img/2023-08-29/Untitled%202.png)

제로 컨볼루션 레이어의 가중치와 편향은 0으로 초기화되기 때문에, 첫 번째 학습 단계에서 다음과 같이 식을 정의할 수 있다. 또한 $\boldsymbol{y_c}$는 $\boldsymbol{y}$와 동일한 값을 가지게 된다.

![Untitled](/assets/img/2023-08-29/Untitled%203.png)

이는 첫 번째 학습 단계에서 trainable, locked copy 블록의 모든 입력과 출력들은 ControlNet이 존재하지 않을 때와 동일하다는 것을 의미한다. 다시 말해, ControlNet이 어떤 신경망 블록에 적용되더라도 최적화 전까지는 깊은 신경망 특징에 어떤 영향도 미치지 않는다는 것으로 이해할 수 있다. 신경망 블록의 품질은 완벽히 보존되며, 추가적인 최적화는 처음부터 해당 레이어를 학습하는 것과 비교해 fine tuning하는 것 만큼이나 빠르게 이루어 질 수 있다.

다음은 제로 컨볼루션 레이어의 그래디언트 계산을 추론하는 내용이다. 1$×$1 컨볼루션 레이어가 가중치 $\boldsymbol{W}$와 편향 $\boldsymbol{B}$를 가질 때, 어떤 공간 위치 $p$와 채널 인덱스 $i$에서 입력 맵 $\boldsymbol{I} ∈ \mathbb{R} ^{h×w×c}$가 주어진 경우, forward pass는 다음과 같이 표현할 수 있다.

![Untitled](/assets/img/2023-08-29/Untitled%204.png)

제로 컨볼루션은 최적화 이전에 $\boldsymbol{W=0}, \boldsymbol{B=0}$을 가지므로 0이 아닌 $\boldsymbol{I}_{p,i}$의 모든 위치에서 그래디언트는 다음과 같다. 

![Untitled](/assets/img/2023-08-29/Untitled%205.png)

제로 컨볼루션은 feature term $\boldsymbol{I}$의 그래디언트를 0으로 만들 수 있지만 가중치와 편향의 그래디언트는 영향을 받지 않음을 볼 수 있다. feature $\boldsymbol{I}$가 0이 아닌 한, 첫 번째 경사 하강 interation에서 가중치 $\boldsymbol{W}$는 0이 아닌 행렬로 최적화 될 것이다. 우리의 경우에는 feature term은 샘플링된 입력 데이터나 조건 벡터로, 자연스레 0이 아님을 보장한다. 예를 들어, 전체 손실 함수 $\cal L$과 learning rate $\beta_{lr}\not =0$인 고전적인 경사 하강법에서 outside gradient $\partial\cal L/\partial\cal Z(\boldsymbol{I};\\{\boldsymbol{W},\boldsymbol{B}\\})$가 0이 아니라면, 다음 식이 성립한다.

![Untitled](/assets/img/2023-08-29/Untitled%206.png)

$\boldsymbol{W}^*$는 하나의 경사 하강 단계 후의 가중치이며, 이 단계 후에 아래 식과 같이 0이 아닌 그래디언트를 얻고 신경망은 학습을 시작하게 된다. 이러한 방법으로 제로 컨볼루션은 0에서 최적화된 매개변수로 점진적으로 성장하는 연결 레이어가 된다.

![Untitled](/assets/img/2023-08-29/Untitled%207.png)

## ControlNet in Image Diffusion Model

논문에서는  ControlNet을 사용하는 방법을 소개하기 위해 Stable Diffusion에 적용한 예시를 설명하고 있다. Stable Diffusion은 대규모 text-to-image diffusion 모델로 U-net 구조의 형태를 가진다. 자세한 구조는 아래 그림을 참고하도록 하자. 

![Untitled](/assets/img/2023-08-29/Untitled%208.png)

Stable Diffusion은 전체 512$×$512 이미지 데이터셋을 64$×$64 latent 이미지로 변환해 사용한다. 이를 위해 ControlNet도 image-based condition을 64$×$64 feature space로 변환해야 한다. 이를 위해 4개의 컨볼루션 레이어로 구성된 작은 네트워크 $\mathcal{E}(\cdot)$를 사용해 image-space condition $$\boldsymbol{c}_{\mathrm{i}}$$를 feature map $$\boldsymbol{c}_{\mathrm{f}}$$로 인코딩한다.

![Untitled](/assets/img/2023-08-29/Untitled%209.png)

또한 네트워크 그림과 같이, ControlNet은 Stable Diffusion의 12개의 인코딩 블록과 1개의 중간 블록에 대해 trainable copy를 생성한다. 출력은 U-net의 12개 skip-connection과 1개의 중간 블록에 더해진다. ControlNet을 활용해 U-net의 각 레벨을 제어하게 되는 것이다.

이러한 연결 방식은 원래의 가중치가 잠겨있기 때문에 학습 중에 원래 인코더의 그래디언트 계산이 필요하지 않게 된다. 이는 학습 속도를 높이고 GPU 메모리를 절약하는데 도움이 된다.

## Training

Diffusion 알고리즘은 이미지 $$\boldsymbol{z}_{0}$$에 대해 점진적으로 노이즈를 추가하여 이미지 $$\boldsymbol{z}_{t}$$를 생성하게 된다. 이때 $t$는 노이즈가 더해진 횟수를 의미한다. time step $t$와 text prompt $$\boldsymbol{c}_{t}$$, task-specfic condition $$\boldsymbol{c}_{f}$$를 포함하는 조건 집합이 주어질 때, 이미지 확산 알고리즘은 다음의 목적 함수로 네트워크 $\epsilon_\theta$를 학습하여, 노이즈 이미지 $$\boldsymbol{z}_{t}$$에 추가된 노이즈를 예측한다. 

![Untitled](/assets/img/2023-08-29/Untitled%2010.png)

이러한 목적 함수는 fine tuning diffusion model에도 직접 사용될 수 있다. 

학습 중에 저자들은 랜덤하게 텍스트 프롬프트 $$\boldsymbol{c}_{t}$$의 50$\%$를 빈 문자열로 바꾼다. 이는 입력 조건 맵으로부터 semantic content를 인식하는 ControlNet의 능력을 용이하게 한다. Stable Diffusion 모델에서 프롬프트가 보이지 않을 때, 인코더가 프롬프트 대신 입력 조건 맵으로부터 더 많은 의미를 학습하려는 경향이 있기 때문이다.

## Improved Training

Computation device가 매우 제한적이거나 매우 강력한 경우 ControlNet의 학습을 개선하기 위한 전략에 대해 논의한다.

Small-scale training의 경우 ControlNet과 Stable Diffusion 사이 연결을 부분적으로 끊음으로써 수렴 속도를 가속화할 수 있다는 것을 발견했다.  

Large-scale training의 경우 과적합의 위험이 상대적으로 낮기 때문에, 먼저 충분한 횟수 동안 ControlNet을 훈련한 후, Stable Diffusion의 모든 가중치를 해제하고 전체 모델을 함께 훈련시킬 수 있다. 이러한 방법은 보다 문제 특화된 모델을 이끌어 낼 수 있을 것이다.

# Experiment

## Experimental Settings

- CFG(Classifier-free guidance) scale : 9.0
- Sampler : DDIM
- Step : 20

모델 테스트에는 4가지 타입의 프롬프트를 사용했다.

(1) No prompt : 빈 문자열 “”을 사용한다.

(2) Default prompt : “a professional, detailed, high-quality image”를 사용한다. Stable Diffusion은 프롬프트를 주지 않을 경우 랜덤 텍스쳐 맵을 생성하는 경향이 있다. 더 나은 설정은 “an image”, “a nice image”와 같은 의미없는 프롬프트를 입력하는 것이다.

(3) Automatic prompt : 자동 이미지 캡션 방법을 사용해 프롬프트를 생성하고 이를 다시 diffusion에 사용한다.

(4) User prompt : 사용자 입력 프롬프트를 사용한다.

## Qualitative Results

다음은 각각 Canny edge와 Openpose 이미지를 condition으로 사용한 결과이다. 이외의 다른 예제 및 실험 결과들은 논문을 참고하도록 하자.

![Untitled](/assets/img/2023-08-29/Untitled%2011.png)

![Untitled](/assets/img/2023-08-29/Untitled%2012.png)

# Limitation

입력 이미지의 의미가 잘못 인식된 경우, 강력한 프롬프트를 제공하더라도 부정적인 영향을 제거하기 어려울 수 있다.

![Untitled](/assets/img/2023-08-29/Untitled%2013.png)