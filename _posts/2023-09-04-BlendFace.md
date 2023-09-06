---
title: "[논문 리뷰] BlendFace: Re-designing Identity Encoders for Face-Swapping"

categories:
  - 논문 리뷰

date: 2023-09-04 
tag: [Face-Swapping, BlendFace]
---
# BlendFace: Re-designing Identity Encoders for Face-Swapping

> ICCV 2023. [[Paper](https://arxiv.org/abs/2307.10854)] [[Github](https://mapooon.github.io/BlendFacePage/)]
> 

# Introduction

Face-swapping은 표정이나 헤어스타일 같은 target의 속성을 보존하면서 target identity를 source identity로 대체하는 것을 목표로 한다. 얼굴 인식 모델에서는 face-swapping을 위한 강력한 identity encoder를 제공하여 소스 입력에서 생성된 이미지로의 identity 이전을 강화한다. 

![img1.png](/assets/img/2023-09-04/img1.png)

그러나 이러한 ArcFace 같은 identity encoder는 identity attribute entanglement 문제를 발생시킨다. 그림에서 보듯, ArcFace 기반의 face-swapping 모델은 원치 않는 속성을 스왑하는 경향이 있는 것으로 나타났다. 이는 동일 인물의 이미지가 일부 속성들에 대해 강한 상관 관계를 가지기 때문이며, 얼굴 인식 모델은 그러한 속성들을 identity로 잘못 학습하게 되어 face-swapping 모델의 학습에서 잘못된 지도를 야기한다.

BlendFace는 face-swapping을 위한 새로운 identity encoder로, 잘 분리된 identity feature를 제공한다. BlendFace는 swapped image로 ArcFace를 학습시켜 모델이 얼굴 속성에 집중하지 않고 swapped face와 real face의 유사도 분포 격차를 줄이도록 한다. 그런 다음 BlendFace를 사용하여 face-swapping 모델을 학습시킨다. BlendFace는 source feature 추출기 및 손실 함수에서의 identity guidance 역할을 수행한다.

![img2.png](/assets/img/2023-09-04/img2.png)

그림과 같이 기존의 identity encoder를 BlendFace로 교체함으로써 face-swapping model이 disentangled face-swapping 결과를 생성하도록 학습된다. 

# Attribute Bias in Face Recognition Models

논문에서는 먼저 face-swapping을 위한 identity encoding에 대해 재고하기 위해 ArcFace에 대한 실험 및 분석 진행한다.

## Identity Distance Loss

Face-swapping 문제에서는 GT 이미지를 얻기 어렵기 때문에, 소스와 대상 이미지가 서로 다른 인물일 때 생성된 이미지는 segmentation이나 3D face shape과 같은 feature-based loss를 이용해 학습하게 된다.

대부분의 이전 방법들은 ArcFace를 사용하여 소스 입력으로부터 identity 정보를 추출하고, 소스 이미지 $X_{s}$와 스왑된 이미지 $Y_{s,t}$간의 identity distance를 측정했다. Identity distance에 대한 식은 다음과 같다.

![img3.png](/assets/img/2023-09-04/img3.png){: width="60%",height="60%"}{: .center}

$E_{id}$는 ArcFace 인코더, $cos\< u,v \>$는 벡터 $u,v$에 대한 코사인 유사도를 나타낸다.

## Analysis of Identity Similarity

다음은 face-swapping의 관점에서 ArcFace의 속성 편향을 VGGFace2 데이터셋에서 탐색해본다.

![img4.png](/assets/img/2023-09-04/img4.png)

먼저 $$X_{i_{j}}$$로 표현되는 $i$번째 인물의 $j$번째 이미지를 랜덤하게 샘플링한다. 그 다음, $$X_{i_{j}}$$와 동일 인물의 모든 이미지$$\{X_{i_{1}}, X_{i_{2}},\cdots,X_{i_{n_{i}}}\}$$에 대해 코사인 유사도를 계산한다. $n_{i}$는 인물 $i$에 대한 이미지 수를 나타낸다.

Face X-ray에서 영감을 받아, 각 $X_{i_{j}}$에 대해 신원이 $i$가 아닌 100개 이미지를 무작위로 샘플링하고, $$X_{i_{j}}$$에 가장 가까운 얼굴 랜드마크를 가지는 이미지 $$\tilde{X}_{i_{j}}$$를 찾는다. $$X_{i_{j}}$$와 $$\tilde{X}_{i_{j}}$$ 사이의 코사인 유사도도 계산한다.

$$X_{i_{j}}$$의 마스크 $$M_{i_{j}}$$와 $$\tilde{X}_{i_{j}}$$의 마스크 $$\tilde{M}_{i_{j}}$$을 곱해 마스크 $$\hat{M}_{i_{j}}$$를 생성할 수 있다: $$\hat{M}_{i_{j}}=Blur(M_{i_{j}}\odot\tilde{M}_{i_{j}})$$. $X_{i_{j}}$의 Lab space(색상 공간)에서 색상 통계 $\mu$와 $\sigma$를 $$\tilde{X}_{i_{j}}$$의 공간으로 전송한다. 마스크 $$\hat{M}_{i_{j}}$$을 이용하여 $X_{i_{j}}$와 $$\tilde{X}_{i_{j}}$$를 블렌딩해 $$\tilde{X}_{i_{j}}$$의 얼굴을 $X_{i_{j}}$로 대체한다. 식으로 나타내면 다음과 같다. 여기서 $\odot$은 point-wise product, $$\hat{X}_{i_{j}}$$는 합성된 스왑 이미지를 의미한다.

![img5.png](/assets/img/2023-09-04/img5.png){: width="60%",height="60%"}{: .center}

다음으로 $X_{i_{j}}$와 대체된 이미지 $$\{\hat{X}_{i_{1}}, \hat{X}_{i_{2}},\cdots,\hat{X}_{i_{n_{i}}}\}$$ 간의 코사인 유사도를 계산한다. 이러한 절차를 모든 인물에 대해 반복하고, 다음과 같은 유사도 분포를 얻어냈다. 

![img6.png](/assets/img/2023-09-04/img6.png)

실험의 핵심 결과는 다음과 같다.

1) 동일 인물이라도 다른 두 이미지의 유사도는 거의 0.85 이하이다. 

2) Anchor 이미지와 합성된 이미지의 유사도는 실제 이미지와의 유사도에 비해 낮은 결과를 보인다. 이는 얼굴의 색상 분포와 얼굴 외부 영역의 특성이 유사도에 강한 영향을 미친다는 것을 나타낸다. 

결과적으로 전통적인 얼굴 인식 모델을 사용해 identity loss를 최소화하는 것은 대상 속성을 보존하는데 충돌이 있음을 가정한.

# BlendFace

## Pre-training with Swapped Faces

이전의 논의처럼, 실제 얼굴 데이터셋에서 학습되는 전통적인 얼굴 인식 모델은 각 인물의 이미지가 일부 특성에서 높은 상관관계를 가지기 때문에 우연히 속성 편향을 학습하게 된다. 논문에서는 이러한 문제를 해결하기 위해, 속성이 스왑된 합성 얼굴 이미지로 얼굴 인식 모델을 학습해 편향을 보정할 수 있는 identity encoder, BlendFace를 개발한다. 

기본 모델로는 ArcFace를 채택하고, 이를 스왑된 속성을 가진 blended 이미지로 학습시킨다. 학습하는 동안, 각 샘플에 대해 [Analysis of Identity Similarity](https://www.notion.so/BlendFace-Re-designing-Identity-Encoders-for-Face-Swapping-1d8837bded9f433786ea58befa821ffc?pvs=21)에서와 동일한 방식으로 입력 이미지의 속성을 확률 $p$로 스왑한다. ArcFace의 손실 함수는 다음과 같다.

![img7.png](/assets/img/2023-09-04/img7.png){: width="60%",height="60%"}{: .center}

$\theta_{y_{i}}$는 인코더의 deep feature vector와 weight vector의 각도를 의미하며, $K, s, m$은 각각 클래스 수, scale, margin을 나타낸다. 

![img8.png](/assets/img/2023-09-04/img8.png)

이러한 사전 훈련은 swapped face와 real positive face의 분포 격차를 줄이게 된다. (+ Swapped face로 학습을 진행했음에도 same과 different의 분포는 거의 변하지 않는 것을 확인할 수 있다.)

## Face-Swapping with BlendFace

![img9.png](/assets/img/2023-09-04/img9.png)

Face-swapping model의 구조는 다음과 같다. 소스, 타겟, 생성 이미지를 각각 $X_{s},X_{t},Y_{s,t}(=G(X_{s},X_{t}))$로 나타낸다. 모델은 AEI-Net의 구조를 따르며, 블렌딩 마스크 $\hat{M}$의 예측을 위해 속성 인코더의 마지막 층에 컨볼루션, 시그모이드 레이어를 추가해 사용한다. 소스 identity 인코딩과 distance loss $$\cal{L}_{id}$$ 계산에서 사용된 ArcFace는 BlendFace로 교체하고, 블렌딩 마스크 예측기는 속성 인코더로 통합시킨다. 예측된 마스크 $\hat{M}$를 이용해 foreground face image $$\tilde{Y}_{s,t}$$와 target image $X_{t}$를 블렌딩해 최종 이미지를 얻는다.

![img10.png](/assets/img/2023-09-04/img10.png){: width="60%",height="60%"}{: .center}

모델에서, 동일한 대상 이미지에 대한 블렌딩 마스크는 소스 이미지와 독립적으로 같아야 한다고 가정한다. 손실은 블렌딩한 결과 $Y_{s,t}$에서 계산되므로 중간 생성 얼굴 $\tilde{Y}_{s,t}$는 얼굴 외부에서 노이즈가 발생한다.

Binary cross entropy loss $$\cal{L}_{mask}$$를 사용하여 예측된 마스크 $\hat{M}$와 face-parsing model로부터 얻은 GT 마스크 $M$에 대해 계산한다. $$\cal{L}_{mask}$$는 다음처럼 나타낼 수 있으며, $x$와 $y$는 이미지의 공간 좌표를 의미한다.

![img11.png](/assets/img/2023-09-04/img11.png){: width="60%",height="60%"}{: .center}

Reconstruction loss는 소스와 타겟 입력이 identity가 같은 서로 다른 이미지일 때 활성화된다. 소스 및 타겟 이미지에 대한 동일한 identity는 $p=0.2$의 확률로 샘플링된다. 

![img12.png](/assets/img/2023-09-04/img12.png){: width="60%",height="60%"}{: .center}

FaceShifter에서 사용한 attributes loss 대신 cycle consistency loss를 사용한다. 

![img13.png](/assets/img/2023-09-04/img13.png){: width="60%",height="60%"}{: .center}

Adversarial loss $$\cal{L}_{adv}$$는 Gau-GAN에서의 것을 사용한다. 최종적인 loss $\cal{L}$은 다음과 같다. 

![img14.png](/assets/img/2023-09-04/img14.png){: width="60%",height="60%"}{: .center}

# Experiment

## Implementation Detail

### Pretraining of BlendFace

- Dataset : MS-Celeb-1M
- Batch size : 1024 / Epochs : 20

### Training of face-swapping model

- Dataset : VGGFace2
- Batch size : 32 / Iteration : 300k

## Experiment Setup

FaceForensics++ 데이터셋을 사용해 face-swapping model을 평가한다. 데이터셋은 Deepfakes, Face2Face, FaceSwap, Neural Textures, FaceShifter의 각각 1000개의 실제 비디오와 1000개의 생성된 비디오를 포함한다.

생성된 이미지의 충실도를 평가하기 위해 6가지 지표를 고려한다. ArcFace, BlendFace, face shape, expression, gaze.

## Comparison with Previous Methods

![img15.png](/assets/img/2023-09-04/img15.png)

![img16.png](/assets/img/2023-09-04/img16.png)

# Limitations

1) 얼굴 모양을 거의 변형하기 어렵다.

2) 손과 같은 hard occlusion 보존이 어렵다.