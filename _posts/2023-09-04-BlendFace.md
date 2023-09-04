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

Face-swapping은 얼굴 표정이나 헤어스타일 같은 대상의 특징들을 보존하면서 대상 인물이 소스 이미지의 인물로 대체되는 얼굴 이미지를 생성하도록 한다. 얼굴 인식 모델에서는 face-swapping을 위한 강력한 identity encoder를 제공하여 소스 입력에서 생성된 이미지로의 identity 이전을 강화한다. 

그러나 identity encoder로 사용되는 얼굴 인식 모델의 편향된 가이던스는 identity attribute entanglement 문제를 발생시킨다. 

![img1.png](BlendFace%20Re-designing%20Identity%20Encoders%20for%20Face-%201d8837bded9f433786ea58befa821ffc/img1.png)

전통적인 얼굴 인식 모델인 ArcFace의 실패 사례는 다음 그림에서 확인할 수 있는데, ArcFace 기반의 face-swapping 모델은 원치 않는 속성을 바꾸는 경향이 있다. 이는 동일 인물의 이미지가 일부 속성들에 대해 강한 상관 관계를 가지기 때문이며, 얼굴 인식 모델은 그러한 속성들을 identity로 잘못 학습하게 되어 face-swapping 모델의 학습에서 잘못된 지도를 야기한다.

BlendFace는 face-swapping을 위한 새로운 identity encoder로, 잘 분리된 identity feature를 제공한다. BlendFace는 swapped image로 ArcFace를 학습시켜 모델이 얼굴 속성에 집중하지 않고 swapped face와 real face의 유사도 분포 간의 격차를 줄이도록 한다. 그런 다음 BlendFace를 사용하여 face-swapping 모델을 학습시키는데, BlendFace는 source feature 추출기 및 손실 함수에서의 identity guidance 역할을 수행한다.

![img2.png](BlendFace%20Re-designing%20Identity%20Encoders%20for%20Face-%201d8837bded9f433786ea58befa821ffc/img2.png)

그림과 같이 기존의 identity encoder를 소스 특징 추출과 손실 계산에서 BlendFace로 교체함으로써 face-swapping model이 더 분리된 face-swapping 결과를 생성하도록 학습된다. 

# Attribute Bias in Face Recognition Models

논문에서는 먼저 face-swapping을 위한 identity encoding에 대해 재고하기 위해 ArcFace에 대한 실험을 진행한다. 핵심 관찰점은 한 개인의 특징을 다른 사람의 특징으로 대체하는 것이 ArcFace에 내재된 속성 편향을 나타내는 identity similarity의 저하를 초래한다는 것이다.

## Identity Distance Loss

Face-swapping에서의 어려움 중 하나는 GT 이미지가 없다는 점이다. 학습 중에 소스와 대상 입력이 되는 서로 다른 인물의 두 이미지가 주어지면, 생성된 이미지들은 소스 인물과 대상의 특징을 보존하기 위해 일부 특징 기반 손실에 의해 제약을 받는다.

이전의 대부분의 방법들은 소스 입력으로부터 인물 정보를 추출하고, 소스 이미지 $X_{s}$와 스왑된 이미지 $Y_{s,t}$간의 identity distance를 측정하기 위해 대규모 얼굴 인식 데이터셋에서 학습된 ArcFace를 채택했다. Identity distance는 다음과 같이 측정되며, $E_{id}$는 ArcFace 인코더, $cos\lang u,v \rang$는 벡터 $u,v$에 대한 코사인 유사도를 나타낸다.

![img3.png](BlendFace%20Re-designing%20Identity%20Encoders%20for%20Face-%201d8837bded9f433786ea58befa821ffc/img3.png)

## Analysis of Identity Similarity

다음은 face-swapping의 관점에서 ArcFace의 속성 편향을 VGGFace2 데이터셋에서 탐색해본다.

![img4.png](BlendFace%20Re-designing%20Identity%20Encoders%20for%20Face-%201d8837bded9f433786ea58befa821ffc/img4.png)

먼저 랜덤하게 $i$번째 인물의 $j$번째 이미지를 샘플링한다. 이는 $X_{i_{j}}$로 표현할 수 있다. 그 다음, $X_{i_{j}}$와 동일 인물의 모든 이미지$\{X_{i_{1}}, X_{i_{2}},\cdots,X_{i_{n_{i}}}\}$에 대해 코사인 유사도를 계산한다. $n_{i}$는 인물 $i$에 대한 이미지 수를 나타낸다.

Face X-ray에서 영감을 받아, 각 $X_{i_{j}}$에 대해 신원이 $i$가 아닌 100개 이미지를 무작위로 샘플링하고, $X_{i_{j}}$에 가장 가까운 얼굴 랜드마크를 가지는 이미지 $\tilde{X}_{i_{j}}$를 찾는다. $X_{i_{j}}$의 Lab space(색상 공간)에서 색상 통계 $\mu$와 $\sigma$를 $\tilde{X}_{i_{j}}$의 공간으로 전송한다. 두 이미지에 대해 마스크 $\hat{M}_{i_{j}}$를 이용하여 블렌딩해, $\tilde{X}_{i_{j}}$의 얼굴을 $X_{i_{j}}$로 대체한다. 마스크 $\hat{M}_{i_{j}}$는 $X_{i_{j}}$의 내부 마스크 $M_{i_{j}}$와 $\tilde{X}_{i_{j}}$의 $\tilde{M}_{i_{j}}$을 곱해 생성된다. 이는 다음과 같은 식으로 나타낼 수 있다.

![img5.png](BlendFace%20Re-designing%20Identity%20Encoders%20for%20Face-%201d8837bded9f433786ea58befa821ffc/img5.png)

여기서 $\odot$은 point-wise product, $\hat{X}_{i_{j}}$는 합성된 스왑 이미지이며, $\hat{M}_{i_{j}}=Blur(M_{i_{j}}\odot\tilde{M}_{i_{j}})$이다. 

다음으로 $X_{i_{j}}$와 대체된 이미지 $\{\hat{X}_{i_{1}}, \hat{X}_{i_{2}},\cdots,\hat{X}_{i_{n_{i}}}\}$ 간의 코사인 유사도와, $X_{i_{j}}$와 가장 가까운 이미지 $\tilde{X}_{i_{j}}$ 사이의 코사인 유사도도 계산한다. 이러한 절차를 모든 인물에 대해 반복하고, 다음과 같은 유사도 분포를 얻어냈다. 

![img6.png](BlendFace%20Re-designing%20Identity%20Encoders%20for%20Face-%201d8837bded9f433786ea58befa821ffc/img6.png)

실험의 핵심 결과는 다음과 같다.

1) 동일 인물이라도 다른 두 이미지의 유사도는 거의 0.85 이하이다. 

2) Anchor 이미지와 합성된 이미지의 유사도는 실제 이미지와의 유사도에 비해 낮은 결과를 보인다. 이는 얼굴의 색상 분포와 얼굴 외부 영역의 특성이 유사도에 강한 영향을 미친다는 것을 나타낸다. 

결과적으로 전통적인 얼굴 인식 모델을 사용해 identity loss를 최소화하는 것은 대상 속성을 보존하는데 충돌이 있음을 가정할 수 있다.

# BlendFace

## Pre-training with Swapped Faces

이전의 논의처럼, 실제 얼굴 데이터셋에서 학습되는 전통적인 얼굴 인식 모델은 각 인물의 이미지가 일부 특성에서 높은 상관관계를 가지기 때문에 우연히 속성 편향을 학습하게 된다. 논문에서는 이러한 문제를 해결하기 위해, 속성이 스왑된 합성 얼굴 이미지로 얼굴 인식 모델을 학습해 편향을 보정할 수 있는 identity encoder, BlendFace를 개발한다. 

기본 모델로는 ArcFace를 채택하고, 이를 스왑된 속성을 가진 blended 이미지로 학습시킨다. 학습하는 동안, 각 샘플에 대해 [Analysis of Identity Similarity](https://www.notion.so/BlendFace-Re-designing-Identity-Encoders-for-Face-Swapping-1d8837bded9f433786ea58befa821ffc?pvs=21)에서와 동일한 방식으로 입력 이미지의 속성을 확률 $p$로 스왑한다. ArcFace의 손실 함수는 다음과 같다.

![img7.png](BlendFace%20Re-designing%20Identity%20Encoders%20for%20Face-%201d8837bded9f433786ea58befa821ffc/img7.png)

$\theta_{y_{i}}$는 인코더의 deep feature vector와 weight vector의 각도를 의미한다. $K, s, m$은 각각 클래스 수, scale, margin을 나타낸다. 이러한 사전 훈련은 Swapped와 Same의 분포 격차를 줄인다.

## Face-Swapping with BlendFace

![img8.png](BlendFace%20Re-designing%20Identity%20Encoders%20for%20Face-%201d8837bded9f433786ea58befa821ffc/img8.png)

Face-swapping model의 구조는 다음과 같다. 소스, 타겟, 생성 이미지를 각각 $X_{s},X_{t},Y_{s,t}(=G(X_{s},X_{t}))$로 나타낸다. 모델은 AEI-Net의 구조를 따르며, 블렌딩 마스크 $\hat{M}$의 예측을 위해 속성 인코더의 마지막 층에 컨볼루션, 시그모이드 레이어를 추가해 사용한다. 소스 identity 인코딩과 distance loss $\cal{L}_{id}$ 계산에서 사용된 ArcFace는 BlendFace로 교체하고, 블렌딩 마스크 예측기는 속성 인코더로 통합시킨다. 예측된 마스크 $\hat{M}$를 이용해 foreground face image $\tilde{Y}_{s,t}$와 target image $X_{t}$를 블렌딩해 최종 이미지를 얻는다.

![img9.png](BlendFace%20Re-designing%20Identity%20Encoders%20for%20Face-%201d8837bded9f433786ea58befa821ffc/img9.png)

모델에서, 동일한 대상 이미지에 대한 블렌딩 마스크는 소스 이미지와 독립적으로 같아야 한다고 가정한다. 손실은 블렌딩한 결과 $Y_{s,t}$에서 계산되므로 중간 생성 얼굴 $\tilde{Y}_{s,t}$는 얼굴 외부에서 노이즈가 발생한다.

예측된 마스크 $\hat{M}$은 face-parsing model로부터 얻은 GT 마스크 $M$과 binary cross entropy loss $\cal{L}_{mask}$에 의해 감독된다. $\cal{L}_{mask}$는 다음처럼 나타낼 수 있으며, $x$와 $y$는 이미지의 공간 좌표를 의미한다.

![img8.png](BlendFace%20Re-designing%20Identity%20Encoders%20for%20Face-%201d8837bded9f433786ea58befa821ffc/img8%201.png)

Reconstruction loss를 활성화할 때는 같은 이미지를 대신해 소스와 타겟 입력에 대해 동일한 identity를 공유하는 다른 이미지를 넣는다. 소스 및 타겟 이미지에 대한 동일한 identity는 $p=0.2$의 확률로 샘플링된다. 

![img10.png](BlendFace%20Re-designing%20Identity%20Encoders%20for%20Face-%201d8837bded9f433786ea58befa821ffc/img10.png)

기존의 FaceShifter에서 사용한 attributes loss 대신 cycle consistency loss를 사용하며, Gau-GAN의 adversarial loss $\cal{L}_{adv}$를 사용한다. 

![img11.png](BlendFace%20Re-designing%20Identity%20Encoders%20for%20Face-%201d8837bded9f433786ea58befa821ffc/img11.png)

최종적인 loss $\cal{L}$은 다음과 같다. 

![img12.png](BlendFace%20Re-designing%20Identity%20Encoders%20for%20Face-%201d8837bded9f433786ea58befa821ffc/img12.png)

# Experiment