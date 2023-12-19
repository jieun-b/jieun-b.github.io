---
title: "[논문 리뷰] MegaPortraits: One-shot Megapixel Neural Head Avatars"

categories:
  - 논문 리뷰

date: 2023-09-22 
tag: [AI, GAN, Reenactment]
---
# MegaPortraits: One-shot Megapixel Neural Head Avatars

> ACMMM 2023. [[Paper](https://arxiv.org/abs/2207.07621)]
> 

![img1.png](/assets/img/2023-09-22/img1.png)

# Introduction

MegaPortraits는 단일 이미지로 고해상도 인간 아바타를 생성하기 위한 새로운 모델이다. 논문의 주요 contribution은 다음과 같다. 

1. One-shot neural avatar 생성을 위한 새로운 모델을 제시한다. 512×512 해상도에서의 cross-reenactment 결과는 SOTA 품질을 달성한다. 모델은 아바타의 외관을 latent 3D volume으로 표현하고, 이를 latent motion representation과 결합한다. 여기에 시스템이 latent motion과 appearance representation의 사이를 더 잘 분리할 수 있도록 contrastive loss를 포함한다. 추가로 눈의 애니메이션을 위한 gaze loss도 사용된다.
2. 중간 해상도의 비디오로 학습된 모델은 추가적인 고해상도 이미지를 사용해서 메가픽셀 (1024×1024) 해상도로 업그레이드 된다. 제안된 방법은 동일한 데이터셋을 사용한 다른 베이스라인 방법들에 비해 cross-reenactment 작업에서 더 뛰어난 결과를 보였다.
3. 메가픽셀 모델을 현대 GPU에서 130FPS로 실행되는 10배 빠른 student 모델로 증류한다. 메인 모델은 이전에 보지 못한 사람에 대해서도 아바타 생성을 할 수 있는 반면, student 모델은 특정 외모에 관해서만 훈련되기 때문에 빠른 속도를 가질 수 있다. 이는 딥페이크 생성에 오용을 방지하고 낮은 렌더링 지연을 달성할 수 있다.

# Method

모델은 두 단계로 학습되며, 빠른 추론을 위한 distillation 단계도 추가로 제안한다.

## Base model

![img2.png](/assets/img/2023-09-22/img2.png)

먼저 학습 비디오에서 두 프레임 $$\mathbf{x}_{s}$$와 $$\mathbf{x}_{d}$$를 랜덤하게 샘플링한다. 이때 $$\mathbf{x}_{d}$$는 $$\mathbf{x}_{s}$$와 동일 인물(비디오)이면서 다른 모션을 취하는 이미지가 된다. 드라이버 프레임 $$\mathbf{x}_{d}$$는 입력과 예측된 이미지 $$\hat{\mathbf{x}}_{s\to d}$$의 GT로 사용하게 된다.

소스 프레임 $$\mathbf{x}_{s}$$는 appearance encoder $$\mathbf{E}_\mathrm{app}$$을 통과해 local volumetric feature $$\mathbf{v}_{s}$$와 global desciptor $$\mathbf{e}_{s}$$를 출력한다. 병렬로 소스 이미지와 드라이버 이미지를 각각 motion encoder $$\mathbf{E}_\mathrm{mtn}$$에 통과시켜 motion descriptor를 계산한다. 모션 인코더의 출력은 head rotation $$\mathbf{R}_{s/d}$$, translation $$\mathbf{t}_{s/d}$$, latent expression descriptor $$\mathbf{z}_{s/d}$$이다. 

그 다음 source tuple $$(\mathbf{R}_{s}, \mathbf{t}_{s}, \mathbf{z}_{s}, \mathbf{e}_{s})$$는 warping generator $$\mathbf{W}_{s\to}$$의 입력으로 들어가 3D warping field $$\mathbf{w}_{s \to}$$를 생성한다. $$\mathbf{w}_{s \to}$$는 canonical 좌표 공간으로의 매핑을 통해 volumetric feature $$\mathbf{v}_{s}$$에서 모션 데이터를 제거한다. 이러한 3D warping operation은 $\circ$으로 표현된다. 모션 데이터가 제거된 feature는 3D convolutional network $\mathbf{G}_\mathrm{3D}$에 의해 처리된다. 

Driver tuple $$(\mathbf{R}_{d}, \mathbf{t}_{d}, \mathbf{z}_{d}, \mathbf{e}_{s})$$은 warping generator $\mathbf{W}_{\to d}$로 들어가게 된다. 출력 $$\mathbf{w}_{\to d}$$는 드라이버 모션을 적용하는데 사용된다. 계산된 feature는 다음과 같은 식으로 나타낼 수 있다.

![img3.png](/assets/img/2023-09-22/img3.png){: width="60%" height="60%"}

마지막으로 driver volumetric feature $$\mathbf{v}_{s \to d}$$는 카메라 프레임으로 직교 투영된다. 이 연산은 $\cal{P}$로 표기되며, 그 결과 나타나는 2D feature map은 2D convolutional network $$\mathbf{G}_\mathrm{2D}$$에 의해 출력 이미지로 디코딩된다.

![img4.png](/assets/img/2023-09-22/img4.png){: width="60%" height="60%"}

위에서 설명한 네트워크들의 결합을 $\mathbf{G}_\mathrm{base}$라 부르고 다음처럼 표기할 수 있다.

![img5.png](/assets/img/2023-09-22/img5.png){: width="60%" height="60%"}

전체 아이디어를 다시 정리해보면, 먼저 volumetric feature를 정면 시점으로 회전시키고, $$\mathbf{z}_{s}$$로부터 디코딩된 얼굴 표정 모션을 제거해 이를 3D convolution network으로 처리한 다음, 드라이버의 머리 회전과 모션을 적용하는 것이다. 머리 회전 데이터 측정은 사전 학습된 네트워크를 사용하지만 latent expression vectors $$\mathbf{z}_{s/d}$$와 warping은 직접적인 지도 학습 없이 훈련된다.  

![img6.png](/assets/img/2023-09-22/img6.png)

학습에 사용되는 손실 함수는 이미지 합성을 위한 perceptual loss와 GAN loss, 학습 규제와 모션 분리를 위한 cycle consistency loss를 사용한다.

Perceptual loss는 예측된 이미지 $$\hat{\mathbf{x}}_{s\to d}$$의 모션과 외관을 GT $$\mathbf{x}_{d}$$와 일치하도록 한다. 논문에서는 3가지 유형의 사전 학습된 네트워크를 사용하는데, Regular ILSVRC pre-trained VGG19는 이미지의 일반적인 콘텐츠를, 얼굴 인식을 위해 학습된 VGGFace는 얼굴 외관을, VGG16 기반의 gaze loss는 시선 방향을 일치시키기 위해 사용한다. Gaze loss와 관련한 네트워크는 SOTA 시선 감지 시스템을 증류하기 위해 학습되었다. 이러한 모든 네트워크를 사용해 예측된 이미지 $$\hat{\mathbf{x}}_{s\to d}$$와 GT 이미지 $$\mathbf{x}_{d}$$로부터 얻은 feature map 간의 weighted L1 distance를 계산한다. 최종적인 perceptual loss는 다음과 같이 개별 perceptual loss의 가중 결합으로 나타난다. 

![img7.png](/assets/img/2023-09-22/img7.png){: width="60%" height="60%"}

Adversarial loss는 예측된 이미지의 사실성을 보장한다. 이전 연구를 따라, $\mathbf{G}_\mathrm{base}$와 함께 힌지 적대적 손실을 사용해 multi-scale patch discriminator를 학습시킨다. 또한 학습 안정성을 위해 표준 feature-matching loss를 포함한다.

![img8.png](/assets/img/2023-09-22/img8.png){: width="60%" height="60%"}

Cycle consistency loss는 motion descriptor를 통한 appearance leakage를 방지하기 위해 사용된다. 학습 중에 motion descriptor는 GT와 동일한 이미지를 사용해 계산되기 때문에, 이 규제가 없다면 조명이나 헤어스타일 같은 데서 드라이버 이미지와 소스 이미지 간의 차이가 있을 때 심각한 artifact가 발생한다. 손실은 추가적인 source-driving pair $$\mathbf{x}_{s^*}$$와 $$\mathbf{x}_{d^*}$$를 사용하여 계산된다. 이 이미지는 다른 비디오로부터 샘플링된 것으로, 현재 $$\mathbf{x}_{s}$$, $$\mathbf{x}_{d}$$ 쌍과는 다른 외관을 가진다. 그런 다음 full base model을 적용하여 cross-reenacted image $$\hat{\mathbf{x}}_{s^*\to d}=\mathbf{G}_\mathrm{base}(\mathbf{x}_{s^*},\mathbf{x}_{d})$$를 생성하고 motion descriptor $$\mathbf{z}_{d^*}=\mathbf{E}_\mathrm{mtn}(\mathbf{x}_{d^*})$$를 별도로 계산한다. 생성한 이미지 $$\hat{\mathbf{x}}_{s\to d}$$와 $$\hat{\mathbf{x}}_{s^*\to d}$$에 대한 motion descriptor $$\mathbf{z}_{s\to d}$$, $$\mathbf{z}_{s^*\to d}$$도 네트워크를 통해 얻어낸다. 그런 다음 motion descriptor를 positive pair $\cal{P}$와 negative pair $\cal{N}$으로 정렬한다:$$\cal{P}=\{(\mathbf{z}_{s\to d},\mathbf{z}_{d}),(\mathbf{z}_{s^*\to d},\mathbf{z}_{d})\},\cal{N}=\{(\mathbf{z}_{s\to d},\mathbf{z}_{d^*}),(\mathbf{z}_{s^*\to d},\mathbf{z}_{d^*})\}$$. 추가적인 설명을 덧붙이자면, positive pair의 $$(\mathbf{z}_{s\to d},\mathbf{z}_{d})$$의 경우, 원래의 이미지 $$\mathbf{x}_{s}$$, $$\mathbf{x}_{d}$$을 모델에 통과시켰을 때 얻을 수 있는 $$\mathbf{z}_{d}$$는 드라이버 이미지 $$\mathbf{x}_{d}$$의 모션을 의미하게 된다. $$\mathbf{z}_{s\to d}$$는 $$\hat{\mathbf{x}}_{s\to d}$$를 모션 인코더에 넣어 얻은 결과로, $$\hat{\mathbf{x}}_{s\to d}$$ 이미지는 외관은 소스, 모션은 드라이버 $$\mathbf{x}_{d}$$의 모션을 취하고 있기 때문에 $$\mathbf{z}_{s\to d},\mathbf{z}_{d}$$는 같은 모션임을 알 수 있다. 이렇게 정렬된 쌍들은 코사인 거리를 계산하는데 사용된다.

![img9.png](/assets/img/2023-09-22/img9.png){: width="60%" height="60%"}

여기서 $s$와 $m$은 하이퍼 파라미터이며, 계산된 거리는 large margin cosine loss (CosFace) 계산에 사용된다.

![img10.png](/assets/img/2023-09-22/img10.png){: width="60%" height="60%"}

Base model을 훈련하기 위한 total loss는 다음과 같이 표현된다.

![img11.png](/assets/img/2023-09-22/img11.png){: width="60%" height="60%"}

추가적으로, 이러한 손실은 예측과 정답에 대해 foreground region만 사용해 계산된다. 따라서 모델은 배경 생성 기능이 없으며, 이는 모델의 성능을 방해한다는 것을 발견했다. 대신에 사전 학습된 inpainting(없는 부분을 채움), matting(전경과 배경 구분) 모델을 통해 배경을 사후 학습한다. Inpainting 시스템을 통해 background plate를 얻고 계산된 matte를 이용해 alpha-compositing으로 예측 이미지와 배경을 결합한다. 

## High-resolution model

High-resolution model 학습에서는 base model $$\mathbf{G}_\mathrm{base}$$를 고정시키고 image-to-image translation 네트워크 $$\mathbf{G}_\mathrm{enh}$$만 학습하게 된다. $$\mathbf{G}_\mathrm{enh}$$는 입력 $$\hat{\mathbf{x}}$$을 512$×$512에서 1024$×$1024 해상도를 가지는 $$\hat{\mathbf{x}}^\mathrm{HR}$$로 매핑시킨다. 고해상도 사진 데이터셋을 사용해 모델을 학습시키며, 모든 이미지가 다른 인물이라 가정한다. 이는 모션만 다른 소스, 드라이버 쌍을 형성할 수 없다는 것을 의미한다. 

모델은 두 그룹의 손실 함수를 사용해 학습한다. 첫 번째 그룹은 standard super-resolution objective를 나타내며, $L_{1}$ loss($$\mathcal{L}_\mathrm{MAE}$$로 표기)와 GAN loss $$\mathcal{L}_\mathrm{GAN}$$을 사용한다. 두 번째 그룹은 비지도 방식으로 작동하며 모델이 cross-driving 시나리오에서의 이미지 생성을 잘 수행하도록 한다. 먼저 훈련 이미지 $$\mathbf{x}^\mathrm{HR}$$와 추가 이미지 $$\mathbf{x}_\mathrm{c}^\mathrm{HR}$$를 샘플링하고, initial reconstruction $$\hat{\mathbf{x}}_\mathrm{c}=\mathbf{G}_\mathrm{base}(\mathbf{x}^\mathrm{LR},\mathbf{x}_\mathrm{c}^\mathrm{LR})$$를 생성한다. Base model을 통해 예측된 이미지 $$\hat{\mathbf{x}}_\mathrm{c}$$는 $$\mathbf{G}_\mathrm{enh}$$에 통과하여 고해상도 이미지 $$\hat{\mathbf{x}}_\mathrm{c}^\mathrm{HR}=\mathbf{G}_\mathrm{enh}(\hat{\mathbf{x}}_\mathrm{c})$$를 생성한다. $$\hat{\mathbf{x}}_\mathrm{c}^\mathrm{HR}$$에 대한 고해상도 GT가 없기 때문에 patch discriminator를 사용하여 그 분포를 GT와 일치하도록 한다. 또한 cycle-consistency loss를 적용하여 저해상도에서 콘텐츠 보존을 강제하도록 한다. Cycle-consistency loss에 대한 식은 다음과 같다.

![img12.png](/assets/img/2023-09-22/img12.png){: width="60%" height="60%"}

$$\mathrm{DS}_{k}$$는 $k$번 다운 샘플링 연산을 의미한다. $$\mathbf{G}_\mathrm{enh}$$에 대한 final objective는 예측된 이미지 $$\hat{\mathbf{x}}^\mathrm{HR}$$와 GT 이미지 $$\mathbf{x}^\mathrm{HR}$$에 대해 계산된 adversarial, perceptual loss와 $$\hat{\mathbf{x}}_\mathrm{c}^\mathrm{HR}$$와 $$\mathbf{x}^\mathrm{HR}$$에 대해 계산된 adversarial loss $$\mathcal{L}^\mathrm{c}_\mathrm{adv}$$, cycle-consistency loss $$\mathcal{L}^\mathrm{c}_\mathrm{cyc}$$를 포함한다.

![img13.png](/assets/img/2023-09-22/img13.png){: width="60%" height="60%"}

## Student model

마지막으로 작은 conditional image-to-image translation 네트워크 $$\mathbf{G}_\mathrm{DT}$$를 이용해 모델을 distillation한다. $$\mathbf{G}_\mathrm{DT}$$는 student model이 되며, base 모델과 enhancer가 결합된 full(teacher) 모델 $$\mathbf{G}_\mathrm{HR}=\mathbf{G}_\mathrm{enh}*\mathbf{G}_\mathrm{base}$$의 예측을 모방하기 위해 학습된다. Student는 teacher 모델로 pseudo-ground truth를 생성하여 cross-driving 모드에서만 학습한다. Student 네트워크는 제한된 수의 아바타에 대해 학습하기 때문에 인덱스 $i$를 사용해 조건을 부여한다. 인덱스는 N개의 외관을 가진 집합 $$\{\mathbf{x}_{i}\}^N_{i=1}$$으로부터 이미지를 선택한다. 그러므로 학습은 driving frame $$\mathbf{x}_{d}$$와 인덱스 $i$를 샘플링하고, 두 이미지를 매칭하는 방식으로 이루어진다.

![img14.png](/assets/img/2023-09-22/img14.png){: width="60%" height="60%"}

네트워크는 perceptual, adversarial loss를 결합해 훈련된다.

# Experiments

Base model의 훈련 및 평가에는 VoxCeleb2와 VoxCeleb2HQ를 이용한다. VoxCeleb2HQ는 VoxCeleb2의 고품질 버전으로 512$×$512 해상도에서 이용되며 VoxCeleb2는 256$×$256 해상도에서 사용된다. 고해상도 모델의 학습에는 FFHQ 데이터셋이 사용되며, student 모델의 학습에는 셀피 비디오 및 사진으로 이루어진 독점적인 데이터셋을 사용한다.

## Cross-reenactment evaluation

512×512 해상도에서 base model에 대해 질적 평가한 결과이다. 위 두 줄은 cross-reenactment 시나리오에 대한 이미지이며 마지막 줄은 self-reenactment 시나리오에 대한 결과이다. 

![img15.png](/assets/img/2023-09-22/img15.png){: width="80%" height="80%"}

## High-resolution evaluation

다음은 base model에 super-resolution method를 적용하고 비교한 결과이다.

![img16.png](/assets/img/2023-09-22/img16.png)

Student 모델이 100개의 아바타에 대해 학습하고 생성한 결과이다.

![img17.png](/assets/img/2023-09-22/img17.png){: width="80%" height="80%"}