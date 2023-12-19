---
title: "[논문 리뷰] SemanticStyleGAN: Learning Compositional Generative Priors for Controllable Image Synthesis and Editing"

categories:
  - 논문 리뷰

date: 2023-09-26
tag: [AI, GAN, Image synthesis, Editing]
---
# SemanticStyleGAN: Learning Compositional Generative Priors for Controllable Image Synthesis and Editing

> CVPR 2022. [[Paper](https://arxiv.org/abs/2112.02236)] [[Github](https://github.com/seasonSH/SemanticStyleGAN)]
> 

![img1.png](/assets/img/2023-09-26/img1.png)

# Introduction

SemanticStyleGAN은 local semantic part를 학습해 컨트롤 가능한 이미지 합성을 수행하도록 한다. Semantic part는 semantic segmentation mask에 의해 정의되며, semantic part를 기반으로 모델의 latent code를 factorization 한다. 각각의 semantic part는 local latent code로 모듈레이션 되고, 이미지는 local feature map을 조합하여 합성된다. Local latent code는 다른 방법들과 달리 semantic part의 structure와 texture를 둘 다 컨트롤 할 수 있다.

![img2.png](/assets/img/2023-09-26/img2.png)

모델은 StyleGAN과 같이 generic prior로 사용될 수 있으며 StyleGAN을 위해 설계된 latent manipulation 방법들과 결합해 출력 이미지를 편집하고 더 정확한 local 컨트롤을 가능하게 한다.

# Methodology

전형적인 GAN 프레임워크는 표준 정규 분포 $\cal{Z}$의 벡터 $\textbf{z}$를 이미지로 매핑하는 생성자를 학습한다. StyleGAN에서는 데이터 분포의 비선형성을 다루기 위해, $\textbf{z}$를 먼저 MLP에 통과시켜 latent code $\textbf{w}∼\cal{W}$로 매핑한다. $\cal{W}$ 공간은 서로 다른 해상도에서 출력 스타일을 컨트롤 하기 위해 $\cal{W}^+$공간으로 확장된다. 그러나 이러한 latent code는 엄밀한 의미가 없고 개별적으로 사용하기 어렵다. 

본 논문에서는 서로 다른 semantic 영역에 대해 $\cal{W}^+$공간을 분리하도록 한다. 이미지 $x_{i}$와 이미지의 semantic segmentation mask $y_{i}∈\{0, 1\}^{H×W×K}$에 대한 라벨이 있는 데이터셋 $D={(x_{1},y_{1}),(x_{2},y_{2}),…,(x_{n},y_{n})}$가 주어질 때, 생성자는 다음과 같이 factorize된 $\cal{W}^+$를 제공한다. 

![img3.png](/assets/img/2023-09-26/img3.png){: width="60%" height="60%"}

여기서 $K$는 semantic class의 수를 나타내며, 각각의 local latent code $\textbf{w}^{k}∈\cal{W}^{k}$는 segmentation label로 정의된 $k$번째 semantic 영역의 shape과 texture를 컨트롤한다. $\textbf{w}^\text{base}∈\cal{W}^\text{base}$는 포즈와 같은 대략적인 structure를 제어하기 위한 공유 코드이다. 각각의 $$\textbf{w}^{k}$$는 다시 shape code $$\textbf{w}^{k}_{s}$$와 texture code $$\textbf{w}^{k}_{t}$$로 분해된다.

## Generator

![img4.png](/assets/img/2023-09-26/img4.png)

Generator의 전체 구조는 그림과 같다. StyleGAN2와 비슷하게 먼저 8개의 레이어로 이루어진 MLP를 통해 $\textbf{z}$를 중간 코드 $\textbf{w}$로 매핑한다. 그 다음 $K$개의 local generator가 $\textbf{w}$를 이용해 서로 다른 semantic part를 모델링한다. Render net R은 local generator로부터 혼합된 결과를 입력으로 받아 RGB 이미지와 semantic segmentation mask를 출력한다.

### Local Generator

![img5.png](/assets/img/2023-09-26/img5.png)

Local generator는 modulated MLP를 사용한다. 이는 합성된 출력에 대해 명시적인 공간 제어를 가능하게 한다. Local generator $g_{k}$는 Fourier feature(position encoding) $\textbf{p}$와 latent code를 입력으로 받아 feature map $$\textbf{f}_{k}$$와 pseudo-depth map $$\textbf{d}_{k}$$를 출력한다.

![img6.png](/assets/img/2023-09-26/img6.png){: width="60%" height="60%"}

계산 비용을 줄이기 위해 입력 Fourier feature map과 출력은 최종 출력 이미지보다 작은 $H^{c}×W^{c}$로 크기를 줄인다. 학습 중에 각각의 local generator에서는 독립적으로  $$\textbf{w}^{base},\textbf{w}^{k}_{s},\textbf{w}^{k}_{t}$$ 사이 스타일 믹싱을 수행한다. 이를 통해 서로 다른 local 파트와 shape, texture가 공동으로 잘 합성되도록 한다. 한편, Pseudo-depth map의 경우 실제 depth map은 아니지만 z-buffering process를 모방하는 composition 전략으로 사용되기 때문에 depth라고 불린다.

### Fusion

Fusion 단계에서는 먼저 pseudo-depth map으로부터 coarse segmentation mask $\textbf{m}$을 생성한다. 이는 소프트맥스 함수를 통해 수행된다.

![img7.png](/assets/img/2023-09-26/img7.png){: width="60%" height="60%"}

$$\textbf{m}_{k}(i,j)$$는 마스크의 $k$번째 클래스에서의 픽셀 $(i,j)$를 나타내며 $$\textbf{d}_{k}(i,j)$$도 마찬가지이다. 다음으로 마스크와 feature map $$\textbf{f}_{k}$$를 element-wise 곱을 통해 전체 계산된 feature map $\textbf{f}$를 얻는다. 

![img8.png](/assets/img/2023-09-26/img8.png){: width="60%" height="60%"}

Feature map $\textbf{f}$는 출력 이미지에 대한 모든 정보를 포함하고 있으며 렌더링을 위해 $R$로 보내진다. 일부 클래스들이 투명한 경우에는 특징을 집계하는데 마스크 $\textbf{m}$을 직접 사용하는 것이 문제가 될 수 있다. 따라서 투명한 클래스의 경우 수정된 버전 $\tilde{\textbf{m}}$을 사용한다. 자세한 내용은 appendix를 참고하길 바란다.

### Render Net

Render net $R$은 기존의 StyleGAN2 생성자와 유사하지만 일부 수정된 부분을 가진다. Modulated convolution layer를 사용하지 않고 출력은 입력 특징 맵에만 의존하게 된다. 두 번째로 특징 맵은 $16×16, 64×64$ 해상도에서 입력되며 feature concatenation은 $64×64$ 해상도에서 수행된다. 저해상도의 특징 맵을 입력으로 사용하면 다른 부분들에 대해 블렌딩이 더 잘 이루어지게 된다. 마지막으로 소프트맥스 출력과 실제 세그멘테이션 마스크 사이 내재적 간격 때문에 마스크 $\textbf{m}$으로 직접 훈련하는 것이 어렵다는 것을 발견했다. 따라서 각각의 컨볼루션 레이어 뒤에 ToRGB branch 이외에 SemanticGAN과 유사하게 ToSeg branch를 더해 잔차를 출력하도록 한다. 이는 coarse segmentation mask $\textbf{m}$을 최종 마스크 $$\hat{y}=upsample(\textbf{m})+\Delta\textbf{m}$$으로 정제한다. 여기서는 최종 마스크가 coarse mask에서 크게 벗어나지 않도록 regularization loss가 필요하다.

![img9.png](/assets/img/2023-09-26/img9.png){: width="60%" height="60%"}

## Discriminator and Learning Framework

결합 분포 $p(x,y)$를 모델링 하기 위해 판별자는 RGB 이미지와 세그멘테이션 마스크를 입력으로 받아야 한다. 일반적인 연결 방식은 세그멘테이션 마스크의 큰 그래디언트 때문에 작동되지 않기 때문에 dual-branch discriminator $D(x,y)$를 제안한다. 이는 $x$와 $y$에 대한 두 개의 컨볼루션 브랜치를 가진다. 출력은 완전 연결 레이어에서 합산된다. 이러한 설계는 추가적인 R1 regularization loss를 통해 세그멘테이션 브랜치의 gradient norm을 별도로 정규화한다. 

![img10.png](/assets/img/2023-09-26/img10.png){: width="60%" height="60%"}

# Experiments

## Semantic-aware and Disentangled Generation

CelebAMask-HQ 데이터셋에서 학습한 모델이 생성한 결과이다. 이미지는 512$×$512 해상도에서 생성되었다.

![img11.png](/assets/img/2023-09-26/img11.png){: width="80%" height="80%"}

구성 요소 별로 합성을 진행한 결과이다.

![img12.png](/assets/img/2023-09-26/img12.png)

모델의 latent interpolation에 대한 결과이다.

![img13.png](/assets/img/2023-09-26/img13.png){: width="80%" height="80%"}

## Results on Other Domains

모델을 다양한 도메인에서 fine-tuning하고 헤어스타일을 변경한 예시이다.

![img14.png](/assets/img/2023-09-26/img14.png){: width="80%" height="80%"}