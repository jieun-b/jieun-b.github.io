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

![img1.png](/assets/img/2023-08-29/img1.png)

# Introduction

본 논문에서는 사전 학습된 text-to-image diffusion 모델에 대한 조건부 제어를 학습하기 위해 end-to-end 신경망 아키텍쳐인 ControlNet을 제안한다. ControlNet은 사전 학습된 모델의 가중치들을 잠가 기존 모델의 성능을 보존하고, 인코딩 레이어의 가중치를 trainable copy로 복제한다. 이러한 구조에서 기존의 대형 모델은 다양한 조건부 제어 학습을 위한 강력한 backbone으로 사용된다. Trainable copy와 locked 모델은 가중치가 0으로 초기화된 zero convolution layer로 연결되며, 컨볼루션 가중치들은 학습 동안 점진적으로 성장하게 된다. 이는 학습 초기에 기존 모델의 deep feature에 해로운 노이즈가 추가되지 않도록 하고, trainable copy의 large-scale pretrained backbone이 그러한 노이즈에 의해 손상되지 않도록 보호한다.

# Method

## ControlNet

![img2.png](/assets/img/2023-08-29/img2.png){: width="80%" height="80%"}

파라미터 $\Theta$를 가지는 학습된 신경망 블록을 $\cal F(\cdot;\Theta)$라고 가정해보자. 여기서 네트워크 블록은 신경망을 구성하는 레이어의 집합을 의미한다. 예를 들어, 2D feature map $\boldsymbol{x} ∈ \mathbb{R} ^{h×w×c}$가 있을 때, $\cal F(\cdot;\Theta)$는 그림 (a)와 같이 $\boldsymbol{x}$를 feature map $\boldsymbol{y}$로 변환하게 된다.

![img3.png](/assets/img/2023-08-29/img3.png){: width="60%" height="60%"}

이러한 사전 학습된 신경망 블록에 ControlNet을 추가하기 위해서는 기존 블록의 매개변수 $\Theta$를 잠그고, 동시에 매개변수 블록을 trainable copy로 복사한다. 이 복사본의 매개변수는 $\Theta_c$로 표시된다. Trainable copy는 외부 조건 벡터 $\boldsymbol{c}$를 입력으로 가지게 된다. 이를 Stable Diffusion 같은 대형 모델에 적용하면, locked parameter는 기존 모델을 보존하는 동시에 traninable copy는 해당 모델을 재사용해 다양한 입력 조건을 처리할 수 있게 된다.

Trainable copy는 $\cal Z(\cdot;\cdot)$로 표현되는 zero convolution layer를 통해 locked model에 연결된다. Zero convolution layer는 weight와 bias가 모두 0으로 초기화된 1$×$1 컨볼루션 레이어로, ControlNet을 빌드업 하기 위해 파라미터 $\Theta_{z1}$와 $\Theta_{z2}$를 갖는 두 개의 zero convolution 인스턴스를 사용한다. 최종적으로 ControlNet은 다음과 같은 식을 계산하게 된다.

![img4.png](/assets/img/2023-08-29/img4.png){: width="60%" height="60%"}

첫 번째 학습 단계에서는 zero convolution layer의 가중치와 편향이 0으로 초기화되기 때문에, 위 식의 $\cal Z(\cdot;\cdot)$항들은 모두 0으로 계산된다. 따라서 $\boldsymbol{y_c}=\boldsymbol{y}$가 성립하게 된다. 이러한 방식으로, 학습이 시작될 때 trainable copy에서 해로운 노이즈가 신경망 레이어의 hidden state에 영향을 미치지 못하도록 한다. 또한, $\cal Z(\boldsymbol{c};\Theta_{z1})=0$이고 trainable copy는 입력 이미지 $\boldsymbol{x}$를 받기 때문에 trainable copy는 fully functional하며 사전 학습된 모델의 성능을 보존해 추가 학습을 위한 백본으로 작용할 수 있다. Zero convolution은 초기 학습 단계에서 그래디언트로 랜덤 노이즈를 제거함으로써 백본을 보호한다.

## ControlNet for Text-to-Image Diffusion

ControlNet이 large pretrained diffusion model에 조건부 제어를 추가하는 방법을 설명하기 위해 Stable Diffusion을 예시로 활용한다. Stable Diffusion은 encoder, middle block, skip-connected decoder를 포함하는 U-Net 구조의 형태를 가지고 있다. 모델은 전체 25개 블록으로 구성되어 있는데, 인코더와 디코더 각각 12개 블록을 포함하며 하나의 중간 블록이 있다. 25개 블록 중 8개의 블록은 다운 샘플링 또는 업 샘플링 컨볼루션 레이어이며, 다른 17개 블록은 메인 블록으로 각각 4개의 resnet 레이어와 2개의 ViT를 포함한다. 

다음 그림에서 SD Encoder Block A는 4개의 resnet 레이어와 2개의 ViT로 이루어져 있으며 $×$3은 블록이 3번 반복됨을 의미한다. 텍스트 프롬프트는 CLIP 텍스트 인코더를 사용해 인코딩되며, diffusion timestep은 positional encoding을 이용해 time 인코더로 인코딩된다.

![img5.png](/assets/img/2023-08-29/img5.png){: width="80%" height="80%"}

ControlNet의 구조는 U-net의 각 인코더 레벨에 적용된다. ControlNet은 Stable Diffusion의 12개의 인코딩 블록과 1개의 중간 블록에 대해 trainable copy를 생성하고, 출력은 U-net의 12개 skip-connection과 1개의 중간 블록에 더해진다.

이렇게 ControlNet을 연결하는 방식은 기존 파라미터를 잠가 잠금된 인코더에서 finetuning을 위한 그래디언트 계산이 필요하지 않게 된다. 이는 학습 속도를 높이고 GPU 메모리를 절약하는데 도움이 된다.

Stable Diffusion은 denoising process를 latent space에서 처리하기 때문에 512$×$512 이미지 데이터셋을 64$×$64 latent 이미지로 변환해 사용한다. ControlNet을 Stable Diffusion에 추가하기 위해 입력 조건 이미지를 64$×$64 feature space vector로 변환해 Stable Diffusion의 크기와 맞출 수 있도록 한다. 이를 위해 4개의 컨볼루션 레이어로 구성된 작은 네트워크 $\mathcal{E}(\cdot)$를 사용해 image-space condition $$\boldsymbol{c}_\mathrm{i}$$를 feature space conditioning vector $$\boldsymbol{c}_\mathrm{f}$$로 인코딩한다.

![img6.png](/assets/img/2023-08-29/img6.png){: width="60%" height="60%"}

## Training

Diffusion 알고리즘은 이미지 $$\boldsymbol{z}_{0}$$에 대해 점진적으로 노이즈를 추가하여 이미지 $$\boldsymbol{z}_\mathrm{t}$$를 생성하게 된다. 이때 $t$는 노이즈가 더해진 횟수를 의미한다. Time step $\boldsymbol{t}$와 text prompt $$\boldsymbol{c}_\mathrm{t}$$, task-specfic condition $$\boldsymbol{c}_\mathrm{f}$$를 포함하는 조건 집합이 주어질 때, 이미지 확산 알고리즘은 다음의 목적 함수로 네트워크 $\epsilon_\theta$를 학습하여, 노이즈 이미지 $$\boldsymbol{z}_\mathrm{t}$$에 추가된 노이즈를 예측한다. 

![img7.png](/assets/img/2023-08-29/img7.png){: width="60%" height="60%"}

$\cal{L}$은 전체 diffusion 모델의 learning objective이며, ControlNet을 통한 finetuning diffusion model에 직접 사용된다.

학습에서 텍스트 프롬프트 $$\boldsymbol{c}_\mathrm{t}$$의 $$50\%$$는 랜덤하게 빈 문자열로 바꾼다. 이는 입력 조건 맵에서 semantic content를 인식하는 ControlNet의 능력을 향상시킨다. 

학습하는 동안 zero convolution이 네트워크에 노이즈를 추가하지 않기 때문에 모델은 항상 고품질 이미지를 예측할 수 있다. 모델은 점진적으로 제어 조건을 학습하지 않고 갑자기 조건 이미지를 따라가게 되는데 이러한 현상은 sudden convergence phenomenon이라 부르며, 다음 그림에서 확인할 수 있다.

![img8.png](/assets/img/2023-08-29/img8.png){: width="80%" height="80%"}

## Inference

### Classifier-free guidance resolution weighting

Stable Diffusion은 Classifier-Free Guidance(CFG)에 의존해 고품질 이미지를 생성한다. CFG는 다음과 같이 정의된다. $\epsilon_\mathrm{prd}=\epsilon_\mathrm{uc}+\beta_\mathrm{cfg}(\epsilon_\mathrm{c}-\epsilon_\mathrm{uc})$. $\epsilon_\mathrm{prd}$, $\epsilon_\mathrm{uc}$, $\epsilon_\mathrm{c}$, $\beta_\mathrm{cfg}$는 각각 모델의 최종 출력, 조건이 없는 출력, 조건부 출력, 사용자 지정 가중치를 나타낸다. 

ControlNet을 통해 조건 이미지가 추가될 때는 $\epsilon_\mathrm{uc}$와 $\epsilon_\mathrm{c}$에 모두 추가하거나 $\epsilon_\mathrm{c}$에만 추가할 수도 있다. 만약 프롬프트가 없는 경우, $\epsilon_\mathrm{uc}$와 $\epsilon_\mathrm{c}$에 모두 추가하면 CFG guidance가 완전히 제거되고 $\epsilon_\mathrm{c}$에만 추가하면 guidance가 매우 강력해진다. 이는 각각 그림 b, c에서 확인할 수 있다.

![img9.png](/assets/img/2023-08-29/img9.png){: width="80%" height="80%"}

이에 대한 해결 방법으로 조건 이미지를 $\epsilon_\mathrm{c}$에 추가하고, Stable Diffusion과 ControlNet 사이 각각의 연결에 가중치 $w_{i}$를 곱한다. CFG guidance의 강도를 줄임에 따라 그림 d와 같은 결과를 얻을 수 있다. 이는 CFG Resolution Weighting으로 부르게 된다.

### Composing multiple ControlNets

Stable Diffusion의 단일 인스턴스에 여러 조건 이미지를 적용하기 위해 해당 ControlNet의 출력을 Stable Diffusion 모델에 직접 추가할 수 있다. 이에 추가적인 가중치나 선형 보간은 필요하지 않다.

![img10.png](/assets/img/2023-08-29/img10.png){: width="80%" height="80%"}

# Experiment

## Qualitative Results

![img11.png](/assets/img/2023-08-29/img11.png)

## Ablative Study

ControlNet 구조에 대한 두 가지 실험을 진행한다.

(1) Zero convolution을 가우시안 가중치로 초기화된 표준 convolution layer로 대체해본다. 

(2) 각 블록의 trainable copy를 단일 컨볼루션 레이어로 대체해본다. 이는 ControlNet-lite라고 부르게 된다.

다음으로 4가지 프롬프트 설정에 대해 테스트 해 본다.

(1) 프롬프트를 사용하지 않는 경우로, 빈 문자열을 사용한다.

(2) 조건 이미지의 객체를 완전히 포함하지 않는 부족한 프롬프트를 사용한다. 논문에서는 default prompt로 “a high-quailty, detailed, and professional image”를 설정했다.

(3) 조건 이미지의 의미를 변경하는 모순된 프롬프트를 사용한다.

(4) 필요한 콘텐츠의 의미를 설명하는 완벽한 프롬프트를 사용한다.

![img12.png](/assets/img/2023-08-29/img12.png)