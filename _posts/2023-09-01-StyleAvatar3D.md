---
title: "[논문 리뷰] StyleAvatar3D: Leveraging Image-Text Diffusion Models for High-Fidelity 3D Avatar Generation"

categories:
  - 논문 리뷰

date: 2023-09-01 
tag: [AI, Diffusion, 3D Generation, Avatar]
---
# StyleAvatar3D: Leveraging Image-Text Diffusion Models for High-Fidelity 3D Avatar Generation

> arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19012)] [[Github](https://github.com/icoz69/StyleAvatar3D)]
> 

![img1.png](/assets/img/2023-09-01/img1.png)

# Introduction

StyleAvatar3D는 스타일화된 3D 아바타 생성을 위한 새로운 프레임워크를 제시한다. Pre-trained image-text diffusion model를 통해 데이터를 생성하고, 학습에는 EG3D를 사용한다. 데이터 생성에서 텍스트 프롬프트를 통해 아바타의 스타일과 얼굴 속성을 정의할 수 있으며, 이는 아바타 생성의 유연성을 크게 향상시킨다. 

StableDiffusion을 기반으로 하는 ControlNet을 활용해 predefined pose를 가이드로 2D 이미지를 생성하고, 추가적인 image-pose misalignment 문제를 해결하기 위해 coarse-to-fine pose-aware discriminator를 제안한다. 마지막으로 이미지 입력을 통해 조건부 3D를 생성하기 위해 latent diffusion 모델을 StyleGAN의 latent style space에서 학습시킨다.

# Method

## Generating Multi-View Images

StyleAvatar3D는 포즈 가이드가 있는 다중 뷰 이미지를 생성하기 위해 ControlNet을 활용한다. 아바타 스타일 정의는 텍스트를 통해 할 수 있다. 데이터셋 생성 파이프라인은 다음 그림과 같다.

![img2.png](/assets/img/2023-09-01/img2.png)

ControlNet $$\cal{C}_\theta$$ 은 pose image $I_\mathrm{p}$와 text prompt $T$를 입력으로 받아 스타일화된 이미지 $$I_{s}:I_{s}=\cal{C}_\theta (\mathit{I}_\mathrm{p},\mathit{T})$$ 를 생성한다. 텍스트 프롬프트 $T$는 positive 프롬프트와 negative 프롬프트로 구성되어있다: $$T=(T_\mathrm{pos},T_\mathrm{neg})$$. 두 프롬프트는 각각 이미지에서 원하는 특성과 원하지 않는 특성으로 설명될 수 있다.

가이드를 위한 포즈 이미지는 엔진의 기존 3D 아바타 모델을 사용한다. 아바타 모델에서 포즈 이미지를 추출하기 위해 아바타 머리 중심을 월드 좌표계의 원점으로 설정하고, 카메라가 이 원점을 향하도록 한다. 아바타 정면의 pitch, yaw 각도는 0도로 가정한다. 카메라의 위치는 -180도에서 180도의 yaw 범위와 -30도에서 30도의 pitch 범위 내에서 랜덤하게 샘플링 된다. 다시 말해, 카메라는 원점을 기준으로 미리 결정된 반지름 만큼의 거리에서, 제한된 각도 범위를 가지고 회전하여 다중 뷰 이미지를 생성해낸다.

![img3.gif](/assets/img/2023-09-01/img3.gif)

포즈 이미지 $I_\mathrm{p}$는 세 가지 유형으로, depth map, human pose(Openpose), 이를 통합한 hybrid guidance이다. 포즈 이미지들은 모두 RGB 이미지이며, 포즈 이미지가 엔진 내에서 생성되기 때문에 합성된 이미지 $I_s$의 카메라 파라미터 $c$도 동시에 얻을 수 있다. 

### View-related prompts

ControlNet은 사전 학습된 human pose estimator에 의해 제공되는 pseudo pose label에 의존하기 때문에, 얼굴 각도가 큰 아바타 합성이 제대로 이루어지지 않는 경우가 많다. 이 문제는 depth guidance를 사용할 때도 나타날 수 있는데, depth map이 특히 머리 뒤쪽의 포즈를 정확하게 반영하지 못할 수 있기 때문이다. 

이에 논문에서는 얼굴 측면이나 뒷면과 같은 특정 뷰를 생성하기 위해 view-related prompt $T_{\mathrm{view}}$를 positive prompt로 포함시키고, 눈이나 코와 같은 보이지 않는 얼굴 특징과 연관된 negative prompt를 $T_\mathrm{neg}$에 도입한다. 

### Attribute-related prompts

StableDiffusion은 유사한 얼굴 특성을 가진 편향된 아바타를 만드는 경향이 있다. 이러한 문제는 생성된 데이터셋의 다양성을 제한시킬 수 있는데 이를 해결하기 위해 수동으로 attribute-related prompt $T_\mathrm{att}$를 도입한다. 이 프롬프트는 헤어스타일, 얼굴 표정, 눈 모양과 같은 측면을 포함하며, 20개의 다른 얼굴 속성을 가진다. 생성 과정에서는 5개의 얼굴 속성을 랜덤하게 샘플링하고 각 속성에 대해 하나의 카테고리를 선택한다. 

결과적으로 positive prompt는 style-related prompt $T_\mathrm{style}$, view-related prompt $T_\mathrm{view}$, attribute-related prompt $T_\mathrm{att}$로 구성된다.

## Addressing the Issue of Image-Pose Misalignment

합성된 다중 뷰 이미지로 3D generator를 학습할 때 중요한 과제는 image-pose misalignment 문제이다. 생성된 이미지의 아바타 포즈가 포즈 이미지와 일치하지 않을 때 발생하는데, ControlNet은 guidance와 정렬에 용이한 얼굴 특징들로 정확하게 정면 이미지를 생성해낸다. 그러나 측면이나 후면 뷰 합성에는 어려움이 있으며 pose annotation이 생성된 이미지에 더 이상 맞지 않는 상황이 발생할 수 있다.

![img4.png](/assets/img/2023-09-01/img4.png)

논문에서는 새로운 coarse-to-fine discriminator를 제안하여 pose annotation이 정확히 일치하지 않는 상황에서도 다중 뷰 이미지를 학습할 수 있도록 한다. 각 이미지는 두 가지 유형의 pose annotation과 연관된다. 하나는 더 정확한 pose annotation에 해당하는 fine pose label $c_\mathrm{fine}$과 다른 하나는 이미지 뷰의 일반화된 표시를 제공하는 coarse pose label $c_\mathrm{coarse}$이다.

먼저 모든 렌더링 뷰를 yaw와 pitch 값을 기반으로 $N_\mathbf{group}$개의 그룹으로 나눈다. 각 그룹에는 one-hot yaw representation과 one-hot pitch representation이 할당된다. fine pose label을 얻기 위해 큰 그룹 번호를 할당하고, 작은 그룹 번호는 coarse pose label을 얻기 위해 할당된다. 두 유형의 라벨은 yaw와 pitch의 one-hot 표현을 연결(concatenate)해 표현된다. 판별자에서 사용되는 최종 포즈 라벨은 fine label과 coarse label을 연결하여 형성된다. $c=c_\mathrm{fine}\|\|c_\mathrm{coarse}$. 

학습에서는 하나의 pose annotation 유형을 샘플링해 사용하고, 다른 하나는 0으로 설정하는데, 높은 정렬 정확도를 보이는 confident view에서는 fine pose annotation의 높은 샘플 확률 $p_\mathrm{h}$가 할당된다. 반대로 낮은 정확도를 보이는 뷰는 낮은 샘플 확률 $p_\mathrm{l}$을 할당한다. Confident view는 정면에 가까운 뷰로 정의할 수 있는데, 경험에 따르면 이러한 뷰가 생성된 이미지와 가장 정확하게 정렬될 가능성이 높기 때문이다.

## Image-Guided 3D Generation through Latent Diffusion

EG3D는 스타일 코드와 생성자를 최적화하여, 출력을 목표 입력 이미지에 정렬하도록 하는 pivotal tuning을 이용해 조건화된 얼굴을 생성한다. 3D GAN의 경우, 입력 이미지의 렌더링된 뷰, 즉 포즈를 추가로 제공해야 하는데, 복잡한 스타일의 경우 스타일화된 아바타의 포즈 추정은 어려울 수 있다. 

이에 StyleGAN의 latent style space $\mathcal{W}$에서 작동하는 조건부 diffusion 모델을 개발한다. 학습된 3D generator로부터 이미지와 스타일 벡터 쌍을 무작위로 샘플링한다. 구체적으로는 생성자로부터 랜덤하게 생성된 아바타의 정면 이미지를 렌더링하고 스타일 벡터를 기록하는 것을 의미한다.

Diffusion model의 목표는 렌더된 정면 이미지를 가이드로 노이즈로부터 스타일 벡터를 diffuse하는 것이다. Diffusion model $\boldsymbol{\epsilon}_\theta$에는 PriorTransformer를 사용하는데, 노이즈 스타일 벡터 $\boldsymbol{w}$와 정면 이미지의 CLIP 임베딩 $\boldsymbol{y}$를 입력으로 받아 노이즈 $\boldsymbol{\epsilon}$을 예측한다. 

![img5.png](/assets/img/2023-09-01/img5.png){: width="60%" height="60%"}

학습하는 동안에는 classifier-free diffusion guidance에서 사용한 방식을 이용하는데 조건 임베딩은 확률 $p_\mathrm{drop}$으로 랜덤하게 0이 된다. 추론에서는 제안된 조건으로 3D 아바타를 생성하기 위해 guidance strength $\lambda$를 조절할 수 있다.

학습이 완료되면 기존의 3D generator에서의 스타일 매핑 네트워크를 우리의 학습된 diffusion model로 대체하여 입력 이미지에 따라 조건화된 3D 아바타를 생성할 수 있다. 이러한 방식은 렌더링을 위한 입력 이미지의 포즈를 측정할 필요가 없어 스타일화된 아바타 생성의 정확성을 향상시킨다.

# Experiments

## Dataset

데이터 합성을 위한 50가지의 아바타 스타일을 수집한다. 생성된 모든 이미지는 512$×$512 해상도이다. Analysis 및 ablation study를 위해, 500,000장의 혼합 스타일 데이터셋을 생성한다. 이미지들은 50가지 스타일에서 고르게 샘플링 되었으며 hybrid guidance를 채택한다. 

포즈 이미지로 depth를 사용할 때, 엔진에서 생성된 100,000개의 아바타로부터 Midas model을 사용해 depth map을 추출했다. Human pose guidance의 경우, 하나의 아바타에서 서로 다른 시점에서의 Openpose annotation을 렌더링한다. 추가적으로 학습에서 데이터 증대를 위해 합성된 이미지와 포즈 라벨을 수평으로 뒤집는다.

## Results

### Influence of guidance and prompts on dataset construction

다음은 데이터셋 구축에서 guidance와 prompt의 영향에 대해 나타낸 실험 결과이다.

![img6.png](/assets/img/2023-09-01/img6.png)

노란색 박스를 보면 뒷 머리 이미지를 생성할 때, view-specific prompt가 실패를 줄이는데 효과적임을 알 수 있다. 또한 속성 관련 프롬프트를 사용했을 때 아바타 외모가 더 다양하게 생성된 것을 확인할 수 있다. 가이드 방식의 경우 하이브리드 가이드 방식이 전반적으로 아바타의 품질과 안정성을 향상시켰다. 

### Effectiveness of coarse-to-fine discriminators

![img7.png](/assets/img/2023-09-01/img7.png){: width="80%" height="80%"}

### Latent space walk

무작위 두 입력 벡터 사이에서 선형 보간을 수행한 결과이다. 동시에 렌더링 각도를 선형으로 변경하여 GAN이 latent space에서 어떻게 이미지를 생성하는지, 시점 변화에 어떻게 응답하는지 관찰할 수 있다.

![img8.png](/assets/img/2023-09-01/img8.png)

### Validation of Image-Conditioned 3D Generation

![img9.png](/assets/img/2023-09-01/img9.png)

### Visualization of meshes

Marching cube 알고리즘을 사용해 Tri-plane으로부터 추출된 메쉬를 시각화한 결과이다.

![img10.png](/assets/img/2023-09-01/img10.png)

### LoRA-based cartoon character reconstruction

다음은 StableDiffusion 기반의 LoRA 모델을 사용하여 텍스트로 정의된 스타일을 스타일 혹은 주제에 대한 몇 장의 예시 이미지로 바꿔 실험을 진행한다. 만화 캐릭터를 선택하고 LoRA를 학습시키기 위해 해당 캐릭터에 대한 10장의 이미지를 인터넷에서 모은다. 학습이 완료되면 LoRA를 이용해 다중 뷰 이미지를 생성한다.

![img11.png](/assets/img/2023-09-01/img11.png)