---
title: "DeepSpeed와 ZeRO"
date: 2025-08-08 00:00:00 +0900
layout: post
categories: [공부, Deep Learning]
tags: [DeepSpeed, ZeRO]
image: /assets/img/posts/2025/08/deepspeed-zero/deepspeed-overview.gif
---
# 1. DeepSpeed란?

![deepspeed-overview](/assets/img/posts/2025/08/deepspeed-zero/deepspeed-overview.gif)

DeepSpeed는 Microsoft에서 개발한 **PyTorch 기반 대규모 분산 학습 최적화 라이브러리**로, 수십억에서 수조 개의 파라미터를 갖는 초대형 모델을 단일 노드의 다중 GPU 또는 다중 노드 환경에서도 효율적으로 학습할 수 있도록 설계되어 있다.

DeepSpeed는 데이터 병렬화, 모델 병렬화, 옵티마이저 및 통신 최적화 등 다양한 기술을 포함하며, 그중 대표적인 메모리 효율화 기법이 **ZeRO (Zero Redundancy Optimizer)**이다.

# 2. ZeRO (Zero Redundancy Optimizer)

ZeRO는 대규모 분산 학습 환경에서 **모델 상태의 중복 저장을 제거**하기 위해 설계된 메모리 최적화 기법이다.

전통적인 데이터 병렬 방식에서는 각 GPU가 **모델 파라미터, 그라디언트 및 옵티마이저 상태의 전체 복사본을 유지**하기 때문에 모델 크기가 커질수록 GPU 메모리 사용량이 급격히 증가한다.

ZeRO는 이 문제를 해결하기 위해 해당 상태들을 **GPU(및 필요 시 CPU) 간에 샤딩(partitioning)하여 저장**하고, 필요한 통신 스케줄을 동적으로 관리함으로써 메모리 사용량을 줄이면서도 효율적인 분산 학습을 가능하게 한다.

## 2.1 ZeRO의 세 가지 단계

ZeRO는 데이터 병렬 학습의 메모리 비효율성을 해결하기 위해, 모델 상태(파라미터, 그라디언트, 옵티마이저 상태)를 **GPU 간에 샤딩(sharding)** 하는 방식을 세 단계로 확장한다.

단계가 높아질수록 분산되는 범위가 넓어져 **메모리 절감 효과는 커지지만**, 통신량도 함께 증가한다.

| Stage | 분산 대상 | 설명 |
| --- | --- | --- |
| **ZeRO-1** | Gradients | 그라디언트를 GPU 간 분산 저장 |
| **ZeRO-2** | Gradients + Optimizer states | 옵티마이저 상태까지 분산 저장 |
| **ZeRO-3** | Gradients + Optimizer states + Parameters | 모델 파라미터까지 포함한 **전체 상태 분산** |

### ZeRO-1 : Gradient Sharding

- **기존 Data Parallel과 차이점**은 각 GPU가 **전체 그라디언트**를 보관하지 않고, **자신의 담당 구간만 저장**한다는 점이다.
- 역전파 단계에서 **All-Reduce 대신 Reduce-Scatter**를 사용하여, 각 GPU가 계산한 그라디언트를 합치면서 **일부 구간만 보유**하게 된다.
- 예를 들어 4개의 GPU를 사용하면 GPU0은 gradient 텐서의 앞 25%, GPU1은 그다음 25%를 보관하는 식으로 분할된다.
- 옵티마이저 스텝 직전에는 필요한 부분만 다른 GPU로부터 받아 파라미터 업데이트를 수행한다.

### ZeRO-2 : Gradient + Optimizer State Sharding

- ZeRO-1 의 구조에 더해, 옵티마이저 상태(예: Adam의 1차, 2차 모멘트 m / v 벡터)를 GPU 간에 분산 저장한다.
- 업데이트 시 필요한 조각만 통신으로 가져와 연산 후 다시 분산 저장하기 때문에, **옵티마이저 메모리 사용량이 GPU 수에 비례하여 감소**한다.
- ZeRO-1 / 2에서는 모든 GPU가 여전히 전체 파라미터 사본을 가지지만, 저장되는 gradient 및 optimizer state는 부분적이다.

### ZeRO-3 : Full State Sharding (Parameters Included)

- ZeRO-3 는 모델 파라미터 자체까지 GPU 간에 샤딩한다.
- 따라서 각 GPU는 자신의 담당 파라미터 조각만 보유하고, **Forward / Backward 단계에서 필요할 때 다른 GPU로부터 파라미터를 동적으로 All-Gather** 한다.
- 연산이 끝나면 해당 파라미터를 즉시 메모리에서 해제하기 때문에, **전체 모델을 동시에 로딩하지 않아도 된다.**
- 역전파 중에는 gradient를 계산한 후 바로 Reduce-Scatter 하여 해당 파라미터를 보유한 GPU에 전달한다.
- Optimizer Step에서는 각 GPU가 자기 담당 구간만 업데이트 하고, 필요 시 다른 GPU에 전송하여 모델 동기화를 유지한다.

### 성능 요약

| 단계 | 메모리 절감 효과 | 통신량 변화 |
| --- | --- | --- |
| ZeRO-1 | 최대 4× 감소 | 매우 적음 |
| ZeRO-2 | 최대 8× 감소 | 소폭 증가 (효율적 통신) |
| ZeRO-3 | GPU 수에 비례해 절감 (N× 수준) | 약 50% 통신 증가 가능 |

ZeRO는 이처럼 GPU 메모리 사용량을 극적으로 줄이지만, 모델 상태를 GPU 간에 분산 저장하므로 학습 중에는 필연적으로 통신이 발생한다.

ZeRO-1 과 ZeRO-2는 주로 역전파 단계에서만 통신이 필요한 반면, ZeRO-3는 순전파와 역전파 모두에서 통신이 발생하여 통신 부담이 가장 크다.

### Gradient Accumulation

ZeRO는 GPU 메모리 사용량을 줄이지만, 모델 파라미터와 옵티마이저 상태를 분산 저장하는 구조상 **단일 GPU에서 처리 가능한 배치 크기의 한계는 그대로 존재한다.** 이를 보완하기 위해 사용되는 것이 Gradient Accumulation으로, ZeRO와 함께 적용 시 대규모 배치 학습을 모사하면서도 메모리 효율을 극대화할 수 있다.

이 기법은 작은 배치를 여러 번 순차적으로 처리하면서 그라디언트를 누적한 뒤, 일정 횟수마다 `optimizer.step()`을 수행하는 방식으로, 논리적으로 더 큰 배치 크기로 학습한 것과 유사한 효과를 낸다. 이를 통해 메모리 한계 내에서도 학습 안정성과 성능을 향상시킬 수 있다.

DeepSpeed에서는 `gradient_accumulation_steps` 항목을 통해 이 기능을 쉽게 설정할 수 있으며, ZeRO의 메모리 절감 효과와 함께 사용하면 적은 GPU 자원으로도 효율적인 대규모 학습이 가능하다.

## 2.2 ZeRO Extensions

### ZeRO-Offload / ZeRO-Infinity

ZeRO는 GPU 메모리 절감을 위해 일부 모델 상태를 **CPU 메모리**나 **NVMe 스토리지**로 오프로드할 수 있다.

- **ZeRO-Offload**: 옵티마이저 상태를 CPU로 이동
- **ZeRO-Infinity**: 옵티마이저 상태뿐 아니라 파라미터와 그라디언트까지 NVMe로 오프로드

### DeepSpeed Inference

ZeRO의 원리를 확장한 **DeepSpeed Inference 엔진**은 학습뿐 아니라 추론에서도 효율을 높인다.

- 8bit/4bit 양자화 및 연산 최적화
- Token caching을 통한 추론 속도 향상
- 멀티-GPU 기반 대규모 모델 추론 지원
