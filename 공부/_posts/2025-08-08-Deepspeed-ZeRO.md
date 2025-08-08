---
title: "DeepSpeed와 ZeRO"
date: 2025-08-08 00:00:00 +0900
layout: post
tags: [DeepSpeed, ZeRO]
---
# 1. DeepSpeed란?

![그림1.gif](/assets/img/공부/2025-08-08/그림1.gif)

DeepSpeed는 Microsoft에서 개발한 PyTorch 기반의 오픈소스 라이브러리로, 대규모 모델 학습의 속도, 비용, 메모리 사용량, 사용 편의성을 크게 향상시킨다.

특히 1000억 개 이상의 파라미터를 가진 모델도 단일 또는 다수의 GPU에서 효율적으로 학습할 수 있도록 설계되었다.

DeepSpeed는 다양한 최적화 기법을 제공하며, 그 핵심 기술 중 하나가 바로 ZeRO(Zero Redundancy Optimizer) 이다.

# 2. ZeRO (Zero Redundancy Optimizer)

ZeRO는 대규모 분산 학습에서 발생하는 메모리 중복 문제를 해결하기 위해 설계된 메모리 최적화 기법이다.

기존의 Data Parallel 방식에서는 모든 GPU가 전체 모델 파라미터의 복사본을 유지하기 때문에, 모델 크기가 커질수록 메모리 낭비가 심해진다. 반면 ZeRO는 모델 상태(파라미터, 그라디언트, 옵티마이저 상태) 를 GPU 간에 분산 저장함으로써 이러한 중복을 제거한다.

또한, ZeRO는 동적인 통신 스케줄링을 통해 분산된 상태를 효율적으로 공유하며, 메모리 사용량 절감과 통신 비용 간의 균형을 유지한다.

## 2.1 ZeRO의 세 가지 단계

ZeRO는 기능 범위에 따라 세 가지 단계로 나뉜다. 단계가 높아질수록 더 많은 메모리를 절약할 수 있지만, 그에 따라 통신량도 증가한다.

| Stage | 분산 대상 | 설명 |
| --- | --- | --- |
| **ZeRO-1** | Gradients | 그라디언트를 GPU 간 분산 저장 |
| **ZeRO-2** | Gradients + Optimizer states | 옵티마이저 상태까지 분산 저장 |
| **ZeRO-3** | Gradients + Optimizer states + Parameters | 모델 파라미터까지 포함한 **전체 상태 분산** |

### 성능 요약

| 단계 | 메모리 절감 효과 | 통신량 증가 |
| --- | --- | --- |
| ZeRO-1 | 최대 4배 | 없음 |
| ZeRO-2 | 최대 8배 | 없음 |
| ZeRO-3 | 최대 Nd배 (GPU 수에 비례) | 약 50% 증가 |

ZeRO는 GPU 메모리 사용량을 획기적으로 줄일 수 있지만, 모델 상태를 GPU 간에 분산 저장하기 때문에 학습 중에는 필요한 정보를 다른 GPU에서 불러오는 통신이 발생한다.

ZeRO-1과 ZeRO-2는 주로 역전파(backward) 과정에서 통신이 일어나며, 비교적 통신량 증가가 적은 반면, ZeRO-3는 순전파(forward)와 역전파 모두에서 통신이 필요해 통신량이 약 50%까지 증가할 수 있다.

### Gradient Accumulation

ZeRO는 모델 파라미터와 옵티마이저 상태 등을 분산시켜 GPU 메모리를 크게 절약할 수 있지만, 여전히 한 번에 처리할 수 있는 배치 크기에는 제한이 있을 수 있다. 이를 보완하기 위해 자주 함께 사용되는 기법이 Gradient Accumulation이다.

이 기법은 작은 배치를 여러 번 순차적으로 처리하면서 그라디언트를 누적한 뒤, 일정 횟수마다 `optimizer.step()`을 수행하는 방식으로, 논리적으로 더 큰 배치 크기로 학습한 것과 유사한 효과를 낸다. 이를 통해 메모리 한계 내에서도 학습 안정성과 성능을 향상시킬 수 있다.

DeepSpeed에서는 `gradient_accumulation_steps` 항목을 통해 이 기능을 쉽게 설정할 수 있으며, ZeRO의 메모리 절감 효과와 함께 사용하면 적은 GPU 자원으로도 효율적인 대규모 학습이 가능하다.

## 2.2 기타 관련 기능

### ZeRO-Offload

ZeRO는 GPU 메모리 사용량을 줄이기 위해 일부 모델 상태를 **CPU 메모리** 또는 **디스크(NVMe)** 로 오프로드할 수 있다.

특히 **ZeRO-2와 함께 사용**하면, GPU 메모리가 제한적인 환경에서도 대규모 모델 학습이 가능하다.

- 옵티마이저 상태를 **CPU**로 오프로드 (ZeRO-Offload)
- 파라미터와 그라디언트를 **디스크**로 오프로드 (ZeRO-Infinity)

### DeepSpeed Inference

ZeRO는 학습뿐 아니라 **추론 최적화**에도 활용할 수 있다.

**DeepSpeed Inference**는 다음과 같은 기능을 통해 LLM 등 대규모 모델의 추론 속도와 효율을 개선한다:

- 8bit 및 4bit 파라미터 양자화
- 토큰 캐싱 (token caching)
- 멀티-GPU 추론 지원