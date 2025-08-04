---
title: "PyTorch DataLoader 이해"
date: 2025-08-01 02:00:00 +0900
layout: post
tags: [PyTorch, DataLoader, Dataset, Sampler]
---
# 1. DataLoader

DataLoader는 데이터셋에 대한 반복 가능 객체를 나타내며, 아래와 같은 인자를 받는다.

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

# 2. Dataset

PyTorch에서는 Map-style datasets과 Iterable-style datasets 두 가지 타입의 데이터셋을 지원한다. 여기서는 Map-style datasets만 다룬다.

### Map-style datasets

Map-style dataset은 인덱스나 키를 통해 개별 데이터 샘플에 접근할 수 있는 형태의 데이터셋이며, `__getitem__()`과 `__len__()` 프로토콜을 구현해야 한다.

예를 들어, `dataset[idx]`와 같이 사용하여 idx번째 데이터에 접근할 수 있다.

# 3. Sampler

Sampler는 데이터 로딩에 사용되는 인덱스/키의 순서를 지정하는 데 사용된다. `shuffle=True`로 설정하면 PyTorch는 내부적으로 `RandomSampler`를 사용한다. 직접 샘플러를 지정하고 싶을 경우, `sampler` 인자에 사용자 정의 `Sampler` 객체를 넘겨주면 되며, 이때는 `shuffle=False`로 설정해야 한다.

# 4. BatchSampler

BatchSampler는 한 번에 하나의 배치 인덱스 리스트를 생성한다.

`batch_size`, `drop_last` 옵션을 활용해 `sampler` 기반으로 자동 생성할 수 있으며, 직접 지정하고 싶을 경우, `batch_sampler` 인자에 BatchSampler 객체를 넘겨주면 된다.

# 5. collate_fn

`collate_fn`은 하나의 배치를 구성할 때, 샘플 리스트를 어떻게 묶을지를 정의하는 함수이다. 

DataLoader는 `collate_fn`에 `List[dataset[i]]`를 전달하며, 기본적으로 `default_collate`를 사용한다.

사용자가 직접 정의하면, 다양한 입력 형식(PIL 이미지, 서로 다른 길이의 시퀀스 등)에 대해 맞춤형 처리가 가능하다.

예를 들어, 다음과 같이 PIL 이미지는 리스트로 유지하고, 라벨은 텐서로 묶는 방식으로 정의할 수 있다.

```python
def collate_fn(batch):
    return {
        "images": [item["image"] for item in batch],
        "labels": torch.stack([item["label"] for item in batch])
    }
```

# 6. 주요 인자 정리

| 인자 | 설명 | 예시 단위 | 함께 쓰면 안 되는 것 |
| --- | --- | --- | --- |
| `sampler` | 개별 샘플 인덱스 순서를 정의하는 반복 가능한 객체 | `[0, 3, 1, 2]` | `shuffle`, `batch_sampler` |
| `batch_sampler` | 배치 단위의 인덱스 리스트를 생성하는 반복 가능한 객체 | `[[0, 3, 7], [4, 5, 2]]` | `sampler`, `shuffle`, `batch_size` |
| `batch_size` | 몇 개씩 자동으로 묶을지 지정 | `batch_size=4` | `batch_sampler` |
| `shuffle` | 인덱스를 무작위로 섞어주는 기능 (`RandomSampler`를 내부적으로 사용) | `True` (랜덤 순서) | `sampler`, `batch_sampler` |
| `collate_fn` | `List[dataset[i]]`를 받아 하나의 배치 텐서/딕트 구조로 변환 | `List[Sample] → Tensor/Dict` | 항상 사용 가능 |