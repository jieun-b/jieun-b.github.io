---
title: "PyTorch DataLoader"
date: 2025-08-01 02:00:00 +0900
layout: post
categories: [공부, Pytorch]
tags: [PyTorch, DataLoader, Dataset, Sampler]
---

# 1. DataLoader 개요

PyTorch의 DataLoader는 데이터셋을 반복(iteration) 가능한 형태로 불러오는 클래스이며, 학습 과정에서 **배치 처리, 셔플링, 병렬 로딩** 등을 지원한다.

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

# 2. Dataset

PyTorch에서는 두 가지 타입의 데이터셋을 지원한다.

- **Map-style dataset**
- **Iterable-style dataset**

(이 글에서는 Map-style만 다룬다.)

## Map-style datasets

Map-style dataset은 인덱스나 키를 통해 개별 데이터 샘플에 접근할 수 있는 구조로, `__getitem__()`과 `__len__()` 메서드를 구현해야 한다.

```python
class CustomDataset(Dataset):
    def __getitem__(self, idx):
        return data[idx]
    def __len__(self):
        return len(data)
```

# 3. Sampling

| 항목 | Sampler | BatchSampler |
| --- | --- | --- |
| 역할 | 데이터셋의 **개별 인덱스 순서** 결정 | 샘플러를 감싸 **배치 단위 인덱스 묶음** 생성 |
| 입력 인자 | 데이터셋 | `sampler`, `batch_size`, `drop_last` |
| 반환 단위 | int (단일 인덱스) | list[int] (배치 인덱스) |
| DataLoader 인자 | `sampler` | `batch_sampler` |

## 3.1 Sampler

Sampler는 **데이터의 인덱스 순서를 정의**하는 클래스이다.

DataLoader는 내부적으로 이 샘플러를 사용해 데이터셋에서 어떤 순서로 샘플을 불러올지를 결정한다.

- `shuffle=True`일 경우 PyTorch는 자동으로 `RandomSampler`를 사용한다.
- 직접 순서를 제어하고 싶다면 `sampler` 인자에 사용자 정의 Sampler를 지정하며, 이때는 `shuffle=False`로 설정해야 한다.

```python
from torch.utils.data import DataLoader, SequentialSampler

data = range(10)
sampler = SequentialSampler(data)  # 0~9 순서대로 반환
loader = DataLoader(data, batch_size=3, sampler=sampler)

for batch in loader:
    print(batch)
```

## 3.2 BatchSampler

BatchSampler는 Sampler를 감싸 **batch 단위의 인덱스 리스트**를 반환하는 클래스이다.

- `batch_size`, `drop_last` 옵션을 사용하면 `sampler`를 기반으로 자동 생성할 수 있다.
- 직접 제어하고 싶을 경우, `batch_sampler` 인자에 BatchSampler 객체를 넘기면 된다.

```python
from torch.utils.data import BatchSampler, SequentialSampler

sampler = SequentialSampler(range(10))
batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)

for batch in batch_sampler:
    print(batch)
```

# 4. Batch 구성

## collate_fn

`collate_fn`은 **DataLoader가 하나의 배치를 구성할 때**, 여러 샘플을 어떻게 묶을지를 정의하는 함수이다.

- `DataLoader`는 각 배치마다 `List[dataset[i]]` 형태의 샘플 리스트를 `collate_fn`에 전달한다.
- 기본적으로 `default_collate` 함수를 사용하며, 텐서나 넘파이 배열, 숫자, 딕셔너리 등은 자동으로 묶인다.
- 그러나 입력 형식이 다양하거나 길이가 일정하지 않은 경우(예: PIL 이미지, 다른 길이의 시퀀스 등)에는 **사용자 정의** `collate_fn`을 구현하여 원하는 방식으로 배치를 구성할 수 있다.
    
    ```python
    # 이미지 리스트와 라벨 텐서를 하나의 배치로 묶는 collate_fn 예시
    def collate_fn(batch):
        return {
            "images": [item["image"] for item in batch],
            "labels": torch.stack([item["label"] for item in batch])
        }
    ```
    

# 5. 주요 인자 정리

| 인자 | 설명 | 예시 단위 | 함께 쓰면 안 되는 것 |
| --- | --- | --- | --- |
| `sampler` | 개별 샘플 인덱스 순서를 정의하는 반복 가능한 객체 | `[0, 3, 1, 2]` | `shuffle`, `batch_sampler` |
| `batch_sampler` | 배치 단위의 인덱스 리스트를 생성하는 반복 가능한 객체 (`sampler`를 감쌈) | `[[0, 3, 7], [4, 5, 2]]` | `sampler`, `shuffle`, `batch_size` |
| `batch_size` | 한 배치에 포함할 샘플 개수 | `batch_size=4` | `batch_sampler` |
| `shuffle` | 인덱스를 무작위로 섞어주는 기능 (`sampler`가 없을 때 `RandomSampler`를 내부적으로 사용) | `True` (랜덤 순서) | `sampler`, `batch_sampler` |
| `collate_fn` | `List[dataset[i]]`를 받아 하나의 배치 텐서/딕트 구조로 변환 | `List[Sample] → Tensor/Dict` | 항상 사용 가능 |