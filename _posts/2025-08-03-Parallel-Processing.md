---
title: "Python 병렬 처리"
date: 2025-08-03 00:00:00 +0900
layout: post
categories: [공부, Python]
tags: [ParallelProcessing, Multiprocessing, Multithreading]
---
# 1. 병렬 처리란?

여러 작업을 동시에 실행하여 전체 처리 시간을 줄이는 기법이다.

- 하나의 작업을 여러 단위로 나누어 동시에 처리한다.
- I/O 요청 처리, 이미지 변환, 데이터 전처리, 수치 계산 등에 활용된다.

# 2. Thread와 Process

## 2.1 Thread (스레드)

- 프로세스 내에서 실행되는 **작은 실행 단위**이다.
- **메모리를 공유**하며, 실행 비용이 낮고 전환 속도가 빠르다.
- 공유 자원에 대한 **동기화 이슈**가 발생할 수 있다.

## 2.1 Process (프로세스)

- 실행 중인 **독립된 프로그램 단위**이다.
- **고유한 메모리 공간**을 사용하여 다른 프로세스와 메모리를 공유하지 않는다.
- 안정성이 높지만 **전환 및 통신 비용이 크다**.

# 3. 병렬 처리 방식

## 3.1 멀티스레딩 (Multi-threading)

- 하나의 프로세스에서 **여러 스레드**를 동시에 실행하는 방식이다.
- **메모리 공간을 공유**하여 자원 접근이 빠르다.
- Python에서는 GIL(Global Interpreter Lock)로 인해 **CPU 연산에서는 효과가 미미하다**.
- **I/O 작업**에 적합하다 (예: 파일 읽기/쓰기, 웹 요청 등).

```python
from concurrent.futures import ThreadPoolExecutor

def task(x):
    return x * 2

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(task, [1, 2, 3, 4]))
```

## 3.2 멀티프로세싱 (Multi-processing)

- 여러 개의 **독립된 프로세스**를 생성하여 병렬로 작업을 수행하는 방식이다.
- 각 프로세스는 메모리를 분리하여 사용하므로 **GIL의 영향을 받지 않는다**.
- **CPU 중심 연산**에 적합하다 (예: 이미지 처리, 수치 계산 등).

```python
from concurrent.futures import ProcessPoolExecutor

def task(x):
    return x * x

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(task, [1, 2, 3, 4]))
```

## 3.3 비동기 처리 (AsyncIO)

- **이벤트 루프 기반**의 비동기 프로그래밍 모델이다.
- 싱글 스레드에서 실행되지만, **I/O 대기 시간 동안 다른 작업을 수행할 수 있다**.
- 대량의 **네트워크 요청**, **웹 크롤링**, **API 호출**에 적합하다.

```python
import asyncio

async def task(x):
    await asyncio.sleep(1)
    return x * 2

async def main():
    results = await asyncio.gather(*(task(i) for i in range(4)))
    print(results)

asyncio.run(main())
```

## 3.4 GPU 병렬 처리

- 수천 개의 코어를 가진 **GPU를 이용한 병렬 연산** 방식이다.
- 대규모 행렬 연산, 딥러닝 학습 등에 사용된다.
- PyTorch, TensorFlow, CUDA 등을 통해 구현할 수 있다.

# 4. 병렬 처리 도구

| 도구 | 설명 | 적합한 작업 |
| --- | --- | --- |
| `ThreadPoolExecutor` | 여러 스레드를 동시에 실행, GIL 영향 있음 | 파일 I/O, API 요청 |
| `ProcessPoolExecutor` | 여러 프로세스를 실행, GIL 우회 가능 | 이미지 처리, 수치 연산 |
| `asyncio` | 이벤트 루프 기반 비동기 처리 | 수천 개의 API 병렬 호출 |
| `multiprocessing` | 저수준 프로세스 직접 제어 가능 | 병렬 연산 구현 필요 시 |

## Thread vs Process 주요 비교

| 항목 | Thread 기반 | Process 기반 |
| --- | --- | --- |
| 실행 단위 | 스레드 (공유 메모리) | 프로세스 (독립 메모리) |
| GIL 영향 | 있음 | 없음 |
| 적합 작업 | 파일 I/O, API 호출 | 이미지 처리, 행렬 연산 |
| 속도 특성 | 빠른 전환, 낮은 오버헤드 | 초기화 비용 큼, 병렬 처리 우수 |
| 메모리/자원 | 적게 사용 | 메모리 더 사용함 |
| 통신 비용 | 낮음 | 높음 |

# 5. Global Interpreter Lock (GIL)

GIL(Global Interpreter Lock)은 CPython 인터프리터에서 동시에 하나의 스레드만 실행할 수 있도록 제한하는 전역 락이다.

- Python 멀티스레딩이 **실제로는 병렬로 실행되지 않는 이유**이다.
- **I/O 중심 작업**에서는 병목이 거의 없다.
- **CPU 중심 작업**에서는 GIL로 인해 병렬 처리가 제대로 이루어지지 않는다.

### GIL의 우회 방법

- 멀티프로세싱을 사용하면 각 프로세스는 GIL을 독립적으로 가진다.
- NumPy, Cython, PyTorch 등 **C로 구현된 연산 모듈**을 사용하면 성능을 개선할 수 있다.
- PyPy, Jython 등 다른 인터프리터를 사용하는 것도 방법이다.