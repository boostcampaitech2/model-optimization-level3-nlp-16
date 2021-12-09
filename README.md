# model-optimization-level3-nlp-16

## Table of Contents

1. [Introduction](#introduction)
2. [Project Detail](#project-detail)
3. [How to Use](#how-to-use)

<br/>


# Introduction

## TEAM : NLPRIME

#### Members

|                            김아경                            |                            김현욱                            |                            김황대                            |                            박상류                            |                            정재현                            |                            최윤성                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/70522267?v=4)](https://github.com/EP000) | [![Avatar](https://avatars.githubusercontent.com/u/31470457?v=4)](https://github.com/powerwook) | [![Avatar](https://avatars.githubusercontent.com/u/59689327?v=4)](https://github.com/kimhwangdae) | [![Avatar](https://avatars.githubusercontent.com/u/60460317?v=4)](https://github.com/psrpsj) | [![Avatar](https://avatars.githubusercontent.com/u/13325436?v=4)](https://github.com/JHyunJung) | [![Avatar](https://avatars.githubusercontent.com/u/80210706?v=4)](https://github.com/choi-yunsung) |



## Project Overview

프로젝트 기간: 2021.11.22 - 2021.12.02 (2 Weeks)

#### 프로젝트 목표

- 분리수거 로봇에 가장 기초 기술인 쓰레기 분류기를 만들면서 실제로 로봇에 탑재될 만큼 작고 계산량이 적은 모델 만들기
- 기 개발된 모델을 AutoML 및 최적화 기법들을 이용하여 어느 정도 이상의 성능(time : 60s, f1 70.00)을 유지하며 작은 크기와 빠른 추론속도를 갖는 모델 만들기

#### Dataset

- Segmentation / Object detecion task 를 풀기 위해 제작된 COCO format의 재활용 쓰레기 데이터인 TACO 데이터
- 단순한 Classification 문제로 설정하기 위해 TACO 데이터셋의 Bounding box를 crop 한 데이터 사용

- 총 6개의 카테고리 (Metal, Paper, Paperpack, Plastic, Plasticbag, Styrofoam)로 구성
  - Train data: 20,851 장의 .jpg 이미지
  - Test data: 5,217 장의 .jpg 이미지 (private 2,611장, public 2,606장)

#### Evaluation

- 최종 score 는 F1 score 와 Inference time 을 통해 계산
  - F1 Score: 불균형 데이터에 사용되는 분류 성능 지표로, 대회 기준 모델의 F1 score에서 제출한 모델의 F1 score의 차이에 상수를 구한 뒤 sigmoid 함수 적용
  - Inference Time: 대회 기준 모델의 추론 시간 대비 제출한 모델의 추론 시간 계산

#### Final Score

- 23th / 38 teams

| Metric         | Public Score | Private Score |
| -------------- | ------------ | ------------- |
| F1 score       | 0.6897       | 0.6936        |
| Inference Time | 59.3327      | 59.3327       |
| 최종 Score     | 1.2388       | 1.2195        |


<br/>


# Project Detail

#### Data Augmentation

- Random Augmentation 을 통해 최적의 성능을 보이는 augmentation 기법 선정
- 적용 Augmentation 기법
- `AutoContrast`,`Contrast`,`Rotate`,`Equalize`,`Identity`,`Solarize`,`Color`,`Posterize`,`Contrast`,`Brightness`,`Sharpeness`,`ShearX`,`ShearY`,`TranslateX`,`TranslateY`,`CutOut`,`RandomHorizonatalFlip`

#### Neural Architecture Search (NAS)

- `Optuna` 를 활용해 Neural Architecture Search 진행
- MobileNetv2, MobileNetv3 에 사용된 모듈 사용
- `f1_score`,`params_nums`(parameter 개수), `mean_time` 을 objective의 평가지표로 사용하여 모델의 성능을 유지하면서 작은 크기와 빠른 추론 속도를 갖는 경량화 모델 탐색
- 최종제출 Model Architecture

  <img src="https://user-images.githubusercontent.com/80210706/145322846-a85d0e91-5735-429b-ad5f-2b0800b7393d.png"  width="300" height="600">


<br/>



# How-to-use

#### Install requirements

```bash
pip install -r requirements.txt
```

#### Dataset

- torchvision.dataset 의 ImageFolder 사용
- 본 Repository 에는 dataset이 포함되어 있지 않습니다.

#### Data Config / Model Config

- `code/config/data/taco.yaml` : 데이터셋과 학습에 관련된 파라미터 설정 파일
- `code/config/models/mobilenetv3.yaml` : 모델을 구성하는 설정 파일
  - input_channel : 입력으로 들어오는 채널 수
  - depth_multiple : 모델 반복 횟수의 곱셈 값
  - width_multiple : 채널 갯수의 곱셈 값
  - Backbone : 전체 모델을 구성하는 모듈을 포함하는 리스트

#### Model Optimization

- `tune.py` : Optuna 를 활용해 최적 모델구조와 하이퍼파라미터를 찾는 코드

  ```bash
  python tune.py
  ```

- `train.py` : tune.py를 통해 찾은 모델을 통해 학습하는 코드

  ```bash
  python train.py --model $(모델 파일 경로) --data $(데이터셋 파일 경로)
  ```
