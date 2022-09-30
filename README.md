# Log_anomaly_detection
Anomaly detection with statistical labeling and LSTM model

+ DeepLog 모델 소개 및 특징
2017년에 Min et al.에 의해 제안된 모델로써, 시스템 로그를 자연어 시퀀스로 모델링하기 위해 LSTM(Long Short-Term Memory)을 활용한 심층 신경망 모델이다. 
모든 log entry들은 key 값과 parameter값으로 변환된다. Log Key값은 log bodies excluding specific values이고, parameter vector는 variable values와 현재 로그와 직전 로그 사이의 시간 차이이다. 두 개의 서로 다른 모델을 이용하여 detection stages를 구성하고 자동적으로 정상적인 수행의 로그 패턴을 학습한 다음, 정상적인 범주와 벗어난 로그 패턴이 들어온 경우 이상 징후를 탐지할 수 있다. 또한 DeepLog은 새롭게 변화하는 로그 패턴에 따라서 모델의 업데이트가 가능함으로 효과적인 워크플로우를 구성하는데 도움이 된다. 해당 논문에서 수행한 실험 평가에 따르면 DeepLog는 기존 데이터 마이닝 방법론에 기초한 기존의 로그 기반 이상 탐지 방법을 능가하는 것으로 나타났다.

<img width="777" alt="image" src="https://user-images.githubusercontent.com/67995592/193299760-60ed044b-0b90-456a-b822-5f939e282957.png">

+ DeepLog 이용한 연구목표
로그 이상치를 탐지 하기 위해서 지도 학습의 경우 전체적인 데이터에 대한 라벨이 필요하지만 로그 데이터 특성상 데이터의 불균형이 존재한다. 따라서 데이터의 특성과 분포를 통해 이상치를 탐지하는 비지도 학습이 장점을 가진다. DeepLog 모델은 로그 항목을 특정 구문과 문법 규칙을 따르는 시퀀스의 요소로 사용한다. 이러한 시퀀스 벡터에 기반하여 w = {mt−h, . . . ,mt−1}개의 로그 키가 input으로 주어졌을 때 가장 다음으로 나올 확률이 높은 로그 키 값들을 예측함과 더불어 훈련 데이터에 없는 알려지지 않은 이상 로그 값을 효율적으로 감지할 수 있다.


