# 케라스: 파이썬 딥러닝 라이브러리

<img src='https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png' style='max-width: 600px; width: 90%;' />



## 케라스에 오신것을 환영합니다.

케라스는 Python으로 작성된 고수준의 신경망 API입니다. [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), 혹은 [Theano](https://github.com/Theano/Theano)와 함께 사용할 수 있습니다. 케라스는 신속한 실험을 가능하게 하기 위해 개발되었습니다. *아이디어가 결과물에 이르기까지 걸리는 시간을 최소화 하는것이 바로 좋은 연구의 핵심입니다.*

다음과 같은 딥러닝 라이브러리가 필요한 경우에 케라스를 사용하면 좋습니다. 

- 케라스의 특징인 사용자 친화성, 모듈성, 확장성을 바탕으로 딥러닝 프로토타입을 쉽고 빠르게 만들 수 있습니다. 
- 합성곱 신경망<sub>convolutional networks</sub>, 순환 신경망<sub>recurrent networks</sub>을 모두 지원하며, 이 둘을 자유롭게 조합하여 사용할 수 있습니다.
- 동일한 코드로 CPU와 GPU에서 실행할 수 있습니다. 

[Keras.io](https://keras.io)에서 문서를 찾아 볼 수 있습니다.

케라스는 Python 2.7-3.6과 호환됩니다.


------------------

## 다중 백엔드 케라스<sub>multi-backend Keras</sub>와 tf.keras
**TensorFlow 백엔드와 함께 다중 백엔드 케라스를 사용하는 사용자는 TensorFlow 2.0에 포함된 `tf.keras`로 전환 하기를 권장합니다.** `tf.keras`는 기존의 케라스에 비해 잘 관리되고 있으며, TensorFlow의 다른 기능들(즉시실행<sub>eager execution</sub>, 배포 지원<sub>distribution support</sub> 등)과도 잘 호환됩니다. 

Keras 2.2.5 버전은 케라스의 마지막 2.2.* 버전입니다. 2.2.5 버전은 TensorFlow 1과 Theano 및 CNTK를 지원하는 마지막 릴리스입니다. 

현재 릴리스는 Keras 2.3.0 버전입니다. API가 크게 변경되었으며 TensorFlow 2.0에 대한 지원이 추가되었습니다. 2.3.0 버전은 다중 백엔드 Keras의 마지막 주요 릴리스입니다. 다중 백엔드 케라스는 `tf.keras`로 대체되었습니다. 

다중 백엔드 케라스의 버그 수정은 것은 2020년 4 월까지만 마이너 릴리스를 통해 이루어질 것입니다. 

케라스의 미래에 대한 자세한 내용은 [케라스 회의 노트](http://bit.ly/keras-meeting-notes)를 참고하십시오.

------------------


## 케라스의 주요 철학

- __사용자 친화성.__ 사용자 친화성. 케라스는 기계가 아닌 사람을 위해 디자인된 API입니다. 무엇보다 사용자의 경험을 우선으로 두고 있습니다. 케라스는 다음의 원칙을 지킴으로써, 사용자의 정보에 대한 인지를 원활하게 합니다. 
  - 일관성있고 간결한 API를 제공합니다.
  - 일반적으로 사용되는 기능에서 사용자의 작업을 최소화합니다.
  - 사용자 오류시 명확하고 실행 가능한 피드백을 제공합니다.

- __모듈성.__ 모델은, 최소한의 제한으로 다양한 조합이 가능한 독립적이며 완전히 변경가능한 모듈의 시퀀스 혹은 그래프로 이해할 수 있습니다. 특히 신경망 층<sub>neural layers</sub>, 손실 함수<sub>cost functions</sub>, 최적화 함수<sub>optimizer</sub>, 최초값 설정 규칙<sub>initialization schemes</sub>, 활성화 함수<sub>activation functions</sub>, 정규화 규칙<sub>regularization schemes</sub>은 모두 독립적인 모듈로, 다양하게 조합하여 새로운 모델을 만들어 낼 수 있습니다.

- __확장성.__ 새로운 모듈을 (클래스나 함수의 형태로) 간단하게 추가할 수 있으며, 기존의 모듈이 풍부한 사례를 제공합니다.  모듈을 새로 생성하는 것이 쉬워지면, 사용자는 뛰어난 표현력을 발휘할 수 있게 됩니다. 이러한 확장성을 바탕으로 케라스가 고수준의 연구에 적합하다고 할 수 있습니다.

- __Python과의 호환.__ 케라스는 선언형의 모델 설정파일을 별도로 두지 않습니다. 모델은 Python 코드로 작성되며, 이로써 간결하고, 디버깅이 쉬우며, 확장성이 뛰어납니다.


------------------


## 시작하기: 케라스까지 30초

케라스에서 가장 중요한 데이터 구조는 __모델__ 입니다. 모델은 층<sub>layer</sub> 구성하는 방식입니다. 가장 간단한 모델인 [`Sequential`](https://keras.io/getting-started/sequential-model-guide) 모델은 층을 선형적으로 쌓습니다. 보다 복잡한 구조를 만드려면, [케라스 함수 API](https://keras.io/getting-started/functional-api-guide)를 사용하여 임의의 층 그래프를 생성할 수 있습니다.

`Sequential` 모델입니다.

```python
from keras.models import Sequential

model = Sequential()
```

`.add()`함수를 사용하여 쉽게 층을 쌓을 수 있습니다.

```python
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```

모델이 마음에 드신다면, `.compile()`함수를 사용하여 학습 과정에 대한 설정을 할 수 있습니다.

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

필요한 경우, 최적화 함수에 대한 설정을 추가로 할 수 있습니다. 사용자가 필요할 때에는 완전한 권한을 가질 수 있도록 하되, 간결성을 유지하는 것이 케라스의 주요 철학중 하나입니다 (완전한 권한을 가짐으로써 소스코드의 확장이 용이해집니다).
```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

배치<sub>batch</sub>의 형태로 학습 데이터에 대한 반복작업을 수행할 수 있습니다.

```python
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

혹은, 모델에 배치를 수동으로 전달할 수도 있습니다.

```python
model.train_on_batch(x_batch, y_batch)
```

코드 한 줄로 모델의 성능을 평가해 보십시오.

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

혹은 새로운 데이터에 대해서 예측 결과를 생성해 보십시오.

```python
classes = model.predict(x_test, batch_size=128)
```

이처럼 케라스를 활용하여 질문 응답 시스템, 이미지 분류 모델, 신경망 튜링 기계등의 어떤 모델이라도 빠르게 만들 수 있습니다. 딥러닝의 기본이 되는 아이디어가 간단한데 그 구현이 복잡할 이유가 어디 있겠습니까?

조금  심화된 케라스 튜토리얼을 원하신다면 다음을 참고하십시오.

- [Getting started with the Sequential model](https://keras.io/getting-started/sequential-model-guide)
- [Getting started with the functional API](https://keras.io/getting-started/functional-api-guide)

저장소<sub>repository</sub>의 [examples 폴더](https://github.com/keras-team/keras/tree/master/examples)에서는, 보다 고급 모델을 확인할 수 있습니다. 질문에 대답하는 메모리 신경망<sub>memory networks</sub>, 적층 LSTM<sub>stacked LSTM</sub>을 이용한 문서 생성 등이 있습니다.


------------------


## 설치

케라스를 설치하기 전에, 백엔드 엔진을 먼저 설치해야 합니다. TensorFlow, Theano, 혹은 CNTK 중 하나를 설치하십시오. TensorFlow 백엔드를 사용할 것을 추천드립니다.

- [TensorFlow 설치 설명서](https://www.tensorflow.org/install/).
- [Theano 설치 설명서](http://deeplearning.net/software/theano/install.html#install).
- [CNTK 설치 설명서](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine).

다음의 **선택적 종속**을 설치하는 것도 고려해 보십시오.

- [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (GPU에 케라스를 실행하실 경우 추천합다).
- HDF5 and [h5py](http://docs.h5py.org/en/latest/build.html) (디스크에 케라스 모델을 저장하실 경우 필요합니다).
- [graphviz](https://graphviz.gitlab.io/download/)와 [pydot](https://github.com/erocarrera/pydot) (모델 그래프를 시각화하는 [visualization utilities](https://keras.io/visualization/)에 사용됩니다).

이제 케라스를 설치하시면 됩니다. 케라스를 설치하는 방법은 두 가지가 있습니다.

**(1) PyPI에서 케라스 설치하기(추천)**

참고: 이 설치 단계는 사용자가 Linux 또는 Mac 환경에 있다고 가정합니다. Windows 사용자는 sudo를 빼고 아래의 명령을 실행해야 합니다.

```sh
sudo pip install keras
```

만약 virtualenv를 사용하신다면 sudo를 사용하는 것은 피하는 편이 좋습니다.

```sh
pip install keras
```

**(2) Github 소스를 통해 케라스 설치하기**

다른 방법으로, Github 소스를 통해 케라스를 설치하는 방법입니다.

먼저 `git`명령어를 사용하여 케라스를 clone하십시오.

```sh
git clone https://github.com/keras-team/keras.git
```

그리고 `cd`명령어를 사용하여 케라스 폴더로 이동한 후 install 명령을 실행하십시오.
```sh
cd keras
sudo python setup.py install
```

------------------


## 케라스 백엔드 설정

케라스는 텐서를 처리하는 라이브러리로 Tensorflow를 기본으로 사용합니다. 케라스 백엔드의 설정에 대해서는 [이 설명서](https://keras.io/backend/)를 따라주십시오.

------------------


## 지원하기

다음을 통해서 개발 논의에 참여하거나 질문을 할 수 있습니다.

- [케라스 Google 그룹](https://groups.google.com/forum/#!forum/keras-users).
- [케라스 Slack 채널](https://kerasteam.slack.com). [이 링크](https://keras-slack-autojoin.herokuapp.com/)에서 케라스 Slack 채널에 초대를 신청할 수 있습니다.

또한 **버그 리포트와 기능 요청**은 [GitHub issues](https://github.com/keras-team/keras/issues)에서만 작성항 수 있습니다. 작성하기 전에 [가이드라인](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md)을 반드시 확인해주십시오.


------------------


## 왜 케라스인가요?

케라스(κέρας)는 그리스어로 _뿔_ 이라는 뜻입니다. _Odyssey_ 에서 최초로 언급된, 고대 그리스와 라틴 문학의 신화적 존재에 대한 이야기로, 두 가지 꿈의 정령(_Oneiroi_, 단수 _Oneiros_) 중 하나는 상아문을 통해 땅으로 내려와 거짓된 환상으로 사람을 속이며, 다른 하나는 뿔을 통해 내려와 앞으로 벌어질 미래를 예언합니다. 이는 κέρας (뿔) / κραίνω (이뤄지다), 그리고 ἐλέφας (상아) / ἐλεφαίρομαι (속이다)에 대한 언어유희이기도 합니다.

케라스는 초기에 ONEIROS(Open-ended Neuro-Electronic Intelligent Robot Operating System)라는 프로젝트의 일환으로 개발되었습니다.

>_"Oneiroi are beyond our unravelling --who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ Homer, Odyssey 19. 562 ff (Shewring translation).

------------------
