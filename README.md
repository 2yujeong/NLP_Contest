# NLP Contest
2020 국어 정보 처리 시스템 경진 대회 참가  

### 목표
감성 말뭉치 분석 시스템(감성 분석 소프트웨어) 개발

### 프로젝트 개요
 본 프로젝트는 국어 말뭉치 자료를 활용하여 국어 정보 처리 시스템을 개발하고 이를 통한 국어 정보화의 확대를 목표로 한다. 네이버, 다음, 유튜브 등 컨텐츠에 대한 댓글 서비스를 제공하는 포털 사이트의 리뷰와 댓글을 활용하여 트렌디한 성질의 말뭉치를 구축하고 이를 머신러닝에 적용함으로써 시스템은 작성자의 감성(긍정 혹은 부정)을 판별한다. 시스템의 정확도와 활용성을 높이기 위해 데이터 전처리, 딥러닝 환경 연구, 다양한 분야의 데이터 추출 등 여러 방법을 적용하였으며 최종적으로 국어 말뭉치의 활용성을 높이고 국어 정보 처리 분야의 발전에 기여하고자 한다.

### 구현 환경
* 개발 환경 : Google Colaboratory
* 언어 : Python3.6 <br><br>
## 개발
### 데이터 전처리
 문장 분해를 위해 한글 자연어 처리기인 Okt를 사용하여 형태소를 추출하였다. Okt의 pos()함수는 각 형태소를 Noun, Foreign, Josa, Verb, Adjective, Number, Modifier, Punctuation 등으로 분류한다. 이 중 감성 분석에 유의미하다고 생각되는 명사(Noun)와 동사(Verb), 형용사(Adjective)만을 데이터에서 추출하였다. 부정확하게 추출된 데이터들을 분석해보니 맞춤법 검사기의 필요성이 느껴져 학습 데이터를 구성할 때 미리 맞춤법 검사를 진행하였다. 추출한 형태소의 글자수가 2 미만인 값들은 없앴고 이로 인해 생긴 missing value 역시 모두 제거하였다.  
 딥러닝 시 추출한 데이터들을 숫자형태로 넣어줘야 하므로 추출한 형태소들을 정수로 인코딩 때, 각 숫자들이 나타난 빈도에 따라 가중치를 주기 위해서 Natural Language Toolkit(nltk) 패키지를 사용하였다. 추출한 형태소 데이터들이 전체 데이터에서 몇 번 등장하는 지를 계산하여 그 등장 횟수로 데이터를 대응시키고 None값은 -1로 대응시킨다.
 
 ### 머신러닝 적용
 * 딥러닝
 * Gradient Boosting Classifier(gbc)

### 결과 분석 및 정확도 향상을 위한 방법 적용
* 실험 결과
  - 딥러닝의 경우 52\~56% 정도의 정확도를 보였고 Gradient Boosting Classifier의 경우 68\~62% 정도의 정확도를 보였다. 
  - Test 결과를 보면 짧은 문장일수록, 빈도 수가 낮은 단어로만 구성된 문장일수록 긍정, 부정 분석이 잘 안되는 것으로 분석된다.
* 정확도 향상을 위한 방법 적용
  - 비표준어, 은어, 유행어 사용의 빈도가 높은 성질의 유튜브 댓글 제외
  - 감성 판단이 애매한 형용사와 동사 제외, 명사만 적용
  - Tokenizer를 이용한 정수 인코딩
<br><br>
## 시스템 테스트 방법
### 실행에 필요한 파이썬 라이브러리
  * get_train_data.ipynb : requests, bs4, json, re, pandas, time, selenium
  * NLP_contest.ipynb : konlpy, pandas, keras, sklearn, layers, matplotlib, numpy
### 경로 변경
``` Python
# 학습 데이터 불러오기
file1 = pd.read_csv('/gdrive/My Drive/공모전_국어정보처리/daum_movie_reviews.csv')
file2 = pd.read_csv('/gdrive/My Drive/공모전_국어정보처리/naver_reviews.csv')
file3 = pd.read_csv('/gdrive/My Drive/공모전_국어정보처리/ratings.txt', delimiter = '\t')
```  
불러오려는 학습 데이터 파일의 경로를 실행하는 환경에서 파일이 존재하는 경로로 바꿔준다.  

### 시스템 테스트
``` Python
test = pd.read_csv('/gdrive/My Drive/test.csv') 

test_data = []
for review in test['document']:
  temp = [word for word in okt.nouns(review) if len(word) >= 2]
  test_data.append(temp)

test_data = tokenizer.texts_to_sequences(test_data)
test_data = pad_sequences(test_data, maxlen=15, 
                          truncating = 'post', padding='post', value = 0)

pred = gbc.predict(test_data)

result = pd.DataFrame({
    "document" : test["document"],
    "label" : pred
})

result.to_csv('/gdrive/My Drive/result.csv', index=False)
```
NLP_contest.ipynb의 맨 밑 셀을 보면 위와 같이 코딩되어 있다. test = pd.read_csv(‘’)에 테스트 파일이 존재하는 경로를 입력한 후 코드를 실행하면 result.to_csv(‘’)에서 입력한 경로에 결과 파일이 저장된다. input 파일(테스트 파일)의 형태는 대회 측에서 제공한 네이버 영화 리뷰 파일의 형태와 동일하게 ‘document’ 열에 리뷰가 하나씩 저장되어 있는 형태여야 한다.
