# dab

받아야할 데이터 크기가 크고 모델도 gpu에서 학습해야하니까 colab에서 꼭 학습


## 현재 repository 클론하기
1.googe drive에 이 repository를 저장할 폴더를 하나 만든다\
2.colab notebook을 새로 연다(순전히 repo clone용이다)\
3.colab에 아래 코드를 친다.\

'''python
from google.colab import drive
drive.mount('/content/drive')
%cd '/content/drive/MyDrive/저장하고 싶은 폴더'
!git clone https://github.com/turtle98/dab.git
'''

4.clone이 완료 되면 해당 colab notebook을 끄고, google drive를 다시 열어서 해당 폴더 안에 dab.ipynb 파일을 열어서 그대로 실행시키면 된다

