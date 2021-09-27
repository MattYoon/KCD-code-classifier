import re
from konlpy.tag import Komoran
from word2word import Word2word


class Preprocessor:
    def __init__(self):
        self.komoran = Komoran()
        self.ko2en = Word2word('ko', 'en')
        self.junks = ['&nbsp;', '&lt;', '&gt;', '&amp;', '&quot;', '&apos;']

    def clean_text(self, x):
        for junk in self.junks:
            x = x.replace(junk, '')  # HTML 엔티티 및 탭 제거
        x = re.sub(r'(\n)\1+', r'\n', x)  # 연속된 줄바꿈 제거
        x = x.replace(' \n', '\n')  # 줄바꿈 전 공백 제거
        x = x.replace('\n', '. ')  # 줄바꿈을 마침표로 변경
        x = x.replace('\t', ' ')  # 줄바꿈을 마침표로 변경
        x = re.sub(r'[^a-zA-Z가-힣0-9 /+-,.]', ' ', x)  # 영문, 한글, 숫자 및 특정 기호 외 제거
        x = x.replace(' .', '.')  # 마침표 전 공백 제거
        x = re.sub(r'(\.)\1+', r'. ', x)  # 연속된 마침표 제거
        x = x.replace(',.', ',')
        x = ' '.join(x.split())  # 연속된 공백 제거
        return x

    def translate_noun(self, x):
        noun = self.komoran.nouns(x)
        if noun:
            try:
                translated_noun = self.ko2en(noun[0], n_best=1)
                return translated_noun[0]
            except KeyError:  # word2word에 없는 단어일 경우
                return None
        return None

    def extract_ko_eng(self, x):
        tokens = x.split()
        ko, eng = [], []
        for token in tokens:
            if re.search(r'[a-zA-Z]', token):  # 알파벳이 등장하는 토큰
                # 영문 및 / 외 제거, word2vec에 통과시킬 토큰들
                eng_token = re.sub(r'[^a-zA-Z/+-]', ' ', token)
                eng_token = eng_token.lower()
                eng += eng_token.split()
            elif re.search(r'[가-힣]', token):  # 한글 토큰
                ko.append(token)
                translated = self.translate_noun(token)  # 명사일 경우 번역 시도
                if translated:
                    eng.append(translated.lower())
            else:  # 기타 기호 토큰
                ko.append(token)
        return ' '.join(ko), ' '.join(eng)

    def __call__(self, x):
        x = self.clean_text(x)
        ko, eng = self.extract_ko_eng(x)
        return {'ko': ko, 'eng': eng}
