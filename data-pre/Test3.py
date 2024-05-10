import pandas as pd
 
# reading csv file 
df = pd.read_csv("word1.csv")
from pythainlp.corpus.common import thai_stopwords
thai_stopwords = list(thai_stopwords())
print(thai_stopwords)

#word segmenttation
from pythainlp import word_tokenize
def text_process(A):
    final= "".join(u for u in A if u not in ("?",".",";",":","!","ๆ","ฯ"))
    final= word_tokenize(final)
    final= " ".join(word for word in final)
    #stop_words
    final= " ".join(word for word in final.split() if word.lower not in thai_stopwords)
    return final
#สร้าง column ชื่อว่า text_tokens เก็บค่าAที่ทำการ word_tokenize
#df['text_tokens'] = df["name"].apply(text_process)
#df.to_csv('file.csv',index=False)
#print(df.head(1))

import deepcut
def text_process2(A):
    final= "".join(u for u in A if u not in ("?",".",";",":","!","ๆ","ฯ"))
    final = deepcut.tokenize(final)
    final= " ".join(word for word in final)
    #stop_words
    final= " ".join(word for word in final.split() if word.lower not in thai_stopwords)
    return final
#สร้าง column ชื่อว่า text_tokens เก็บค่าAที่ทำการ word_tokenize
#df['text_tokens'] = df["name"].apply(text_process)
#df.to_csv('file.csv',index=False)
#print(df.head(1))
