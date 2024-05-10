import pandas as pd

#pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows',None)
 
# reading csv file 
df = pd.read_csv("word1.csv")
from pythainlp.corpus.common import thai_stopwords
thai_stopwords = list(thai_stopwords())
#print(thai_stopwords)

#word segmenttation
from pythainlp import word_tokenize
import deepcut
def text_process(A):
    final= "".join(u for u in A if u not in ("?",".",";",":","!","ๆ","ฯ"))
    #final= deepcut.tokenize(final)
    final= word_tokenize(final)
    final= "|".join(word for word in final)
    #stop_words
    final= " ".join(word for word in final.split() if word.lower not in thai_stopwords)
    return final

#สร้าง column ชื่อว่า text_tokens เก็บค่าAที่ทำการ word_tokenize
df['text_tokens'] = df["name"].apply(text_process)
#print(df['text_tokens'])

def tokenize_and_store(text):
    tokens= "".join(u for u in text if u not in ("?",".",";",":","!","ๆ","ฯ","|"," "))
    tokens = deepcut.tokenize(tokens)  # เลือก engine ตามความต้องการ
    return tokens
#df['text_tokens'] = df["name"].apply(text_process).apply(tokenize_and_store)

from pythainlp.tag import pos_tag

def text_pos(text):
    pos = pos_tag(text,corpus="orchid_ud")
    return pos

df['pos'] = df["name"].apply(text_process).apply(tokenize_and_store).apply(text_pos)
df.to_csv('file.csv',index=False)
#print(df["text_tokens"][0])
#print(df["pos"][0])
#x = pd.DataFrame(text_pos(df['text_tokens'][0]))
#print(x)