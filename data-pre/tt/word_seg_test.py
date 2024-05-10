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
    final= "".join(u for u in A if u not in ("?",".",";",":","!","ๆ","ฯ"," "))
    final= word_tokenize(final)
    final= "|".join(word for word in final)
    #stop_words
    final= " ".join(word for word in final.split() if word.lower not in thai_stopwords)
    return final
def text_process2(A):
    final= "".join(u for u in A if u not in ("?",".",";",":","!","ๆ","ฯ"," "))
    final= deepcut.tokenize(final)
    final= "|".join(word for word in final)
    #stop_words
    final= " ".join(word for word in final.split() if word.lower not in thai_stopwords)
    return final
df['text_tokens'] = df["name"].apply(text_process)
df['deepcut'] = df["name"].apply(text_process2)
def count_vertical_bar(text):
    text= "".join(u for u in text if u not in ("?",".",";",":","!","ๆ","ฯ"))
    count = 0
    for char in text:
        if char == '|':
            count += 1
    return str(count)

#df ['Number of cuts token'][0]= count_vertical_bar(df['text_tokens'][0])
#df ['Number of cut deepcut'][0] = count_vertical_bar(df['deepcut'][0])
#df ['Number of cuts token'][1]= count_vertical_bar(df['text_tokens'][1])
#df ['Number of cut deepcut'][1] = count_vertical_bar(df['deepcut'][1])
#df ['Number of cuts token'][2]= count_vertical_bar(df['text_tokens'][2])
#df ['Number of cut deepcut'][2] = count_vertical_bar(df['deepcut'][2])
#df ['Number of cuts token'][3]= count_vertical_bar(df['text_tokens'][3])
#df ['Number of cut deepcut'][3] = count_vertical_bar(df['deepcut'][3])
#df ['Number of cuts token'][4]= count_vertical_bar(df['text_tokens'][4])
#df ['Number of cut deepcut'][4] = count_vertical_bar(df['deepcut'][4])


#df.to_csv('file.csv',index=False)
df1 = df[['Number of cuts token','Number of cut deepcut']]
print(df)
print(df1)