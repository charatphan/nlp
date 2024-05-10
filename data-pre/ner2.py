import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag
from pythainlp.tag.named_entity import ThaiNameTagger

# กำหนดข้อความที่ต้องการตรวจจับ named entities
text = "วันที่ 10 กุมภาพันธ์ 2024 นี้ เมืองไทยเริ่มมีอากาศหนาวลง โรงเรียนสวนกุหลาบอยู่ในตำบลแม่กลอง อำเภอเมือง จังหวัดเชียงใหม่"

# ตัดคำและตำแหน่งภาษาไทย
words = word_tokenize(text)
pos_tags = pos_tag(words)

# หา named entities
ner_tagger = ThaiNameTagger()
named_entities = ner_tagger.get_ner(text)

print("คำและ POS tags:", list(zip(words, pos_tags)))
print("Named entities:", named_entities)