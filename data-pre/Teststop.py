import spacy

# Load pre-trained model for Thai
nlp = spacy.load("en_core_web_sm")

# Function to extract places and names from text
def extract_entities(text):
    doc = nlp(text)

    places = []
    names = []

    for ent in doc.ents:
        # Check if the entity is a location or a person
        if ent.label_ == "LOC":
            places.append(ent.text)
        elif ent.label_ == "PER":
            names.append(ent.text)

    return places, names

# Remove stop words from a list of tokens
def remove_stopwords(tokens, stop_words):
    return [token for token in tokens if token.lower() not in stop_words]

# Example text
text = "ด้านชีวิตในวงการบันเทิง ไต้ฝุ่น กนกฉัตร เข้าประกวด เคพีเอ็น อวอร์ด ครั้งที่ 21 เมื่อปี 2554 สถานที่: กรุงเทพมหานคร คว้ารางวัลนักร้องยอดเยี่ยมแห่งประเทศไทย อันดับที่ 2 หลังจากนั้นเขาได้มีผลงานเพลง รวมถึงม๊โอกาสได้เข้าวงการบันเทิงเต็มตัว มีผลงานการแสดงเรื่อยมา จนถึงปัจจุบัน"

# Define stop words in Thai
stop_words = ["ด้าน", "ใน", "ไต้ฝุ่น", "กนกฉัตร", "เข้า", "ประกวด", "ครั้ง", "ที่", "สถานที่", "กรุงเทพมหานคร", "คว้า", "รางวัล", "ยอดเยี่ยม", "แห่ง", "ประเทศไทย", "อันดับ", "หลังจาก", "เขา", "ได้", "มี", "ผลงาน", "รวมถึง", "ม๊โอกาส", "ได้", "เข้า", "วงการ", "บันเทิง", "เต็มตัว", "มี", "ผลงาน", "การ", "แสดง", "เรื่อย", "มา", "จน", "ถึง", "ปัจจุบัน"]

# Extract entities
places, names = extract_entities(text)

# Remove stop words
places = remove_stopwords(places, stop_words)
names = remove_stopwords(names, stop_words)

print("สถานที่:", places)
print("ชื่อคน:", names)