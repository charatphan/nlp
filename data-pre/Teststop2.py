# Import library
import spacy

# Load the language model
nlp = spacy.load("en_core_web_sm")

# Process the text
text = "This is a sample sentence with some stop words"
doc = nlp(text)

# Remove stop words
filtered_tokens = [token.text for token in doc if not token.is_stop]

# Print the text excluding stop words
print(filtered_tokens)