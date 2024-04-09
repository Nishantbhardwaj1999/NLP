from textblob import TextBlob
from gingerit.gingerit import GingerIt

class SpellCheckerModule:
    def __init__(self):
        self.spell_check = TextBlob("")  # Initialize TextBlob instance for spell checking
        self.grammar_check = GingerIt()  # Initialize GingerIt instance with API key

    def correct_spell(self, text):
        corrected_words = []  # List to store corrected words
        words = text.split()  # Split text into words

        for word in words:
            # Use TextBlob to correct each word and convert to string
            corrected_word = str(TextBlob(word).correct())
            corrected_words.append(corrected_word)

        # Join corrected words into a corrected sentence
        corrected_text = " ".join(corrected_words)
        return corrected_text

if __name__ == "__main__":
    # Replace 'YOUR_GINGER_API_KEY' with your actual Ginger API key
    obj = SpellCheckerModule()

    message = "my naame is nishant , appple , baaanana"

    # Perform spell checking
    corrected_message = obj.correct_spell(message)
    print("Corrected Spell Check:", corrected_message)