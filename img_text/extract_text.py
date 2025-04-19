from PIL import Image
import pytesseract
import nltk
import ssl
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import words
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


class ComicTextExtractor:

    def init_constants(self):
        self.lemmatizer = WordNetLemmatizer()

        self.contraction_map = {
            "'m": "am",
            "'ve": "have",
            "'re": "are",
            "'ll": "will",
            "'d": "would",  # Or could be "had", depending on context
            "'s": "is",     # Can also mean "has" or "does"
            "n't": "not",
            "'t": "not",    # Sometimes used for negation, e.g., "ca'n't"
        }

        self.english_vocab = set(w.lower() for w in words.words())


    def is_valid_token(self, word):
        if word in self.english_vocab or word in '"\'.?!,':
            return True
        
        elif self.lemmatizer.lemmatize(word, pos=wordnet.NOUN) in self.english_vocab:
            return True
        
        return False


    def __init__(self, download_nltk = True): 
        #Enable download for first time. Pass False afterwards
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        if(download_nltk):
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('words') 
            nltk.download('wordnet')
            nltk.download('omw-1.4')

        self.init_constants()


    def extract_page_text(self, image_path):
        image = Image.open(image_path)

        # Use pytesseract to do OCR on the image
        extracted_text = pytesseract.image_to_string(image)

        sentences = []

        for sentence in sent_tokenize(extracted_text.lower()):
            words = []
            for word in word_tokenize(sentence):

                # print(word,end='|')
                if self.is_valid_token(word):
                    words.append(word)

                elif word in self.contraction_map:
                    words.append(self.contraction_map[word])

            sentences.append(" ".join(words))

        return sentences


if __name__ == '__main__':
    image_path = 'img_samples/page_34.jpg'

    cte = ComicTextExtractor(False)

    text = cte.extract_page_text(image_path)

    print(text)

