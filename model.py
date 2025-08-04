from IndicTransToolkit.IndicTransToolkit import IndicProcessor
from collections import OrderedDict
import nltk
nltk.download('punkt')

config = {
    "young": "யங்",
    "park": "பார்க்",
    "movie": "மூவி",
    "gift": "கிஃப்ட்",
    "birthday": "பர்த்டே",
    "party": "பார்டி",
    "time": "நேரம்",
    "science": "சயன்ஸ்",
    "mathematics": "மத்தமெட்டிக்ஸ்",
    "assignment": "அசைன்மென்ட்",
    "deadline": "டெட்லைன்",
    "exhibition": "எக்ஸிபிஷன்",
    "renewable energy": "ரின்யூயபிள் எனர்ஜி",
    "attendance": "அட்டெண்டன்ஸ்",
    "performance": "பெர்ஃபார்மன்ஸ்",
    "portal": "போர்டல்",
    "grammar": "கிராமர்",
    "syllabus": "சிலபஸ்",
    "examination": "எக்ஸாமினேஷன்",
    "website": "வெப்சைட்",
    "screen time": "ஸ்க்ரீன் டைம்",
    "seminar": "செமினார்",
    "mental health": "மென்டல் ஹெல்த்",
    "awareness": "அவேர்னஸ்",
    "digitally": "டிஜிட்டலியாக",
    "information system": "இன்பர்மேஷன் சிஸ்டம்",
    "parent-teacher": "பேரண்ட்-டீச்சர்",
    "management": "மேனேஜ்மென்ட்",
    "homework": "ஹோம்வொர்க்",
    "report card": "ரிப்போர்ட் கார்டு"
}

config_english_words = list(config.keys())

# Use OrderedDict to preserve key order and indices
ordered_config = OrderedDict(config)

# Replace phrases starting with longest ones to avoid partial masking
sorted_keys = sorted(ordered_config.keys(), key=lambda x: -len(x))


key_to_index = {key: i for i, key in enumerate(config)}
# index_to_key = {i: key for key, i in key_to_index.items()}
value_to_index = {config[key]: i for i, key in enumerate(config)}
index_to_key = {i: config[key] for key, i in key_to_index.items()}


def mask_word(sent):
    lower_sent = sent.lower()
    replaced_sent = lower_sent
    for i, key in enumerate(sorted_keys):
        if key in replaced_sent:
            replaced_sent = replaced_sent.replace(key, f"<WORD{list(ordered_config).index(key)}>")
    return replaced_sent

def unmask(sent):
    unmasked_sent = sent
    for i, key in enumerate(sorted_keys):
        unmasked_sent = unmasked_sent.replace(f"<WORD{list(ordered_config).index(key)}>", config[key])
    return unmasked_sent


class Process:
    """
    here we set input_sententce to handle array of sentence
    """
    def __init__(self, input_sentence, src_lan="eng_Latn", tgt_lan="tam_Taml"):
        self.input_sentence = input_sentence
        self.src_lan = src_lan
        self.tgt_lan = tgt_lan
        self.ip = IndicProcessor(inference=True)

    def getBatch(self):
        masked_input_sentences = [ mask_word(sentence) for sentence in self.input_sentence]
        batch = self.ip.preprocess_batch(
            masked_input_sentences,
            src_lang=self.src_lan,
            tgt_lang=self.tgt_lan,
        )
        return batch

    def translate(self, generatedTokens):
        translations = self.ip.postprocess_batch(generatedTokens, lang=self.tgt_lan)
        result = []
        for sentence, translation in zip(self.input_sentence, translations):
            result.append({
                "src": sentence,
                "tgt": unmask(translation)
            })
        return result
