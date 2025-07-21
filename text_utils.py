# IPA Phonemizer: https://github.com/bootphon/phonemizer
import os
import pandas as pd
# _pad = "$"

# # Define common punctuation marks
# _punctuation = ';:,.!?¡¿—…"«»"" |ɟ'

# # Define the basic Hindi (Devanagari) characters plus additional ones
# _letters_hindi = 'अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञABCDEFUVWXYZabcdefghijklmnopqrstuvwxyzUऍऑऎऔॐँंःऄअाि ीुूृॄेैोौ्॒॑॓॔ॕॖॗॠॡॢॣ'
# # _numbers = '०१२३४५६७८९012345678'
# # Define a curated list of IPA symbols particularly useful for Hindi
# _letters_ipa_hindi = "ɑɐəɪiːʊuːɛeːɔoːpbt dkgpʰbʰtʰdʰkʰgʰt͡ʃd͡ʒt͡ʃʰd͡ʒʰʈɖʈʰɖʱmnŋɳfsʃhzʒʋlɾɽjɽʱrxɣqɦˈːβθçʔχʂɱʋɭɻɨɵœøɒæɤʌɯʍɕʑɬɮʎʝ"

# # Combine all symbols into a single list
# symbols = [_pad] + list(_punctuation) + list(_letters_hindi) + list(_letters_ipa_hindi) 

# # Create a dictionary mapping each symbol to its index
# dicts = {}
# # print(symbols)
# # print(len(symbols))
# # print(len(set(symbols)))
# symbols = set(symbols)
# cnt = 0
# for i in symbols:
#     dicts[i] = cnt
#     cnt+=1

# # DEFAULT_DICT_PATH = os.path.join('/home/user/voice/PL-BERT/word_index_dict.txt')
# # def load_dict(dict_path=DEFAULT_DICT_PATH):
# #     df = pd.read_csv(dict_path, header=None, dtype={0: str, 1: int})
# #     word_index_dict = {word.strip('"'): index for word, index in zip(df[0], df[1])}
# #     return word_index_dict
# # dicts = load_dict(DEFAULT_DICT_PATH)
# # letters = list(_letters) + list(_letters_ipa)



# class TextCleaner:
#     def __init__(self, dummy=None):
#         self.word_index_dictionary = dicts

#         print("Loaded symbols:", len(dicts))

#     def __call__(self, text):
#         indexes = []
#         for char in text:
#             try:
#                 idx  = self.word_index_dictionary[char]
#                 indexes.append(idx)
#             except KeyError:
#                 # print(text)
#                 pass
#         return indexes

_pad = "$"
_punctuation = ';:,.!?¡¿—…"“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "õũɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓1ãĩẽ'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        print(len(dicts))
        self.lis = []
        self.file_path = "missing_char.txt"
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                ### chnages ###
                # self.lis.append(char)
                
                
                # self.lis = set(self.lis)
                # with open(self.file_path, 'w') as file:
                #     for item in self.lis:
                #         file.write(item + ' ,')
                ### end ####
                print(text)
        return indexes
