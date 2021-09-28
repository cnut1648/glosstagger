from .Tagger import GlossTagger
from spacy import load
from ewiser.spacy.disambiguate import Disambiguator
import re


class SpacyTagger(GlossTagger):
    def __init__(self, ckpt_path: str, device):
        wsd = Disambiguator(
            ckpt_path, 
            lang="en", batch_size=5, 
            save_wsd_details=False).to(device).eval()
        self.nlp = load("en_core_web_sm", disable=['ner', 'parser'])
        # enable spacy plugin
        wsd.enable(self.nlp, 'wsd')

    
    def read_paragraphs(self, it):
        doc = []
        for line in it:
            line = line.strip()
            line = re.sub(r'\s+', ' ', line)
            if not line and doc:
                yield "\n".join(doc)
                doc.clear()
            else:
                if line:
                    doc.append(line)
        if doc:
            yield "\n".join(doc)
        
    def annotate_and_print(self, it_par):
        SEP = "@#*"
        to_ret = {}
        for par in self.nlp.pipe(it_par, batch_size=5):
            for token in par:
                if token.text == '\n':
                    pass
                    # print()
                else:
                    new_string = token.text + SEP + token.lemma_ + SEP + token.pos_ + SEP
                    if token._.offset:
                        new_string += token._.offset
                        to_ret[token.text] = {
                            "lemma": token.lemma_,
                            "pos": token.pos_,
                            "gloss": token._.offset
                        }
        #             print(new_string, end=' ')
        #     print()
        #     print()
        # print(to_ret)
        return to_ret


    def predict(self, sent: str):
        paras = self.read_paragraphs([sent])
        return self.annotate_and_print(paras)



