from .Tagger import GlossTagger
from esc.esc_pl_module import ESCModule 
from esc.esc_dataset import prepare_sent
from esc.utils.definitions_tokenizer import get_tokenizer
from esc.predict import predict_one_batch

import spacy

class ESCTagger(GlossTagger):
    def __init__(self, ckpt_path: str, device):
        self.model = ESCModule.load_from_checkpoint(ckpt_path).to(device)
        self.model.freeze()

        self.tokenizer = get_tokenizer('facebook/bart-large', False)

        self.nlp = spacy.load("en_core_web_sm")
    
    def predict(self, sent: str):
        batch, tokens = prepare_sent(sent, self.tokenizer, self.nlp)
        predicitons = predict_one_batch(self.model, batch)
        to_ret = {}
        for (token, d), pred in zip(tokens, predicitons):
            d['gloss'] = pred
            to_ret[token] = d
        return to_ret


