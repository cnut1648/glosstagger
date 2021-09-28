from typing import List


class GlossTagger:
    def predict(self, sent: str):
        raise NotImplementedError
    
    def batch_predict(self, sents: List[str]):
        raise NotImplementedError