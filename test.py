from tagger.SpacyTagger import SpacyTagger
from tagger.ESCTagger import ESCTagger
import torch
device = torch.device("cpu")

print("Input: I love research!")
print("EWISER")
tagger = SpacyTagger(
    ckpt_path="/home/jiashu/WSD/ewiser/ewiser.semcor+wngt.pt",
    device=device
)

print(tagger.predict("I love research!"))

print("ESC")
tagger = ESCTagger(
    ckpt_path="/home/jiashu/WSD/esc/escher_semcor_best.ckpt",
    device=device
)

print(tagger.predict("I love research!"))