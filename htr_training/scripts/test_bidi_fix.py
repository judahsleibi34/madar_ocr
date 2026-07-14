import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from lib.ctc_text_encoder import CTCTextEncoder

encoder = CTCTextEncoder("data/processed/vocab.json")

samples = [
    'للقائلين : "سلام على لبنان الخير والعافية" ان موطن العافية والخير باقٍ لأنهما يأبيان ان',
    'شخص منها سنة ١٩٨٧ والتي كانت هدفها ديني لخدمة الرب',
    'EAK1A-33',
    '4',
]

for text in samples:
    print("Original:      ", repr(text))
    encoded = encoder.encode(text)
    print("Encoded (ids):  ", encoded[:15].tolist(), "...")
    decoded = encoder.decode(encoded)
    print("Decoded:        ", repr(decoded))
    visual_ref = encoder.to_visual(text)
    print("Matches visual: ", decoded == visual_ref)
    print()