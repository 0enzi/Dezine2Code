import os 
import torch 
import warnings
from tqdm import tqdm 
from typing import Dict, Optional
from torch.autograd import Variable

from api.utils import InferenceUtils
from sketch2code.inference.Compiler import Compiler
from sketch2code.host.model import *

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.preprocessing.text import Tokenizer


class InferenceEngine(InferenceUtils):
    """Inference engine for the DesignToCode API"""

    def __init__(self, model_weights_path : Optional[Dict[str, str]]=None, vocab_path : Optional[str] = None, device : Optional[str] = None, sequence_length : Optional[str] = None) -> None:
        super().__init__()

        # Loading all the NLP stuffs 

        self._vocab_path = vocab_path if vocab_path is not None else "sketch2code/vocabulary.vocab"
        self.total_sequence_length = sequence_length if sequence_length is not None else 99
        tokenizer = Tokenizer(filters='', split=" ", lower=False)
        tokenizer.fit_on_texts([self.load_vocab_doc(self._vocab_path)])
        self._tokenizer = tokenizer

        # Configuring all the PyTorch model variables 

        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self._model_weights_path = model_weights_path if model_weights_path is not None else {
            "encoder": "sketch2code/model_weights/encoder_resnet34_0.061650436371564865.pt",
            "decoder": "sketch2code/model_weights/decoder_resnet34_0.061650436371564865.pt"
        }
        self._encoder = torch.load(self._model_weights_path['encoder'])
        self._decoder = torch.load(self._model_weights_path['decoder'])
        self._decoder_hidden = self._decoder.init_hidden() 
        
        self._encoder.eval()
        self._decoder.eval()

        print("=> Encoder Loaded...")
        print("=> Decoder Loaded...")
        print("=> Tokenizer Loaded...")
        print("=> Total Sequence Length: {}".format(self.total_sequence_length))

    def predict(self, image_path : str, format : bool = False):
        start_text = '<START>'
        end_text = '<END>'
        predicted = '<START>'
        image = self.read_image(image_path)
        image = Variable(torch.FloatTensor([image]))

     
        for i in tqdm(range(self.total_sequence_length), total=self.total_sequence_length):
            sequence = self._tokenizer.texts_to_sequences([start_text])[0]
            decoder_input = Variable(torch.LongTensor(sequence)).view(1,-1)
            features = self._encoder(image)
            outputs, self._decoder_hidden = self._decoder(features, decoder_input, self._decoder_hidden)
            topv, topi = outputs.data.topk(1)
            ni = topi[0][0][0]
            word = self.word_for_id(ni, self._tokenizer)
            if word is None: continue 
            predicted += word + ' '
            start_text = word 
        word = word + end_text 
        
        if format:
            predicted = Compiler('default').compile(predicted.split())
        return predicted       


if __name__ == '__main__':
    img_path = "/home/anindya/Documents/Github/Dezine2Code/sketch2code/val/2BC033FD-F097-463B-98A8-C1C9CE50B478.png"
    website = InferenceEngine().predict(img_path)

    print("\n", website)