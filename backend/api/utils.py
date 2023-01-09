import cv2
import numpy as np

class InferenceUtils:
    def __init__(self) -> None:
        pass 

    def read_image(self, image_file_path : str) -> np.ndarray:
        """Returns the resized form of the image

        Args:
            image_file_path (str): The path of the image file 

        Returns:
            np.ndarray: Numpy array representing the normalized resized form of the image
        """
        img_rgb = cv2.imread(image_file_path)
        img_grey = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_adapted = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 101, 9)
        img_stacked = np.repeat(img_adapted[...,None],3,axis=2)
        resized = cv2.resize(img_stacked, (224,224), interpolation=cv2.INTER_AREA)
        bg_img = 255 * np.ones(shape=(224,224,3))
        bg_img[0:224, 0:224,:] = resized
        bg_img /= 255
        bg_img = np.rollaxis(bg_img, 2, 0)  
        return bg_img
    
    def load_vocab_doc(self, filename : str) -> str:
        file = open(filename, 'r')
        text = file.read()
        file.close()
        text = text.replace('\n', '')
        return text 

    def word_for_id(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None