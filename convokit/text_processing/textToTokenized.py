from convokit.model import Corpus
from .textProcessor import TextProcessor
from typing import Callable, Optional 

# join tokens from existing parses
join_tokens = lambda p: " ".join([" ".join([tok['tok'] for tok in sent['toks']]) for sent in p])

class TextToTokenized(TextProcessor):
    """
    Transformer that formats utterance texts as space-separated (tokenized) text. Intended especially for languages which do not include spaces between words.

    :param input_field: name of attribute to use as input. This attribute must point to a string, and defaults to utterance.text.
    :param input_filter: a boolean function of signature `input_filter(utterance, aux_input)`.
        Text segmentation will only be applied to utterances where `input_filter` returns `True`.
        By default, will always return `True`, meaning that all utterances will be tokenized.
    :param verbosity: frequency of status messages
    :param replace_text: whether to replace the text to the tokenized version. True by default.
        If False, the tokenized text is stored under attribute 'tokenized'.
    :param save_original: if replacing text, whether to save the original version of the text. If True, saves it
        under the 'original' attribute.
    """
    def __init__(self, tokenizer: Optional[Callable[[str], str]]=None, 
                 input_field="parsed", input_filter=lambda utt, aux: True,
                 verbosity: int = 100, replace_text: bool = True, save_original: bool = True):

        if replace_text:
            if save_original:
                output_field = 'original'
            else:
                output_field = 'tokenized_temp'
        else:
            output_field = 'tokenized'
            
        self.replace_text = replace_text
        self.save_original = save_original
        
        super().__init__(proc_fn=join_tokens, input_field=input_field, input_filter=input_filter,
                         verbosity=verbosity, output_field=output_field)

    def transform(self, corpus: Corpus) -> Corpus:
        
        super().transform(corpus)
        
        if self.replace_text:
            selector = lambda utt_: self.input_filter(utt_, None)
            
            for utt in corpus.iter_utterances(selector):
                
                tokenized_text = utt.retrieve_meta(self.output_field)
                
                if self.save_original:
                    utt.add_meta(self.output_field, utt.text)
                
                utt.text = tokenized_text

            if not self.save_original:
                corpus.delete_metadata('utterance', self.output_field)
        
        return corpus  