from convokit.model import Corpus
from .textProcessor import TextProcessor
from typing import Callable, Optional 

# join tokens from existing parses
join_tokens = lambda p: " ".join([" ".join([tok['tok'] for tok in sent['toks']]) for sent in p])

class TextToTokenized(TextProcessor):
    """
    Transformer that formats utterance texts as space-separated (tokenized) text. Intended especially for languages which do not seperate words with spaces (such as Chinese).

    :param input_field: name of field to use as input. defaults to 'parsed', which stores dependency parses as returned by the TextParser transformer; otherwise expects similarly-formatted input.
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
                 verbosity: int = 1000, replace_text: bool = True, save_original: bool = False):

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
    
    
    def transform_utterance(self, utt, override_input_filter=False):
        """
        Overrides TextProcessor's transform_utterance

        :param utt: utterance or a string
        :param override_input_filter: ignore `input_filter` and compute attribute for all utterances
        :return: the utterance
        """

        if isinstance(utt, str):
            utt = Utterance(text=utt, speaker=Speaker(id="speaker"))
        
        if self.input_field is None:
            raise ValueError('input_fieldrequired.')
        
        else:
            if not override_input_filter:
                if not self.input_filter(utt, self.aux_input): 
                    return utt 
                text_entry = utt.retrieve_meta(self.input_field)
        
        if text_entry is None:
            return utt
        
        if len(self.aux_input) == 0:
            result = self.proc_fn(text_entry)
        else:
            result = self.proc_fn(text_entry, self.aux_input)
            
        if self.replace_text:
            
            if self.save_original:
                utt.add_meta(self.output_field, utt.text)
        
            utt.text = result
        
        else:
            utt.add_meta(self.output_field, result)
        
        return utt