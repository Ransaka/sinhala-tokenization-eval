import torch
import re
from transformers import BertTokenizer, AutoTokenizer
from collections import Counter
from sinlib import Tokenizer as SinlibTokenizerBase

class BaseTokenizer:
    """Base tokenizer interface"""
    def __init__(self, max_length=512):
        self.max_length = max_length
        
    def encode(self, text):
        raise NotImplementedError
        
    def batch_encode(self, texts):
        """Encode a batch of texts and create attention masks"""
        encoded_texts = [self.encode(text) for text in texts]
        max_len = min(max(len(et) for et in encoded_texts), self.max_length)
        
        # Create tensors for input_ids and attention_mask
        input_ids = []
        attention_mask = []
        
        for enc_text in encoded_texts:
            # Padding
            padded = enc_text[:self.max_length] + [self.pad_id] * (max_len - len(enc_text[:self.max_length]))
            input_ids.append(padded)
            
            # Attention mask (1 for real tokens, 0 for padding)
            mask = [1] * min(len(enc_text), self.max_length) + [0] * (max_len - min(len(enc_text), self.max_length))
            attention_mask.append(mask)
            
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)
        }

class ByteLevelTokenizer(BaseTokenizer):
    """Tokenizes text at the byte level using UTF-8 encoding."""
    def __init__(self, max_length=512):
        super().__init__(max_length)
        # Using byte-level tokenization, vocabulary size is 256 (0-255 byte values)
        self.vocab_size = 256
        # Adding special tokens
        self.pad_id = 0  # [PAD]
        self.cls_id = 256  # [CLS]
        self.sep_id = 257  # [SEP]
        # Total vocab size including special tokens
        self.total_vocab_size = self.vocab_size + 3  # 259
        
    def encode(self, text):
        """Convert text to byte sequence"""
        # Encode text to UTF-8 bytes and convert to list of integers
        bytes_list = list(text.encode('utf-8'))
        
        # Add CLS token at beginning
        tokens = [self.cls_id] + bytes_list
        
        # Truncate if necessary and add SEP token
        if len(tokens) > self.max_length - 1:
            tokens = tokens[:self.max_length - 1]
        tokens = tokens + [self.sep_id]
            
        return tokens

class CharacterTokenizer(BaseTokenizer):
    """Tokenizes text at the character level"""
    def __init__(self, max_length=512, texts=None):
        super().__init__(max_length)
        self.char_to_id = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[UNK]': 3}
        self.id_to_char = {0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: '[UNK]'}
        self.pad_id = 0
        self.cls_id = 1
        self.sep_id = 2
        self.unk_id = 3
        
        # Build vocabulary from texts if provided
        if texts:
            self._build_vocab(texts)
        
        self.total_vocab_size = len(self.char_to_id)
        
    def _build_vocab(self, texts):
        """Build character vocabulary from a corpus of texts"""
        # Get all unique characters
        all_chars = set()
        for text in texts:
            all_chars.update(text)
            
        # Add to vocabulary
        for char in sorted(all_chars):
            if char not in self.char_to_id:
                idx = len(self.char_to_id)
                self.char_to_id[char] = idx
                self.id_to_char[idx] = char
                
    def encode(self, text):
        """Convert text to character sequence"""
        # Convert characters to IDs
        char_ids = [self.char_to_id.get(char, self.unk_id) for char in text]
        
        # Add CLS and SEP tokens
        tokens = [self.cls_id] + char_ids
        
        # Truncate if necessary and add SEP token
        if len(tokens) > self.max_length - 1:
            tokens = tokens[:self.max_length - 1]
        tokens = tokens + [self.sep_id]
            
        return tokens

class WordTokenizer(BaseTokenizer):
    """Tokenizes text at the word level"""
    def __init__(self, max_length=512, texts=None, min_freq=2):
        super().__init__(max_length)
        self.word_to_id = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[UNK]': 3}
        self.id_to_word = {0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: '[UNK]'}
        self.pad_id = 0
        self.cls_id = 1
        self.sep_id = 2
        self.unk_id = 3
        self.min_freq = min_freq
        
        # Build vocabulary from texts if provided
        if texts:
            self._build_vocab(texts)
            
        self.total_vocab_size = len(self.word_to_id)
        
    def _build_vocab(self, texts):
        """Build word vocabulary from a corpus of texts"""
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = re.findall(r'\w+|[^\w\s]', text)
            word_counts.update(words)
            
        # Add words that meet minimum frequency
        for word, count in word_counts.items():
            if count >= self.min_freq and word not in self.word_to_id:
                idx = len(self.word_to_id)
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word
                
    def encode(self, text):
        """Convert text to word sequence"""
        # Split text into words
        words = re.findall(r'\w+|[^\w\s]', text)
        
        # Convert words to IDs
        word_ids = [self.word_to_id.get(word, self.unk_id) for word in words]
        
        # Add CLS and SEP tokens
        tokens = [self.cls_id] + word_ids
        
        # Truncate if necessary and add SEP token
        if len(tokens) > self.max_length - 1:
            tokens = tokens[:self.max_length - 1]
        tokens = tokens + [self.sep_id]
            
        return tokens

class WordPieceTokenizer(BaseTokenizer):
    """Wrapper for pre-trained WordPiece tokenizers"""
    def __init__(self, model_name='keshan/SinhalaBERTo', max_length=512):
        super().__init__(max_length)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Make sure you have the transformers library installed.")
            print("pip install transformers")
            raise
            
        self.pad_id = self.tokenizer.pad_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.total_vocab_size = len(self.tokenizer.vocab)
        
    def encode(self, text):
        """Use HuggingFace tokenizer for encoding"""
        # Encode without special tokens first
        tokens = self.tokenizer.encode(text, add_special_tokens=True, padding='max_length',truncation=True, max_length=self.max_length)
        return tokens
        
    def batch_encode(self, texts):
        """Use HuggingFace batch_encode_plus"""
        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }


class SinlibTokenizer(BaseTokenizer):
    """Wrapper for the sinlib Tokenizer"""
    
    def __init__(self, max_length=512, texts=None):
        super().__init__(max_length)
        self.tokenizer = SinlibTokenizerBase(max_length=max_length)
        
        # If texts are provided, train the tokenizer
        if texts:
            self.tokenizer.train(texts)
            
        self.pad_id = 0 
        self.special_tokens = self.tokenizer.special_tokens
        
        self.total_vocab_size = 10000  # Default fallback value
        try:
            sample_encoding = self.tokenizer("Sample text", add_bos_token=True, 
                                          allowed_special_tokens=self.tokenizer.special_tokens,
                                          truncate_and_pad=True)
            self.total_vocab_size = max(sample_encoding) + 1000  # Add buffer
        except:
            pass
    
    def encode(self, text):
        """Encode text using sinlib Tokenizer"""
        return self.tokenizer(
            text, 
            add_bos_token=True,
            allowed_special_tokens=self.tokenizer.special_tokens,
            truncate_and_pad=True
        )
    
    def batch_encode(self, texts):
        """Encode a batch of texts and create attention masks"""
        encoded_texts = [self.encode(text) for text in texts]
        
        # Check if all encodings have the same length
        lengths = [len(enc) for enc in encoded_texts]
        if len(set(lengths)) == 1:
            # All encodings have same length, we can simply stack
            return {
                'input_ids': torch.tensor(encoded_texts),
                'attention_mask': torch.ones(len(encoded_texts), lengths[0], dtype=torch.long)
            }
        else:
            # Different lengths, need to create padded batch
            max_len = min(max(lengths), self.max_length)
            
            # Create tensors for input_ids and attention_mask
            input_ids = []
            attention_mask = []
            
            for enc_text in encoded_texts:
                # Truncate if necessary
                enc_text = enc_text[:max_len]
                
                # Padding
                padded = enc_text + [self.pad_id] * (max_len - len(enc_text))
                input_ids.append(padded)
                
                # Attention mask (1 for real tokens, 0 for padding)
                mask = [1] * len(enc_text) + [0] * (max_len - len(enc_text))
                attention_mask.append(mask)
                
            return {
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)
            }
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)



def create_tokenizer(tokenizer_type, max_length=512, train_texts=None):
    """Factory function to create the appropriate tokenizer"""
    if tokenizer_type == "byte":
        return ByteLevelTokenizer(max_length=max_length)
    elif tokenizer_type == "char":
        return CharacterTokenizer(max_length=max_length, texts=train_texts)
    elif tokenizer_type == "word":
        return WordTokenizer(max_length=max_length, texts=train_texts)
    elif tokenizer_type == "wpe":
        return WordPieceTokenizer(max_length=max_length)
    elif tokenizer_type == "sinlib":
        return SinlibTokenizer(max_length=max_length, texts=train_texts)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
