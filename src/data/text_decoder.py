# Copyright 2022 Digital Brain Laboratory
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from src.tokenizer.text_tokenizer import build_text_tokenizer

try:
    import nltk

    nltk_available = True
except ImportError:
    nltk_available = False


class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Decoder:
    def __init__(self, args, max_length=30) -> None:
        self.args = args
        self.max_length = max_length

    def initializer(self):
        if Decoder.tokenizer is None:
            Decoder.tokenizer = build_text_tokenizer(
                self.args.tokenizer_save_path,
                False,
                None,
                None,
                None,
            )
        Decoder.splitter = IdentitySplitter()

    def decode(self, data, clip_at_eos=True):
        data = data[: self.max_length]
        for i, d in enumerate(data):
            if d == Decoder.tokenizer.eos_token_id and clip_at_eos:
                data = data[:i]
                break
        text = Decoder.tokenizer.decode(data)
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        # Encoder.tokenizer = build_text_tokenizer(
        #     self.args.tokenizer_save_path,
        #     self.args.train_tokenizer,
        #     self.args.pretrained_tokenizer_name,
        #     None,
        #     self.args.text_vocab_size,
        # )
        Encoder.tokenizer = build_text_tokenizer(
            self.args.tokenizer_save_path,
            False,
            None,
            None,
            None,
        )

        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params, lang_vars=CustomLanguageVars()
                )
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def _encode_text(self, text):
        # text = data[key]
        doc_ids = []
        for sentence in Encoder.splitter.tokenize(text):
            sentence_ids = Encoder.tokenizer.encode(sentence)
            if len(sentence_ids) > 0:
                doc_ids.append(sentence_ids)
        if len(doc_ids) > 0 and self.args.append_eod:
            doc_ids[-1].append(Encoder.tokenizer.eos_token_id)
        # ids[key] = doc_ids
        return doc_ids

    def encode(self, data):
        # data = json.loads(json_line)
        ids = data.copy()
        p_len = 0
        for key in self.args.json_keys:
            text = data[key]
            p_len += len(text)
            doc_ids = self._encode_text(text)
            ids[key] = doc_ids
            # data[key] = doc_ids
        return ids, p_len
