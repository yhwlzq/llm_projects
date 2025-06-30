import tiktoken
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import Optional, List
import nltk

nltk.download('punkt')

@dataclass
class ChunkConfig:
    embedding_model:str="bert-base-uncased"


class BaseChunker:

    def __init__(self, chunk_size:int=512, overlap:int=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained(ChunkConfig.embedding_model)


    def tokenize(self, text:str) ->List[str]:
        return self.tokenize.tokenizer(text)
    

    def split_text(self, text:str)->List[str]:
        return NotImplementedError



class SemanticChunker(BaseChunker):

    def __init__(self, chunk_size = 512, overlap = 50, embedding_model:str='all-MiniLM-L6-v2'):
        super().__init__(chunk_size, overlap)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.similarity_threshhold = 0.85


    def _calculate_sentence_similarity(self, sentences:List[str])->np.ndarray:
        embeddings = self.embedding_model.encode(sentences)
        return np.dot(embeddings, embeddings.T)

    def split_text(self, text:str)->List[str]:
        sentences = sent_tokenize(text)
        if len(sentences) <=1:
            return [text]
        sim_matrix  = self._calculate_sentence_similarity(sentences)
        chunks = []
        current_chunk = []
        current_length = 0
        for i, sentence in enumerate(sentences):
            sent_tokens = self.tokenize(sentences)
            sent_length = len(sent_tokens)
            if current_chunk:
                last_send_idx = len(current_chunk)-1
                similarity = sim_matrix[i][last_send_idx]
                if similarity <self.similarity_threshhold or current_length + sent_length > self.chunk_size:
                    chunks.append(" ".join(current_chunk))
                    overlap_start = max(0, len(current_chunk)-self.overlap)
                    current_chunk = current_chunk[overlap_start:]
                    current_length = len(self.tokenize(" ".join(current_chunk)))
            current_chunk.append(sentence)
            current_length = current_length+sent_length
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks



class LLMChunker(SemanticChunker):
    def __init__(self,llm, **kwargs):
        super().__init__(kwargs)
        self.llm = llm


    def _identity_topic_shifts(self,sentences: List[str] )->List[str]:
        prompt = """
        分析以下句子序列，标记主题发生变化的位置(返回0表示延续，1表示变化):
        
        句子序列:
        {}
        
        只返回0和1的列表，例如：[0,0,1,0]
        """.format("\n".join(f"{i}.  {s}" for i, s in enumerate(sentences)))

        response = self.llm(prompt)

        try:
            shifts = eval(response)
            return [bool(s) for s in shifts]
        except:
            return [False] * len(sentences)
        

    def split_text(self, text):
        sentences = sent_tokenize(text)
        if len(sentences) <=1:
            return [text]
        topic_shifts = self._identity_topic_shifts(sentences)
        chunks = []
        current_chunk = []
        current_length = 0

        for i, (sentence, is_shift) in enumerate(zip(sentences, topic_shifts)):
            sent_tokens = self.tokenize(sentence)
            sent_length = len(sent_tokens)
            if (is_shift and current_chunk) or current_len+sent_length>self.chunk_size:
                chunks.append(" ".join(current_chunk))
                overlap_start = max(0, len(current_chunk)-self.overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = len(self.tokenize(" ".join(current_chunk)))

            current_chunk.append(sentence)
            current_length+=sent_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks


class LLM_chunkizer:

    @staticmethod
    def estimate_token_count(text:str, tokenizer_model:str='gpt-4'):
        encoding = tiktoken.encoding_for_model(tokenizer_model)
        return len(encoding.encode(text))


    @staticmethod
    def split_document_into_blocks(paragraphs, block_token_limit:int=5000):
        blocks = []
        current_block = ""
        current_token_count = 0

        for paragrahp in paragraphs:
            paragraph_token_count = LLM_chunkizer.estimate_token_count(paragrahp)
            if current_token_count+paragraph_token_count>block_token_limit:
               blocks.append(current_block)
               current_block = paragrahp
               current_token_count = paragraph_token_count
            else:
                current_block+=paragrahp.strip()+"\n"
                current_token_count +=paragraph_token_count
        if current_block:
            blocks.append(current_block.strip())


    @staticmethod
    def chunk_text_with_llm(llm, blocks):
        final_chunks = []
        last_chunk = ""
        last_chunk_2 = ""
        last_chunk_1 = ""
        for block in blocks:
            text = last_chunk+"\n"+block
            prompt =[
                {"role":"system","content":"You are assistant that helps divide documents into logical chunks based on complete ideas."},
                {"role":"system","content":f"please split the following text into logic chunks,using  using '!-!-!-!-!-!-!-!-!-!-!' to separate them. \n\n{text}"}
            ]

            response = llm.invoke(prompt)
            text_to_split = response.content
            splitted_array = text_to_split.split('!-!-!-!-!-!-!-!-!-!-!')

            last_chunk_1 = splitted_array.pop() if splitted_array else ""
            last_chunk_2 = splitted_array.pop() if splitted_array else ""
            last_chunk = last_chunk_2+"\n"+last_chunk_1

            final_chunks.extend(splitted_array)
        
        final_chunks.append(last_chunk_2)
        final_chunks.append(last_chunk_1)

        return final_chunks

from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="deepseek-r1:latest")
instance = LLMChunker(llm=llm)
document ="Hello world! How are you? I'm fine."
instance.split_text(document)



# en = tiktoken.encoding_for_model('gpt-4')
# text = "Hello, world!"
# tokens = en.encode(text)
# print("Tokens:", tokens)  # 输出类似 [9906, 11, 1917, 0]

# # 计算 token 数量
# num_tokens = len(tokens)
# print("Token count:", num_tokens)  # 