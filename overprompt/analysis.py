import os
os.environ["TIKTOKEN_CACHE_DIR"] = "./output/tmp"

import tiktoken
import time

class EffeciencyAnalysis:
    def __init__(self, encoding_name: str) -> None:
        self.encoding_name = "gpt-3.5-turbo"
        self.tokens = []
        self.times = []
    
    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        # encoding = tiktoken.get_encoding(self.encoding_name)
        encoding = tiktoken.encoding_for_model(self.encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    def add_checkpoint(self, start_time:time, end_time:time, prompt) -> None:
        num_tokens = self.num_tokens_from_string(prompt)
        time = end_time - start_time
        self.tokens.append(num_tokens)
        self.times.append(time.total_seconds())
    
    def get_average_prompt_tokens(self, total_lines:int) -> float:
        return sum(self.tokens) / total_lines
    
    def get_total_time_cost(self) -> float:
        return sum(self.times)

    def get_average_time_cost(self, total_lines:int) -> float:
        return sum(self.times) / total_lines