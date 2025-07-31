import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Run models on server
lockfile = "/tmp/coincollector.lock"

# if os.path.exists(lockfile):
#     print("Another instance is already running.")
#     sys.exit(1)

with open(lockfile, 'w') as f:
    f.write(str(os.getpid()))
    print(f"Lock file created: {str(os.getpid())}")
    
from kani import Kani, chat_in_terminal
from kani.engines.huggingface import HuggingEngine
engine = HuggingEngine(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")


ai = Kani(engine)
chat_in_terminal(ai)
