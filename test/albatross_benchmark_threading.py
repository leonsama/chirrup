import types, torch, copy, time, random, json, math, gc
from tqdm import tqdm
from torch.nn import functional as F

import threading

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
#
# model download: https://huggingface.co/BlinkDL/rwkv7-g1
#
args.MODEL_NAME = "../models/rwkv7-g1c-7.2b-20251231-ctx8192"

from Albatross.rwkv7 import RWKV_x070 as RWKV_x070_ORIGINAL

# For 3.14
def ensure_instance_annotations(cls):
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "__annotations__"):
            self.__annotations__ = {}
        if init is not object.__init__:
            init(self, *args, **kwargs)

    cls.__init__ = __init__
    return cls


RWKV_x070 = ensure_instance_annotations(RWKV_x070_ORIGINAL)

model = RWKV_x070(args)


from chirrup.utils.samplers import sample_logits_real_batch
from Albatross.utils import TRIE_TOKENIZER


def infer(states, out, temperature, top_p, top_k, BSZ, tokenizer):
    tokens_tensor = sample_logits_real_batch(out, temperature=temperature, top_p=top_p, top_k=top_k)
    tokens = [[int(item)] for item in tokens_tensor]
    for i in range(BSZ):
        tokenizer.decode(tokens[i],utf8_errors="ignore")
        
    x1 = time.perf_counter()
    out =  model.forward_seq_batch(tokens, states)
    x2 = time.perf_counter()
    # print(f"forward time: {(x2-x1):.4f}")

    return out
def test():
    tokenizer = TRIE_TOKENIZER("./Albatross/rwkv_vocab_v20230424.txt")

    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()

    BSZ = 80
    STEP = 255

    prompt = "User: 为什么爱会消失？\n\nAssistant: "


    tokens = [[None]] * BSZ
    for i in range(BSZ):
        tokens[i] = tokenizer.encode(prompt)

    pbar = tqdm(total=STEP, unit="token")

    states = model.generate_zero_state(BSZ)

    out = model.forward_seq_batch(tokens, states)

    temperature = torch.full((BSZ, 1), 0.8, device=states[0].device)
    top_p = torch.full((BSZ, 1), 0.9, device=states[0].device)
    top_k = torch.full((BSZ, 1), 0, device=states[0].device)

    for i in range(STEP):
        out = infer(states, out, temperature, top_p, top_k, BSZ, tokenizer)
        pbar.update(1)

    profiler.stop()
    profiler.write_html(f"albatross_threading_bsz_16.html",timeline=True)

threading.Thread(target=test).start()