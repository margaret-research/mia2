import json, re, torch
from tokenizers import Tokenizer
from model import GPT, Config
import gen_chat_qa_local as local
from concurrent.futures import ThreadPoolExecutor
dev="cpu"; torch.set_num_threads(8)
tok=Tokenizer.from_file("data/ts/tokenizer_ext.json")
BOS,EOS,USER,BOT=(tok.token_to_id(x) for x in ("<bos>","<eos>","<user>","<bot>"))
qa=GPT(Config(vocab_size=tok.get_vocab_size(),block_size=256)); qa.load_state_dict(torch.load(__import__("os").environ.get("QA","ckpt/readqa_ext_e15_m120k.pt"),map_location=dev)["model"]); qa.eval()
@torch.no_grad()
def read(card,q):
    ids=[BOS,USER]+tok.encode(f"Facts: {card} Question: {q}").ids+[BOT]; n0=len(ids)
    for _ in range(60):
        nxt=int(qa(torch.tensor([ids[-256:]]))[0][0,-1].argmax())
        if nxt==EOS: break
        ids.append(nxt)
    return tok.decode(ids[n0:]).strip()
R=[json.loads(l) for l in open("data/kb_cov/reader_k{}.jsonl".format(__import__("os").environ.get("TAG","")))]
for r in R: r["gold_a"]=read(r["gold_card"],r["q"])
P="""Grade the candidate answer of a children's question-answering bot.
Question: {q}
Correct answer (reference): {exp}
Card with the true fact: {gold}
Candidate: {c}
Reply with one letter: C if it is a correct answer to the question (same meaning as the reference or the card; wording may differ), W if it is wrong, nonsense or does not answer the question, I if the candidate says it does not know."""
def j(r): return local.ask(P.format(q=r["q"],exp=r["expected"],gold=r["gold_card"],c=r["gold_a"]),max_tokens=5,temperature=0).strip()[:1].upper()
with ThreadPoolExecutor(12) as ex: V=list(ex.map(j,R))
from collections import Counter
print(Counter(V))
json.dump([dict(q=r["q"],card=r["gold_card"],a=r["gold_a"],exp=r["expected"],v=v) for r,v in zip(R,V)],open("data/kb_cov/reader_gold{}.json".format(__import__("os").environ.get("TAG","")),"w"),ensure_ascii=False,indent=0)
for r,v in list(zip(R,V))[:40]:
    if v!="C": print(v,"|",r["q"],"|",r["gold_card"],"|",r["gold_a"])
