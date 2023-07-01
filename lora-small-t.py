import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from minlora import add_lora, get_lora_params
from pprint import pprint





tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m-deduped")

num_tokens = model(**tokenizer("test", return_tensors='pt')).logits.shape[-1]

tkns = tokenizer("Write a c function that takes a number x and returns it's square. \nint square(int x) {", return_tensors='pt')


outs = model(tkns['input_ids'])

data = pd.read_parquet('qa-data.parquet', engine='fastparquet')[['prompt', 'chosen']]

def get_batch(batch_size=3):
    batch_xs, batch_ys = [], []
    batch_tkns = []
    biggest = 0
    for _ in range(batch_size):
        q_a = data.iloc[np.random.randint(data.shape[0])]
        batch_tkns.append(tokenizer(q_a['prompt'] + q_a['chosen'], return_tensors='pt')['input_ids'])
        if batch_tkns[-1].shape[1]-1 > biggest:
            biggest = batch_tkns[-1].shape[1]-1
    for i in range(batch_size):
        x = torch.zeros(1, biggest, dtype=torch.int)
        x[:,-batch_tkns[i].shape[1]+1:] = batch_tkns[i][:,:-1]
        y = torch.zeros(1, biggest, num_tokens)
        y[0,np.arange(biggest - batch_tkns[i].shape[1] + 1, biggest), batch_tkns[i][0,1:]] = 1

        batch_xs.append(x)
        batch_ys.append(y)
    batch_xs = torch.cat(batch_xs, dim=0)
    batch_ys = torch.cat(batch_ys, dim=0)
    return batch_xs, batch_ys



train_steps = 1000
batch_size = 3


add_lora(model)


parameters = [
    {"params": list(get_lora_params(model))},
]
opt = optim.Adam(parameters, lr=1e-4)
for i in range(train_steps):
    batch_xs, batch_ys = get_batch(batch_size)
    outs = F.softmax(model(batch_xs).logits, dim=-1)
    loss = torch.mean(torch.square(outs - batch_ys))
    l2 = torch.mean(-torch.sum(batch_ys * torch.log(outs), dim=-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if i % 100 == 99:
        print("\nGenerated:")
        tkns = model.generate(**tokenizer("How do I make money.", return_tensors='pt'), max_new_tokens=100)
        print(tokenizer.decode(tkns[0]))
   




get_batch()

'''
prompt = data['prompt'].iloc[0]
print(prompt)
tkns = tokenizer(prompt, return_tensors='pt')
gen = model(**tkns)
'''

