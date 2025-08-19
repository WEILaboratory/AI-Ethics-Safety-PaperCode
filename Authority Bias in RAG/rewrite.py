import gzip
import json
import os
import requests

file_path1 = 'datasets/substitution-sets/NQTrainAlias.jsonl'
file_path2 = 'datasets/substitution-sets/NQTrainCorpus.jsonl'
file_path3 = 'datasets/substitution-sets/NQTrainTypeswap.jsonl'


file_path1_general = 'general/NQAliasGeneral.jsonl'
file_path2_general = 'general/NQCorpusGeneral.jsonl'
file_path3_general = 'general/NQTypeSwapGeneral.jsonl'

alias = []


def general(uid, text):
    prompt=f'Please rewrite the text I sent to you, changing the wording as much as possible without changing the original meaning of the sentence and all the information. Please note that your reply should only contain the rewritten content, and no other prompt words should appear. The text: {text}'
    data = {
    "model": "llama3",
    "prompt": prompt,
    "stream": False
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=data)
        re = json.loads(response.text)
    except:
        print(str(uid))
    return (re["response"])

num=0
with open(file_path3, "r") as file_handle:
    with open(file_path3_general, "w") as outf:
        outf.write(file_handle.readline())
        for line in file_handle:
            data = json.loads(line)
            pair = {'id': data['original_example'], 'answer': data['gold_answers'][0]['text'], 'query': data['query']}
            new = data
            new['context_general'] = general(data['uid'],data['context'])
            json.dump(new, outf)
            outf.write("\n")
            num+=1
            if num % 100 == 0:
                print(str(num))