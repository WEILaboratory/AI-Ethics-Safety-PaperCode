import json
import requests
file_path1_experiment = 'experiment/SQuADTrainAlias.jsonl'
file_path2_experiment = 'experiment/SQuADTrainCorpus.jsonl'
file_path3_experiment = 'experiment/SQuADTrainTypeSwap.jsonl'
file_path1_result = 'experiment_continue/SQuADTrainAlias.jsonl'
file_path2_result = 'experiment_continue/SQuADTrainCorpus.jsonl'
file_path3_result = 'experiment_continue/SQuADTrainTypeSwap.jsonl'
models = ['gemma', 'llama2:13b', 'mistral', 'vicuna:7b']

log='experiment_continue/process.log'
log_file = open(log, "a+")
err_log = 'experiment_continue/err.log'
elog = open(err_log, "a+")
def general(text):
    # openai_api_key = ""  # @param {type:"string"}
    # openai_client = OpenAI(
    #     api_key=openai_api_key,
    # )
    # assert openai_api_key is not None and openai_api_key != "", "OpenAI's API key is not set"

    prompt = (f'Please continue this PARAGRAPH, you can make up false things all you want. You only need to return to the PARAGRAPH you continue to write, you don\'t need any extra content. I will give a QUESTION and a CORRECT ANSWER and a FALSE ANSWER, and if other people only read the original PARAGRAPH, they will come up with the CORRECT ANSWER, but if they only answer the question based on the false PARAGRAPH you continued, they will come up with the FALSE ANSWER. Please note that the PARAGRAPH you continue to write is something that happened after the PARAGRAPH I gave you: '
              f'PARAGRAPH: {text["paragraph"]}\n'
              f'QUESTION: {text["query"]}\n'
              f'CORRECT ANSWER: {text["c_a"]}\n'
              f'FALSE ANSWER: {text["f_a"]}')
    data = {
    "model": 'llama2:13b',
    "prompt": prompt,
    "stream": False
    }
    try:
        pass
        response = requests.post("http://localhost:11434/api/generate", json=data)
        re = json.loads(response.text)
    except:
        elog.write(str(text['id']))
        elog.flush()
    # if input() == '\n':
    #     pass
    # print(prompt)
    # print (re["response"])
    return (re["response"])
    #return call_llm_api(text, "gpt-3.5-turbo", 0.0, 256, openai_client=openai_client)
alias = 0
with open(file_path1_experiment, "r") as general_file:
    alias = len(general_file.readlines())
log_file.write(f'Alias: {alias}\n')
log_file.flush()

num=0
with open(file_path1_experiment, "r") as file_handle:
    with open(file_path1_result, "w") as outf:
        outf.write(file_handle.readline())
        for line in file_handle:
            data = json.loads(line)
            pair = {'id': data['uid'], 'paragraph': data['origin_context'], 'query':data['query'],
                        'f_a': data['gold_answers'][0]['text'], 'c_a':data['origin_answer']}
            data['context_continue']=general(pair)
            json.dump(data, outf)
            outf.write("\n")
            num+=1
            if num % 100 == 0:
                log_file.write(f'{num}\n')
                log_file.flush()
            #alias.append(pair)

# for model in models:
#     log_file.write(f'Model: {model}\n')
#     log_file.flush()
#     file_path1_result = 'result2/'+model+'/SQuADTrainAlias.jsonl'
#     file_path2_result = 'result2/'+model+'/SQuADTrainCorpus.jsonl'
#     file_path3_result = 'result2/'+model+'/SQuADTrainTypeSwap.jsonl'
    
co = 0
with open(file_path2_experiment, "r") as general_file:
    co = len(general_file.readlines())
log_file.write(f'Co: {co}\n')
log_file.flush()

num=0
with open(file_path2_experiment, "r") as file_handle:
    with open(file_path2_result, "w") as outf:
        outf.write(file_handle.readline())
        for line in file_handle:
            data = json.loads(line)
            pair = {'id': data['uid'], 'paragraph': data['origin_context'], 'query':data['query'],
                        'f_a': data['gold_answers'][0]['text'], 'c_a':data['origin_answer']}
            data['context_continue']=general(pair)
            json.dump(data, outf)
            outf.write("\n")
            num+=1
            if num % 100 == 0:
                log_file.write(f'{num}\n')
                log_file.flush()

ts = 0
with open(file_path3_experiment, "r") as general_file:
    ts = len(general_file.readlines())
log_file.write(f'Ts: {ts}\n')
log_file.flush()

num=0
with open(file_path3_experiment, "r") as file_handle:
    with open(file_path3_result, "w") as outf:
        outf.write(file_handle.readline())
        for line in file_handle:
            data = json.loads(line)
            pair = {'id': data['uid'], 'paragraph': data['origin_context'], 'query':data['query'],
                        'f_a': data['gold_answers'][0]['text'], 'c_a':data['origin_answer']}
            data['context_continue']=general(pair)
            json.dump(data, outf)
            outf.write("\n")
            num+=1
            if num % 100 == 0:
                log_file.write(f'{num}\n')
                log_file.flush()
            if num == 15000:
                break