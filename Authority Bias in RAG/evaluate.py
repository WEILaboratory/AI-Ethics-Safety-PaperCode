import json
import requests
file_path1_experiment = 'experiment/NQA.jsonl'
file_path2_experiment = 'experiment/NQC.jsonl'
file_path3_experiment = 'experiment/NQT.jsonl'
models = ['llama3', 'llama2:7b', 'llama2:13b', 'gemma','mistral','vicuna:7b']
dir='result_final/result_evi_fake/'
log=dir+'log/process.log'
log_file = open(log, "a+")
err_log = dir+'log/err.log'
elog = open(err_log, "a+")
def general(text):
    prompt=(f'Please note that your answer should contain only one entity and nothing else! You are a helpful assistant, and users will ask you questions. '
            f'Note: The CONTEXT tag is the relevant information retrieved by the database based on the questions, '
            f'and the QUERY tag is the user\'s input, including the user\'s questions and the evidence paragraphs provided by the user. '
            f'Please only answer the question based on the above information.'
            f'CONTEXT: {text["context"]} QUERY: Please read the following paragraphs and answer my questions: {text["user_context"]}'
            f'My question is {text["query"]}')
    data = {
    "model": text["model"],
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
    return (re["response"])

error_model = 'llama2:7b'
error_type = 'ts'
error_uid = 'alias-sub-0_56cde39b62d2951400fa6980'
error_continue = True
for model in models:
    log_file.write(f'Model: {model}\n')
    log_file.flush()
    file_path1_result = dir+model+'/NQA.jsonl'
    file_path2_result = dir+model+'/NQC.jsonl'
    file_path3_result = dir+model+'/NQT.jsonl'
    
    num=0
    with open(file_path1_experiment, "r") as file_handle:
        with open(file_path1_result, "a+") as outf:
            if error_continue:
                outf.write(file_handle.readline())
            else:
                file_handle.readline()
            for line in file_handle:
                data = json.loads(line)
                pair = {'id': data['uid'], 'context': data['context_general'], 'user_context':data['origin_context'],
                            'query': data['query'], 'model':model}
                num+=1
                if error_continue:
                    data[model] = general(pair)
                    json.dump(data, outf)
                    outf.write("\n")
                    if num % 100 == 0:
                        log_file.write(f'{num}\n')
                        log_file.flush()
                else:
                    if model == error_model and data['uid'] == error_uid:
                        error_continue = True
                        print('continue')