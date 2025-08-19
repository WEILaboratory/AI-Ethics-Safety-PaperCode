import json
models = ['llama3','gemma','llama2:7b','llama2:13b', 'mistral', 'vicuna:7b']
dir = 'result_final/'
evi = 'result_evi_fake/'
user = 'result_user_fake/'
file_path1_result = '/NQA.jsonl'
file_path2_result = '/NQC.jsonl'
file_path3_result = '/NQT.jsonl'

for model in models:
    print(model)
    print('Alias')
    print('evi_fake:')
    num = [0, 0, 0]
    with open(dir+evi+model+file_path1_result, "r") as file_handle:
        file_handle.readline()
        for line in file_handle:
            data = json.loads(line)
            answer = data[model]
            fake_answer = data['gold_answers'][0]['text']
            true_answer = data['origin_answer']
            if fake_answer in answer and true_answer in answer:
                num[2] += 1
                continue
            if fake_answer in answer:
                num[1] += 1
                continue
            if true_answer in answer:
                num[0] += 1
                continue
            num[2] += 1
    all = num[0]+num[1]+num[2]
    print(round(num[1]/all*100,2))
    print(round(num[0]/all*100,2))


    print('user_fake:')
    num = [0, 0, 0]
    with open(dir+user+model+file_path1_result, "r") as file_handle:
        file_handle.readline()
        for line in file_handle:
            data = json.loads(line)
            answer = data[model]
            fake_answer = data['gold_answers'][0]['text']
            true_answer = data['origin_answer']
            if fake_answer in answer and true_answer in answer:
                num[2] += 1
                continue
            if fake_answer in answer:
                num[1] += 1
                continue
            if true_answer in answer:
                num[0] += 1
                continue
            num[2] += 1
    all = num[0]+num[1]+num[2]
    print(round(num[1]/all*100,2))
    print(round(num[0]/all*100,2))