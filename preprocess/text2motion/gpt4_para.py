import openai
openai.api_key = "xxxx"
openai.api_base = "xxxx"
# both api_type and api_version are required
openai.api_type = "azure"
openai.api_version = "2023-07-01-preview"
import time
def extract_seconds(text, retried=5):
    words = text.split()
    for i, word in enumerate(words):
        if "second" in word:
            return int(words[i - 1])
    return 60
import json
import os
from tqdm import tqdm


text_motion_inst = json.load(open('video2instruction.json'))


final_clarify = '''Below is an instruction provided to a person to do some actions.  Please turn into a fluent sentence or phrase\n\n'''


from PIL import Image
output = {}

for file, sent in tqdm(text_motion_inst.items()):
    
    
    Question = f'''Now I have an instruction '{sent}', please turn it into a natural and diverse sentence or phrase. Give me three examples.\n\n'''
    prompt =   final_clarify + Question + "Your answer:"
    
    
    
    result = ''
    retried = 0

    time_start = time.time()
    while True:
        try:
            time_end = time.time()
            if time_end - time_start > 3600:
                break
            completion = openai.ChatCompletion.create(
                deployment_id="gpt-4-turbo-v", messages=[{"role": "user", "content": prompt}],  temperature= 1, top_p = 0.95, max_tokens = 200
            )
            result = completion['choices'][0]['message']['content']
            if result == '':
                continue
            break
        except Exception as e:
            error = str(e)
            print("retring...", error)
            second = extract_seconds(error, retried)
            retried = retried + 1
            time.sleep(second + 1)

    print(result)
    output[file] = result

with open('t2m_paraphrase_turbo.json', 'w') as f:
    json.dump(output, f)
# Question = '''Now I have ["brow_lowerer", "jaw_drop", "lid_tightener", "upper_lip_raiser", "lips_part"], please turn it into a natural and diverse sentence. \n\n'''



# print(prompt)