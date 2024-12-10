import openai
openai.api_key = "xxxx"
openai.api_base = "xxxxxx"
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


au_system_define = "Action unit is a term used in facial expression analysis to describe specific movements of the facial muscles, which can be used to interpret and understand human emotions and expressions, For example, Inner Brow Raiser is the raising of the inner portion of the eyebrows, which is associated with surprise.\n\n"

task_definition = "Now I have an obtained action units from a video, the form is like this: ['brow_lowerer', 'lips_part', 'Cheek Raiser'], I want to combine them into a sentence, like 'make brow lower and seperated lips, what's more, you can also lift your cheek.'\n\n"


icl_example = '''More examples like this: ["upper_lip_raiser", "lips_part"] -> " make sure lip raised and lips parted"
["brow_lowerer", "lid_tightener"] -> "drop brow, at the same time you can tighten your lid" 
["lips_part", "Lip Corner Puller"] -> "try to part lips, meanwhile stretch lip corner"\n\n'''


final_clarify = '''Pay attention to that AU subject (like brow) should be strictly maintained while AU verb (like lower) can be changed. The way you express the sentence can also be free. You should try to make it deverse, clarity, natural but do not imagine unrealistic subject. 
Avoid using 'you' if possible . 
Do not use adverbs describing degree, such as 'slightly'. 
Do not incorporate temporal information, such as 'begin', 'then'.
Make sure all AU subjects are in the final sentence \n\n'''


from PIL import Image
output = {}

for person in tqdm(sorted(os.listdir('/xxxx/TETF/datasets/MEAD_all/au_detect/'))):
    if person == 'au_paraphrase.json':
        continue
    for video in tqdm(sorted(os.listdir('/xxxx/TETF/datasets/MEAD_all/au_detect/' + person + '/'))):
        try:
            au_json = json.load(open('/xxxx/TETF/datasets/MEAD_all/au_detect/' + person + '/' + video + '/intersection.json'))
            ref_path = '/xxxx/TETF/datasets/MEAD_all/au_detect/' + person + '/' + video + '/0.jpg'
            ref_img = Image.open(ref_path)
        except:
            continue
        au_list = list(au_json.values())[0]
        if len(au_list) == 0:
            continue
        
        Question = f'''Now please observe the image above, I give a predicted action units {au_list}, please turn it into a natural and diverse sentence. If you find a contraction with the image, you can edit the action unit. Give me three examples.\n\n'''
        prompt = au_system_define  + task_definition + icl_example + Question + final_clarify + "Your answer:"
        
        import base64
        IMAGE_PATH = ref_path
        encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
        
        result = ''
        retried = 0

        # print(prompt)
        time_start = time.time()
        while True:
            try:
                time_end = time.time()
                if time_end - time_start > 600:
                    break
                completion = openai.ChatCompletion.create(
                    deployment_id="gpt-4-turbo-v", 
                    messages=[
                    {"role": "user", 
                    "content": [
                                {"type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                }},
                                {"type": "text",
                                "text": prompt}
                                ]
                    }
                    ],
                    temperature= 0.95, top_p = 0.95, max_tokens = 200
                )
                result = completion['choices'][0]['message']['content']
                if result == '':
                    continue
                break
            except Exception as e:
                error = str(e)
                print("retring...", error)
                if 'content management policy' in error:
                    IMAGE_PATH = '/xxxx/TETF/datasets/MEAD_all/au_detect/' + person + '/' + video + '/1.jpg'
                    new_encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
                    while True:
                        try:
                            completion = openai.ChatCompletion.create(
                                deployment_id="gpt-4-turbo-v", 
                                messages=[
                                {"role": "user", 
                                "content": [
                                            {"type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{new_encoded_image}"
                                            }},
                                            {"type": "text",
                                            "text": prompt}
                                            ]
                                }
                                ],
                                temperature= 0.95, top_p = 0.95, max_tokens = 200
                            )
                            result = completion['choices'][0]['message']['content']
                            if result == '':
                                continue
                            break
                        except Exception as e:
                            error = str(e)
                            print("retring...", error)
                            if 'content management policy' in error:
                                break
                            second = extract_seconds(error, retried)
                            retried = retried + 1
                            time.sleep(second + 1) 
                    break
                else:
                    second = extract_seconds(error, retried)
                    retried = retried + 1
                    time.sleep(second + 1)

        print(result)
        output[f'{person}_{video}'] = result
        with open('/xxxx/TETF/datasets/MEAD_all/au_detect/' + person + '/' + video + '/paraphrase_v.txt', 'w') as f:
            f.write(result)

with open('/xxxx/TETF/datasets/MEAD_all/au_detect/au_paraphrase_v.json', 'w') as f:
    json.dump(output, f)
# Question = '''Now I have ["brow_lowerer", "jaw_drop", "lid_tightener", "upper_lip_raiser", "lips_part"], please turn it into a natural and diverse sentence. \n\n'''



# print(prompt)