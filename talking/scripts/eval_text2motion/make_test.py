import json
import random

t2m_data = json.load(open('video2instruction.json'))
t2m_test_data = random.sample(t2m_data.keys(), 200)

expected_list = ['smile', 'angry', 'sad', 'surprise', 'disgust', 'fear', 'neutral', "natural", "happy", "calm", "angry", "sad", "surprised", "disgusted", "fearful",
                    "disappointed", "bored", "excited", "frustrated", "proud", "ashamed", "amused", "excited", "tired", "sleepy", "neutral", "natural", "happy", "calm",
                    "open", "close", "turn", "move", "shake", "nod", "tilt", "rotate", "lift", "lower", "push", "pull", "wave", "point", "grasp", "release", "squeeze","random",
                    "suprised", "free", "annoyed", "eyes", "mouth", "laugh", "smile", "cry", "scream", "yell", "talk", "whisper", "sing", "shout", "speak", "squeak", "squeal",
                    "scared","confused","smile", "frown", "laugh", "wink", "surprise", "anger", "sadness", 
"disgust", "fear", "confusion", "excitement", "happiness", "contentment", 
"disappointment", "embarrassment","nod", "shake head", "tilt head", "turn head", "bob head", 
"raise eyebrows", "scrunch eyebrows", "roll eyes", "head bang", 
"head tilt", "head shake", "head nod", "head turn", "head tilt","head","eyebrows","terrified","sleepy",
"silly","grumpy","mean","curious","frustrated","proud","freestyle","smirk","chin","relax","interested"]


t2m_test_dict = {}
for key in t2m_test_data:
    instruction = t2m_data[key]
    for word in expected_list:
        if word in instruction:
            t2m_test_dict[key] = instruction
            break
    

with open('t2m_test_data.json', 'w') as f:
    json.dump(t2m_test_dict, f, indent=4)