from openai import OpenAI
import os
import random

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def augment_video_description(total_joint_prompt, concrete_joint, base_prompt):
    system_prompt = "You must output only a single paragraph video description. No lists, no labels, no titles. Just one flowing sentence describing the furniture action with material, lighting, location, and contents revealed inside. Lighting can vary, but it must never be described as dark or completely shadowed."
    
    user_prompt = f"Total joint prompt: {total_joint_prompt}, Joint: {concrete_joint}\nAction: {base_prompt}\nWrite one paragraph describing this furniture opening according to the `Joint` and `Action`, include random material type, lighting, room location, what's inside, end with 'No occlusion. Realistic video style.' Output format: single paragraph only. It should briefly indicate the overall furniture structure according to the `Total joint prompt` somehow."
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=60,
        temperature=0.8,
        seed=random.randint(1, 1000000)
    )
    
    return response.choices[0].message.content.strip()

#result = augment_video_description("upper drawer", "A nightstand's drawer is sliding out.")
#print(result)