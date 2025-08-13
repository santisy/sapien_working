from openai import OpenAI
import os
import random

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def augment_video_description(concrete_joint, base_prompt):
    system_prompt = "You must output only a single paragraph video description. No lists, no labels, no titles. Just one flowing sentence describing the furniture action with material, lighting, location, and contents revealed inside."
    
    user_prompt = f"Joint: {concrete_joint}\nAction: {base_prompt}\nWrite one paragraph describing this furniture opening, include random material type, lighting, room location, what's inside, end with 'No occlusion. Realistic video style.' Output format: single paragraph only."
    
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