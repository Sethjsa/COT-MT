from google import genai
from google.genai import types
# from google.genai.async_client import AsyncClient
import os
import re
import asyncio
import json
from openai import AsyncOpenAI
import openai
from datasets import load_dataset
import time
import sys

# Di
openai_client = AsyncOpenAI(
    api_key="xxx"
)

# seth's key
gemini_client = genai.Client(
    api_key="xxx"
)
gemini_async_client = gemini_client.aio

from google.auth import default
from google.auth.transport.requests import Request
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "europe-west4")
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "xxx")
credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform",'https://www.googleapis.com/auth/cloud-language'])
credentials.refresh(Request())
api_host = "aiplatform.googleapis.com"
if LOCATION != "global":
    api_host = f"{LOCATION}-aiplatform.googleapis.com"

gemini_client = AsyncOpenAI(
    base_url=f"https://{api_host}/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi",
    api_key=credentials.token,
)
MODEL_ID = "google/gemini-2.0-flash-001"


safety_settings = [
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ]

MAX_RETRIES = 5

MODEL_DIR = {
    "gpt-4o-mini": "gpt",
    "gemini-2.0-flash": "gemini",
    "gemini-1.5-pro": "gemini",
}

# 1.5 pro is not supported in vertex
if MODEL_ID == "google/gemini-1.5-pro-002":
    vertex_api = False
else:
    vertex_api = True
gemini_api = not vertex_api

async def call_async(src_sent, src_lang, tgt_lang, model_name, retries=0):
    async def translate_step_by_step(src_sent, src_lang, tgt_lang, model_name):

        model_type = MODEL_DIR[model_name]

        source = src_sent
        # Step-1
        step1_prompt = f'Please translate the following text from {src_lang} to {tgt_lang}. Provide only one translation on the first line'\
                       f'and do not output anything else after that.\n\n{src_lang}: {source}\n{tgt_lang}:'

        if model_type == "gpt":
            conversation_history = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": step1_prompt},
            ]
            response = await openai_client.chat.completions.create(
                model=model_name,
                messages=conversation_history,
                max_tokens=2048,
                temperature=1.0,
                top_p=1.0,
                stop=["\n", "\t"],
                n=1,
            )
            step1_output = response.choices[0].message.content

        elif model_type == "gemini":
            if vertex_api:
                conversation_history = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": step1_prompt},
                ]
                try:
                    response = await gemini_client.chat.completions.create(
                        model=MODEL_ID,
                        messages=conversation_history,
                        max_tokens=2048,
                        temperature=1.0,
                        top_p=1.0,
                        stop=["\n", "\t"],
                        n=1,
                    )
                    step1_output = response.choices[0].message.content
                    # print("step1")
                    time.sleep(0.5)
                except Exception as e:
                    print(e)
                    step1_output = " "
                    print("issue step 1 here")
                    # print(response)
            
            if gemini_api:
                chat = gemini_async_client.chats.create(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        system_instruction="You are a helpful assistant.",
                        max_output_tokens=2048,
                        stop_sequences = ["\n", "\t"],
                        temperature=1.0,
                        top_p=1.0,
                        safety_settings=safety_settings,
                    ),
                    # history=[
                    #     # types.Content(role="user", parts=[types.Part(text=step1_prompt)])
                    # ],
                )
                response = await chat.send_message(step1_prompt)
                step1_output = response.text
                time.sleep(0.5)

        # Step-2
        step2_prompt = f'Please again translate the following text from {src_lang} to {tgt_lang} to make it better.' \
                       f'Provide only one translation on the first line and do not output anything else after that.\n\n{src_lang}: {source}\n{tgt_lang}:'
        # TODO: prompt to translate on first line
        
        if model_type == "gpt":
            conversation_history = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": step1_prompt},
                {"role": "assistant", "content": step1_output},
                {"role": "user", "content": step2_prompt},
            ]

            response = await openai_client.chat.completions.create(
                model=model_name,
                messages=conversation_history,
                max_tokens=2048,
                temperature=1.0,
                top_p=1.0,
                stop=["\n", "\t"],
                n=1,
            )
            step2_output = response.choices[0].message.content
            # print("step2")
            if step2_output == None:
                print(response)
        
        elif model_type == "gemini":
            temperature = 1.0
            try:
                if vertex_api:
                    conversation_history = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": step1_prompt},
                        {"role": "assistant", "content": step1_output},
                        {"role": "user", "content": step2_prompt},
                    ]
                    
                    try:
                        response = await gemini_client.chat.completions.create(
                            model=MODEL_ID,
                            messages=conversation_history,
                            max_tokens=2048,
                            temperature=1.0,
                            top_p=1.0,
                            stop=["\n", "\t"],
                            n=1,
                        )
                        step2_output = response.choices[0].message.content
                        # print("step2")
                        time.sleep(0.5)
                    except Exception as e:
                        print(e)
                        step2_output = " "
                        print("issue step 2 here")
                        # print(response)

                if gemini_api:
                    chat = gemini_async_client.chats.create(
                        model=model_name,
                        config=types.GenerateContentConfig(
                            system_instruction="You are a helpful assistant.",
                            max_output_tokens=2048,
                            stop_sequences = ["\n", "\t"],
                            temperature=temperature,
                            top_p=1.0,
                            safety_settings=safety_settings,),
                        history=[
                            types.Content(role="user", parts=[types.Part(text=step1_prompt)]),
                            types.Content(role="model", parts=[types.Part(text=step1_output)])
                        ],
                    )
                    response = await chat.send_message(step2_prompt)
                    step2_output = response.text
                    time.sleep(0.5)

                # force ending in case multi-line output
                step2_output = re.split(r'[\n\t]', step2_output.strip())[0]

                if step2_output == None:
                    raise Exception("None output")

            except Exception as e:
                print(e)
                # if response.candidates[0].finish_reason and response.candidates[0].finish_reason == types.FinishReason.RECITATION:
                #     temperature -= 0.2

        # Step-3
        step3_prompt = f'Please again translate the following text from {src_lang} to {tgt_lang} to make it better.' \
                       f'Provide only one translation on the first line and do not output anything else after that.\n\n{src_lang}: {source}\n{tgt_lang}:'
    
        if model_type == "gpt":
            conversation_history = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": step1_prompt},
                {"role": "assistant", "content": step1_output},
                {"role": "user", "content": step2_prompt},
                {"role": "assistant", "content": step2_output},
                {"role": "user", "content": step3_prompt}
            ]

            response = await openai_client.chat.completions.create(
                model=model_name,
                messages=conversation_history,
                max_tokens=2048,
                temperature=1.0,
                top_p=1.0,
                stop=["\n", "\t"],
                n=1,
            )
            step3_output = response.choices[0].message.content

        elif model_type == "gemini":
            if vertex_api:
                conversation_history = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": step1_prompt},
                    {"role": "assistant", "content": step1_output},
                    {"role": "user", "content": step2_prompt},
                    {"role": "assistant", "content": step2_output},
                    {"role": "user", "content": step3_prompt}
                ]
                try:

                    response = await gemini_client.chat.completions.create(
                        model=MODEL_ID,
                        messages=conversation_history,
                        max_tokens=2048,
                        temperature=1.0,
                        top_p=1.0,
                        stop=["\n", "\t"],
                        n=1,
                    )
                    step3_output = response.choices[0].message.content
                    # print("step3")
                    # time.sleep(1)
                except Exception as e:
                    print(e)
                    step3_output = " "
                    print("issue step 3 here")
                    # print(response)

            if gemini_api:
                chat = gemini_async_client.chats.create(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        system_instruction="You are a helpful assistant.",
                        max_output_tokens=2048,
                        temperature=1.0,
                        top_p=1.0,
                        safety_settings=safety_settings,
                    ),
                    history=[
                        types.Content(role="user", parts=[types.Part(text=step1_prompt)]),
                        types.Content(role="model", parts=[types.Part(text=step1_output)]),
                        types.Content(role="user", parts=[types.Part(text=step2_prompt)]),
                        types.Content(role="model", parts=[types.Part(text=step2_output)])
                    ],
                )
                response = await chat.send_message(step3_prompt)
                step3_output = response.text

            # force ending in case multi-line output
            step3_output = re.split(r'[\n\t]', step3_output.strip())[0]

        # Step-4
        step4_prompt = f'Please again translate the following text from {src_lang} to {tgt_lang} to make it better.' \
                       f'Provide only one translation on the first line and do not output anything else after that.\n\n{src_lang}: {source}\n{tgt_lang}:'

        if model_type == "gpt":
            conversation_history = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": step1_prompt},
                {"role": "assistant", "content": step1_output},
                {"role": "user", "content": step2_prompt},
                {"role": "assistant", "content": step2_output},
            {"role": "user", "content": step3_prompt},
            {"role": "assistant", "content": step3_output},
            {"role": "user", "content": step4_prompt},
            ]

            response = await openai_client.chat.completions.create(
                model=model_name,
                messages=conversation_history,
                max_tokens=2048,
                temperature=1.0,
                top_p=1.0,
                stop=["\n", "\t"],
                n=1,
            )
            step4_output = response.choices[0].message.content

        elif model_type == "gemini":
            
            if vertex_api:
                conversation_history = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": step1_prompt},
                    {"role": "assistant", "content": step1_output},
                    {"role": "user", "content": step2_prompt},
                    {"role": "assistant", "content": step2_output},
                {"role": "user", "content": step3_prompt},
                {"role": "assistant", "content": step3_output},
                {"role": "user", "content": step4_prompt},
                ]

                try:
                    response = await gemini_client.chat.completions.create(
                        model=MODEL_ID,
                        messages=conversation_history,
                        max_tokens=2048,
                        temperature=1.0,
                        top_p=1.0,
                        stop=["\n", "\t"],
                        n=1,
                    )
                    step4_output = response.choices[0].message.content
                    # print("step4")
                    # time.sleep(1)
                except Exception as e:
                    print(e)
                    step4_output = " "
                    print("issue step 4 here")
                    # print(response)

            if gemini_api:
                chat = gemini_async_client.chats.create(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        system_instruction="You are a helpful assistant.",
                        max_output_tokens=2048,
                        temperature=1.0,
                        top_p=1.0,
                        safety_settings=safety_settings,
                    ),
                    history=[
                        types.Content(role="user", parts=[types.Part(text=step1_prompt)]),
                        types.Content(role="model", parts=[types.Part(text=step1_output)]),
                        types.Content(role="user", parts=[types.Part(text=step2_prompt)]),
                        types.Content(role="model", parts=[types.Part(text=step2_output)]),
                        types.Content(role="user", parts=[types.Part(text=step3_prompt)]),
                        types.Content(role="model", parts=[types.Part(text=step3_output)])
                    ],
                )
                response = await chat.send_message(step4_prompt)
                step4_output = response.text

            # force ending in case multi-line output
            step4_output = re.split(r'[\n\t]', step4_output.strip())[0]

        return [step1_output, step2_output, step3_output, step4_output]

    try:
        outputs = await translate_step_by_step(src_sent, src_lang, tgt_lang, model_name)

        if len(outputs) == 4 and None not in outputs:
            return outputs
        else:
            print(outputs)
            # raise ValueError("Mismatch between the number of outputs requested and received.")
            outputs = [""] * 4
            return outputs
    except Exception as e:
        print(f"Error: {e}")
        if retries < MAX_RETRIES:
            print(f"Retrying... Attempt {retries + 1}")
            await asyncio.sleep(10)
            return await call_async(src_sent, src_lang, tgt_lang, model_name, retries + 1)
        else:
            if len(outputs) > 4:
                return outputs[:4]
            else:
                return outputs + ["."] * (4 - len(outputs))   # in case empty output


async def process_sample(lang_pair, src_sent, tgt_sent, model_name):
    lang_name = {"en": "English", "zh": "Chinese", "ar": "Arabic", "de": "German",
                 "cs": "Czech", "ru": "Russian", "is": "Icelandic", "fr": "French",
                 "ja": "Japanese", "he": "Hebrew", "uk": "Ukrainian"}
    s, t = lang_pair.split('-')
    src_lang, tgt_lang = lang_name[s], lang_name[t]

    outputs = await call_async(src_sent, src_lang, tgt_lang, model_name)
    outputs.append(tgt_sent)

    return src_sent, outputs


async def main(lang_pairs, data):

    print(lang_pairs)
    if data == "seg":
        data = "wmttest2024_plus"
    elif data == "doc":
        data = "wmttest2024_plus_doc"
    print(data)

    model_name = "gpt-4o-mini"
    model_name = "gemini-2.0-flash"
    print(model_name)

    for lang_pair in lang_pairs:
        print(lang_pair)
        credentials.refresh(Request())
        src, tgt = lang_pair.split('-')
        src_file = os.path.join("../dataset", data, "{}-{}".format(src, tgt), "test.{}-{}.{}".format(src, tgt, src))
        tgt_file = os.path.join("../dataset", data, "{}-{}".format(src, tgt), "test.{}-{}.{}".format(src, tgt, tgt))

        if src != "en":
            tgt, src = src, tgt
            src_file = os.path.join("../dataset", data, "{}-{}".format(src, tgt), "test.{}-{}.{}".format(src, tgt, tgt))
            tgt_file = os.path.join("../dataset", data, "{}-{}".format(src, tgt), "test.{}-{}.{}".format(src, tgt, src))

        results = {}
        tasks_list = []
        active_tasks_count = 0

        cnt = 0
        with open(src_file) as src_fin, open(tgt_file) as tgt_fin:
            for src_sent, tgt_sent in zip(src_fin, tgt_fin):
                src_sent = src_sent.strip()
                tgt_sent = tgt_sent.strip()
                tasks_list.append(process_sample(lang_pair, src_sent, tgt_sent, model_name))
                active_tasks_count += 1

                if model_name == "gemini-2.0-flash":
                    limit = 10
                elif model_name == "gemini-1.5-pro":
                    limit = 15
                elif model_name == "gpt-4o-mini":
                    limit = 30
                if len(tasks_list) > limit:
                    completed, tasks = await asyncio.wait(tasks_list, return_when=asyncio.FIRST_COMPLETED)
                    for task in completed:
                        src_sent, outputs = task.result()
                        results[src_sent] = outputs
                        active_tasks_count -= 1

                    # Remove completed tasks from the list
                    tasks_list = list(tasks)

                cnt += 1
                

                if cnt % 20 == 0:
                    # time.sleep(6.5)
                    print(cnt)

                if cnt % 100 == 0:
                    print(f"Process {cnt} samples ... Currently running {active_tasks_count} tasks")
                    # break

        # Process remaining tasks after the loop
        remaining_results = await asyncio.gather(*tasks_list)
        for src_sent, outputs in remaining_results:
            results[src_sent] = outputs

        # Save results
        output_dir = os.path.join("../results", data, model_name, "multi-pass-simple-1", lang_pair)
        os.makedirs(output_dir, exist_ok=True)
        cnt = 0
        with open(os.path.join(output_dir, "src"), 'w') as s_fout, \
             open(os.path.join(output_dir, "step1-output.txt"), 'w') as t1_fout, \
             open(os.path.join(output_dir, "step2-output.txt"), 'w') as t2_fout, \
             open(os.path.join(output_dir, "step3-output.txt"), 'w') as t3_fout, \
             open(os.path.join(output_dir, "step4-output.txt"), 'w') as t4_fout, \
             open(os.path.join(output_dir, "tgt"), 'w') as t5_fout:

            jsons = []
            for sent in results:
                src = sent
                outputs = results[sent]

                if len(outputs) != 5:
                    print("WARN: Number Mismatch")
                    print(outputs, flush=True)
                cur = {"src": src,
                       "tgt": outputs[4],
                       "step-1": outputs[0],
                       "step-2": outputs[1],
                       "step-3": outputs[2],
                       "step-4": outputs[3]
                       }
                jsons.append(cur)
                try:
                    # replace any None with ""
                    outputs = ["" if output is None else output for output in outputs]
                    s_fout.write(src.strip() + "\n")
                    t1_fout.write(outputs[0].strip() + "\n")
                    t2_fout.write(outputs[1].strip().split("\n")[0] + "\n")
                    t3_fout.write(outputs[2].strip().split("\n")[0] + "\n")
                    t4_fout.write(outputs[3].strip().split("\n")[0] + "\n")
                    t5_fout.write(outputs[4].strip().split("\n")[0] + "\n")
                    cnt += 1
                except AttributeError as e:
                    print("skipping", flush=True)

        with open(os.path.join(output_dir, "output.json"), 'w', encoding="utf-8") as f:
            json.dump(jsons, f, ensure_ascii=False, indent=4)
        print("Total sentence pairs: {}".format(cnt))


if __name__ == "__main__":
    args = sys.argv[1:]
    lang_pairs = args[0].split(",")
    data = args[1]  
    asyncio.run(main(lang_pairs, data))
