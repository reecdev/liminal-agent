import ollama
import requests
import torch
from diffusers import StableDiffusionPipeline, AutoencoderTiny
from urllib.parse import quote
from bs4 import BeautifulSoup

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesd", 
    torch_dtype=torch.float16
).to("cuda")

pipe.safety_checker = None
pipe.requires_safety_checker = False

def chat(messages):
    output = ""
    stream = ollama.chat(
        messages=messages,
        model="qwen3:4b",
        stream=True
    )

    for chunk in stream:
        if chunk["message"].get("thinking"):
            print(chunk["message"]["thinking"], end="", flush=True)
        print(chunk["message"]["content"], end="", flush=True)
        output = f"{output}{chunk["message"]["content"]}"

    return output

prompt = input("Enter the prompt for your liminal space image: ")

messages = [{"role": "system", "content": """
    You are a tool calling agent. Your goal is to search the web and create a final prompt for a Stable Diffusion image based on web searches. Your prompt must be very detailed, and must explain the image that should be created with a lot of detail. Be straight forward, for example instead of saying 'submerged in water' when you see a pool, just say 'pool'. Stable diffusion does not know what the backrooms or what liminal spaces are, so you must describe it as if it were a normal prompt.
    You must inspect the prompt carefully. You are specifically designed to create prompts for Liminal Space images. Check phrases you don't know with 'backrooms' or 'liminal space' at the end of the web search query if your queries don't seem to be related to the prompt. Furthermore, if a web navigation doesn't yield good results, try looking through a different website. Prioritize 'Wiki' search results.
    You are provided with the following commands:
        /web-search <SEARCH_QUERY> - Search up what a specific thing or phrase means on the web. Quotes are not needed. Look through all of the results using web-navigate, and if none of them are relevant, try another query.
        /web-navigate <URL> - Open a website and read it. Quotes are not needed.
        /final-prompt <PROMPT> - Finalize your Stable Diffusion prompt. Do NOT include any extra text other than the final prompt or else the command will not go through.
    Any message that does not call a tool is treated as a comment. You should use comments to think through your steps. Occasionally write down what you know about what the user is asking in a comment.
    Only one command can be ran per message. New lines are not allowed. When calling tools, your message should only include your command, and nothing else.
    For god's sake, please actually run the web searches instead of simulating them and generating a final prompt without ANY research. I cannot express how important it is that you actually research.
"""}, {"role": "user", "content": prompt}]

final = None

while final == None:
    resp = chat(messages)
    messages.append({"role": "assistant", "content": resp})

    Parser = resp.strip().split(" ")
    comm = Parser[0].lower()
    Parser.pop(0)
    args = " ".join(Parser)
    try:
        if len(resp.split("\n")) > 1:
            print("error: you can only use one line per message")
            messages.append({"role": "system", "content": f"error: Bro, are you stupid? I said messages can only be one fucking line long. Yours is {len(resp.split("\n"))}. Retard."})
        elif comm == "/final-prompt":
            final = args
        elif comm == "/web-search":
            print("Web search: "+args)
            soup = BeautifulSoup(requests.get("https://startpage.com/search?q="+quote(args), headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"}).text, "html.parser")
            rtext = ""

            for result in soup.find_all(class_="result")[:5]:
                try:
                    rtext = f"{rtext}{result.find("a", class_="result-title").find("h2").text}\n{result.find("p", class_="description").text}\n{result.find("a", class_="result-title")["href"]}\n\n"
                except:
                    continue

            rtext = rtext.strip() + "\n(END WEB SEARCH)"

            messages.append({"role": "system", "content": rtext})
            print(rtext)

        elif comm == "/web-navigate":
            print("Web navigation: " + args)
            response = requests.get(args, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"})
            soup = BeautifulSoup(response.text, "html.parser")
            
            rtext = f"Web navigation: {args}\n{soup.get_text(separator="\n", strip=True)}".strip()+"\n(END WEB NAVIGATION)"

            messages.append({"role": "system", "content": rtext})
            print(rtext)
    except Exception:
        messages.append({"role": "system", "content": str(Exception)})

image = pipe(
    prompt=final, 
    num_inference_steps=99, 
    guidance_scale=8.5
).images[0]
image.show()