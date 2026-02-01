# liminal-agent
Liminal-Agent is an interface for Stable Diffusion with a tool-calling and web-searching agent based on Qwen's 4 billion parameter thinking model.
It is able to search and navigate the web for the most up-to-date information and create acurate and detailed prompts which are fed into Stable Diffusion to be converted into an image.

# NOTE
liminal-agent was made for fun. It is not intended for professional use, but is more so a proof-of-concept that implementing tool-calling with image generation is possible and can yield great results for images and concepts that are not in the image generator's training data.

## Prerequisities
Ensure you have PyTorch (GPU accelerated version reccomended), requests, HuggingFace diffusers, and ollama installed.
After you have all of these installed, run `ollama pull qwen3:4b` inside of your terminal to pull the base agent.
If you're running the script for the first time, it should automatically download Stable Diffusion and TinyAutoEncoder for you.

## Running the model
Running the model is very simple. It will ask you for the prompt of the image you want to generate, and then it will run it through the model and spit out an image.
Make sure you save the image before closing the window, as it doesn't autosave!
