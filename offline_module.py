from threading import Thread
from transformers import pipeline
import torch
import streamlit as st
import os
from diffusers import DiffusionPipeline
import ctransformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

#################################
# Local Audio to Text
#################################


@st.cache_resource()
def load_model_audio(model_path="whisper-1", chunk_length_s=30, device="cuda"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        chunk_length_s=chunk_length_s,
    )
    return model


def save_uploaded_audio_file(uploaded_file):
    try:
        # create a directory to save file
        os.makedirs("tempDir", exist_ok=True)
        file_path = os.path.join("tempDir", uploaded_file.name)
        # write file to directory
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        return None


def generate_text_from_audio_local(model, audio_file):
    file_path = save_uploaded_audio_file(audio_file)
    result = model(file_path)["text"]
    return result


#################################
# Local Image Generation
#################################


@st.cache_resource()
def load_model_local_sdxl(
    model_path="CompVis/stable-diffusion-v1-4",
    model_path_refiner=None,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
):

    #### BASE MODEL ####
    base = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        variant=variant,
        use_safetensors=use_safetensors,
    )
    base.enable_model_cpu_offload()

    #### REFINER ####
    if model_path_refiner:
        refiner = DiffusionPipeline.from_pretrained(
            model_path_refiner,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
            variant=variant,
        )
        refiner.enable_model_cpu_offload()
        return base, refiner
    else:
        refiner = None
        return base, refiner


def generate_image_local_sdxl(
    model,
    prompt,
    refiner=None,
    num_inference_steps=20,
    guidance_scale=15,
    high_noice_frac=0.8,
    output_type="image",
    verbose=False,
    temperature=0.7,
):
    if refiner:
        image = model(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            denoising_end=high_noice_frac,
            output_type=output_type,
            verbose=verbose,
            guidance_scale=guidance_scale,
            temperature=temperature,
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noice_frac,
            image=image,
            verbose=verbose,
        ).images[0]

    else:
        image = model(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

    return image


#################################
# Text Generation using LLama 2.7B
#################################


@st.cache_resource()
def load_model_llama2_gguf(model_path, model_file, model_type="llama", gpu_layers=30):
    model = ctransformers.AutoModelForCausalLM.from_pretrained(
        model_path_or_repo_id=model_path,
        model_file=model_file,
        model_type=model_type,
        gpu_layers=gpu_layers,
    )
    return model


def generate_text_llama2_gguf(
    model, prompt, text_area_placeholder, stop=["\n", "Question:", "Q:"]
):
    generated_text = ""
    for text in model(f"Question: {prompt} Amswer :", stream=True, stop=stop):
        generated_text += text
        text_area_placeholder.markdown(generated_text, unsafe_allow_html=True)
    return generated_text


#################################
# Text Generation using mistralai
#################################


@st.cache_resource()
def load_model_mistralai(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path).from_pretrained(model_path)
    return model, tokenizer


def generate_text_streamlit_mistralai(
    model,
    tokenizer,
    prompt,
    text_area_placeholder,
    max_tokens=100,
    top_k=1000,
    top_p=0.95,
    temperature=0.7,
    do_sample=True,
    timeout=10.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    messages = [{"role": "user", "content": prompt}]
    encode_input = tokenizer.apply_chat_template(messages, return_tensors="pt")
    input_ids = encode_input.to(device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=timeout, skip_prompt=True, skip_special_tokens=True
    )

    generate_args = {
        "input_ids": input_ids,
        "max_new_tokenns": max_tokens,
        "streamer": streamer,
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "pad_token_id": tokenizer.eos_token_id,
    }

    thread = Thread(target=model.generate, kwargs=generate_args)
    thread.start()
    generate_text = ""
    for text in streamer:
        generate_text += text
        text_area_placeholder.markdown(generate_text, unsafe_allow_html=True)
    return generate_text


def main():

    # ## Audio Transcription

    # st.title = "Audio Transcription using OpenAI API"
    # model_path = "whisper-1"
    # model = load_model_audio(model_path)
    # audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

    # if audio_file:
    #     if st.button("Transcribe"):
    #         st.audio(audio_file, format="audio/wav")
    #         with st.spinner("Transcribing..."):
    #             response = generate_text_from_audio_local(model, audio_file)
    #             st.write(response)

    # ## Image Generation

    # st.title = "Image Generation"
    # model_path = "CompVis/stable-diffusion-v1-4"
    # model_path_refiner = None
    # base, refiner = load_model_local_sdxl(model_path, model_path_refiner)
    # prompt = st.text_input(
    #     "Enter a prompt", value="Amrita Vishwa Vidhyapeetham university"
    # )
    # if st.button("Generate Image"):
    #     with st.spinner("Generating Image..."):
    #         image = generate_image_local_sdxl(base, prompt, refiner)
    #         st.image(image, caption="Generated Image", use_column_width=True)

    ## Text Generation

    # st.title = "Text Generation using OpenAI API"
    # model_path = "offline/models/models--TheBloke--Llama-2-7B-GGUF/snapshots/b4e04e128f421c93a5f1e34ac4d7ca9b0af47b80"
    # model_file = "llama-2-7b.Q4_K_M.gguf"
    # model = load_model_llama2_gguf(model_path, model_file)
    # prompt = st.text_input(
    #     "Enter a prompt", value="Where is Amrita Vishwa Vidhyapeetam"
    # )
    # text_area_placeholder = st.empty()
    # if st.button("Generate Text"):
    #     generate_text_llama2_gguf(model, prompt, text_area_placeholder)

    st.title = "Text Generation using Mistralai"
    model_path = ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_mistralai(model_path)
    prompt = st.text_input(
        "Enter a prompt", value="Where is Amrita Vishwa Vidhyapeetam"
    )
    text_area_placeholder = st.empty()
    if st.button("Generate Text"):
        generate_text_streamlit_mistralai(
            model, tokenizer, prompt, text_area_placeholder
        )


if __name__ == "__main__":
    main()
