from openai import OpenAI
from apikey import apikey
import os
import streamlit as st
from PIL import Image
import requests
from io import BytesIO


def setup_openai(apikey):
    os.environ["OPENAI_API_KEY"] = apikey
    OpenAI.api_key = apikey
    client = OpenAI()
    return client


############################################
# OpenAI Audio to Text
############################################


def generate_text_from_audio_openai(
    client, audio_file, model="whisper-1", response_format="text"
):
    response = client.audio.transcriptions.create(
        model=model,
        file=audio_file,
        response_format=response_format,
    )
    return response


############################################
# OpenAI Image Generation
############################################


def generate_image_openai(client, prompt, model="dall-e-3", size="1024x1024", n=1):
    response = client.images.generate(model="dall-e-3", prompt=prompt, size=size, n=n)
    image_url = response.data[0].url
    image = requests.get(image_url)
    image = Image.open(BytesIO(image.content))
    return image


############################################
# OpenAI Text Generation
############################################


def generate_text_openai(
    client,
    prompt="About Amrita Vishwa Vidhyapeetam",
    text_area_placeholder=None,
    model="gpt-3.5-turbo",
    temperature=0.5,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stream=True,
    html=False,
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stream=stream,
    )
    complete_response = []
    for chunk in response:
        if chunk.choices[0].delta.content:
            complete_response.append(chunk.choices[0].delta.content)
            result_string = "".join(complete_response)  # Join without additional spaces

            # auto scroll
            lines = result_string.split("\n").count("\n") + 1
            avg_chars_per_line = 50
            lines += len(result_string) // avg_chars_per_line
            height_per_line = 20
            total_height = lines * height_per_line

            if text_area_placeholder:
                if html:
                    text_area_placeholder.markdown(
                        result_string, unsafe_allow_html=True
                    )
                else:
                    text_area_placeholder.text_area(
                        "Generated Text", value=result_string
                    )

    result_string = "".join(complete_response)
    return result_string


def main():
    client = setup_openai(apikey)

    ## Text Generation

    st.title = "Text Generation using OpenAI API"
    prompt = st.text_input(
        "Enter a prompt", value="Where is Amrita Vishwa Vidhyapeetam"
    )
    text_area_placeholder = st.empty()
    if st.button("Generate Text"):
        with st.spinner("Generating Text..."):
            response = generate_text_openai(
                client,
                prompt=prompt,
                text_area_placeholder=text_area_placeholder,
            )

    # ## Image Generation
    # st.title = "Image Generation using OpenAI API"
    # prompt = st.text_input(
    #     "Enter a prompt", value="Amrita Vishwa Vidhyapeetam university"
    # )
    # if st.button("Generate Image"):
    #     with st.spinner("Generating Image..."):
    #         image = generate_image_openai(client, prompt)
    #         st.image(image, caption="Generated Image", use_column_width=True)

    # ## Audio Transcription

    # st.title = "Audio Transcription using OpenAI API"
    # audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

    # if audio_file:
    #     if st.button("Transcribe"):
    #         st.audio(audio_file, format="audio/wav")
    #         with st.spinner("Transcribing..."):
    #             response = generate_text_from_audio_openai(client, audio_file)
    #             st.write(response)


if __name__ == "__main__":
    main()
