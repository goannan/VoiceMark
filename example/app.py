import gradio as gr
import json
import os
import torchaudio
from infer import (
    WatermarkSolver,
    hamming_distance
)

# Predefined watermarks (instead of loading from a JSON file)
watermarks = {
    "VoiceMark": "1000010101010011",
    "Voice Cloning": "1111111001000010",
    "Speech Security": "1011101100001110",
    "Audio Watermarking": "0110110011100010",
    "Deep Learning": "0000100111111000",
    "Artificial Intelligence": "0010000100011111",
    "Hello World": "0001111101110001",
    "Happy New Year": "1101011011011101",
    "World Peace": "0011110010011110",
    "Good Morning": "0000001011000010",
}

# Initialize WatermarkSolver model
solver = WatermarkSolver()
solver.load_model(checkpoint_dir="../", checkpoint_name="voicemark.pth", strict=True)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(
        "## VoiceMark: Zero-Shot Voice Cloning-Resistant Watermarking Approach Leveraging Speaker-Specific Latents"
    )
    with gr.Column():
        gr.Image(
            value="voicemark_overview.png",
            width=925,
            height=487,
            elem_id="overview_image",
            label="overview"
        )
        # Step 1: Upload audio and select watermark
        gr.HTML("<h3 style='text-align: center;'>The overall architecture of our proposed VoiceMark</h3>")

    # Step 1: Upload audio and select watermark
    gr.Markdown(
        """
        **Step 1**: Upload an audio file or select one from the provided samples, choose a watermark, and generate the watermarked audio.
        """
    )
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Upload Audio", type="filepath")
            
            gr.Examples(
                examples=[
                    ["audios/1.wav"],
                    ["audios/2.wav"],
                    ["audios/3.wav"],
                    ["audios/4.wav"],
                    ["audios/5.wav"],
                ],
                inputs=audio_input,
                label="Sample Audios (Click to Use)"
            )

        with gr.Column():
            audio_output = gr.Audio(label="Watermarked Audio", type="filepath")
            watermark_list = gr.Dropdown(
                label="Select Watermark", choices=list(watermarks.keys()), interactive=True
            )
            add_watermark_button = gr.Button("Add Watermark to Audio")

    # Step 2: TTS tools demo links
    gr.Markdown(
        """
        **Step 2**: Download the generated watermarked audio, then use Zero-Shot Voice Cloning tools to generate the cloned audio. Some available tools are:
        - [CosyVoice2: Scalable Streaming Speech Synthesis with Large Language Models](https://www.modelscope.cn/studios/iic/CosyVoice2-0.5B)
        - [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
        - [MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer](https://huggingface.co/spaces/amphion/maskgct)
        """
    )

    # Step 3: Upload cloned audio to decode watermark
    gr.Markdown(
        """
        **Step 3**: Upload the cloned audio and decode your watermark.
        """
    )

    with gr.Row():
        decode_audio_input = gr.Audio(label="Upload Cloned Audio", type="filepath")
        with gr.Column():
            decoded_watermark_output = gr.Textbox(label="Decoded Watermark")
            decode_button = gr.Button("Decode Watermark")

    def process_audio(audio_path, watermark_text):
        if not audio_path:
            return "No audio selected. Please upload or select a sample."
        try:
            watermarked_audio = solver.infer_for_ui(
                audio_path, watermarks[watermark_text]
            )
            return watermarked_audio
        except ValueError as e:
            return str(e)

    add_watermark_button.click(
        process_audio,
        inputs=[audio_input, watermark_list],
        outputs=audio_output
    )

    def decode_watermark(audio_path):
        try:
            detect_prob, decoded_id = solver.decode_for_ui(audio_path)
            if detect_prob < 1e-2:
                return "No matching watermark found"
            closest_match = None
            min_distance = float("inf")
            for text, id_bin in watermarks.items():
                distance = hamming_distance(decoded_id, id_bin, base=16)
                if distance < min_distance:
                    closest_match = text
                    min_distance = distance
            if min_distance < 10:
                return closest_match
            return "No matching watermark found"
        except ValueError as e:
            return str(e)

    decode_button.click(
        decode_watermark, inputs=decode_audio_input, outputs=decoded_watermark_output
    )

# Launch the Gradio app
demo.launch()
