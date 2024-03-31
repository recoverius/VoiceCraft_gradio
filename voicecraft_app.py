import os
import torch
import torchaudio
import gradio as gr
from data.tokenizer import AudioTokenizer, TextTokenizer
import models.voicecraft as voicecraft
from inference_tts_scale import inference_one_sample
import gc
import GPUtil

def align_audio(orig_audio, orig_transcript, temp_folder, align_temp):
    os.makedirs(temp_folder, exist_ok=True)
    os.system(f"cp {orig_audio} {temp_folder}")
    filename = os.path.splitext(orig_audio.split("/")[-1])[0]
    with open(f"{temp_folder}/{filename}.txt", "w") as f:
        f.write(orig_transcript)
    
    os.makedirs(align_temp, exist_ok=True)
    os.system(f"mfa align -j 1 --output_format csv {temp_folder} english_us_arpa english_us_arpa {align_temp}")
    
    return f"{temp_folder}/{filename}.wav", f"{temp_folder}/{filename}.txt", f"{align_temp}/{filename}.csv"

def generate_voice(audio_fn, transcript_fn, align_fn, cut_off_sec, target_transcript, voicecraft_name, encodec_fn, top_k, top_p, temperature, stop_repetition, kvcache, codec_audio_sr, codec_sr, silence_tokens, sample_batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt_fn = f"./pretrained_models/{voicecraft_name}"
    if not os.path.exists(ckpt_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/{voicecraft_name}\\?download\\=true")
        os.system(f"mv {voicecraft_name}\\?download\\=true ./pretrained_models/{voicecraft_name}")
    
    if not os.path.exists(encodec_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
        os.system(f"mv encodec_4cb2048_giga.th ./pretrained_models/encodec_4cb2048_giga.th")
    
    ckpt = torch.load(ckpt_fn, map_location="cpu")
    model = voicecraft.VoiceCraft(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    
    phn2num = ckpt['phn2num']
    text_tokenizer = TextTokenizer(backend="espeak")
    audio_tokenizer = AudioTokenizer(signature=encodec_fn)
    
    info = torchaudio.info(audio_fn)
    prompt_end_frame = int(cut_off_sec * info.sample_rate)
    
    decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition, 'kvcache': kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr, "silence_tokens": silence_tokens, "sample_batch_size": int(sample_batch_size)}
    concated_audio, gen_audio = inference_one_sample(model, ckpt["config"], phn2num, text_tokenizer, audio_tokenizer, audio_fn, target_transcript, device, decode_config, prompt_end_frame)
    
    concated_audio, gen_audio = inference_one_sample(model, ckpt["config"], phn2num, text_tokenizer, audio_tokenizer, audio_fn, target_transcript, device, decode_config, prompt_end_frame)
    
    concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()
    
    # Cleanup memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return concated_audio, gen_audio

def main(orig_audio, orig_transcript, target_transcript, cut_off_sec, voicecraft_name, encodec_fn, top_k, top_p, temperature, stop_repetition, kvcache, codec_audio_sr, codec_sr, silence_tokens, sample_batch_size):
    # Cleanup memory at the beginning
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get the GPU with the most available memory
    gpus = GPUtil.getGPUs()
    gpu = gpus[0]  # Assuming you want to use the first GPU
    
    temp_folder = "./demo/temp"
    align_temp = f"{temp_folder}/mfa_alignments"
    
    audio_fn, transcript_fn, align_fn = align_audio(orig_audio, orig_transcript, temp_folder, align_temp)
    
    # Monitor VRAM usage
    while True:
        # Get current VRAM usage
        vram_usage = gpu.memoryUsed
        total_vram = gpu.memoryTotal
        
        # Check if VRAM usage exceeds a threshold (e.g., 90% of total VRAM)
        if vram_usage > 0.9 * total_vram:
            print("VRAM usage is high. Taking appropriate actions...")
            # Take actions to reduce VRAM usage, such as reducing batch size or clearing memory
            sample_batch_size = max(1, sample_batch_size // 2)
            torch.cuda.empty_cache()
            gc.collect()
        
        concated_audio, gen_audio = generate_voice(audio_fn, transcript_fn, align_fn, cut_off_sec, target_transcript, voicecraft_name, encodec_fn, top_k, top_p, temperature, stop_repetition, kvcache, codec_audio_sr, codec_sr, silence_tokens, sample_batch_size)
        
        # Save the audio Tensors as files
        concated_audio_path = "./demo/temp/concated_audio.wav"
        gen_audio_path = "./demo/temp/gen_audio.wav"
        torchaudio.save(concated_audio_path, concated_audio, int(codec_audio_sr))
        torchaudio.save(gen_audio_path, gen_audio, int(codec_audio_sr))
        
        # Break the loop if generation is successful
        break
    
    return concated_audio_path, gen_audio_path
    

iface = gr.Interface(
    fn=main,
    inputs=[
        gr.Textbox(label="Original Audio File"),
        gr.Textbox(label="Original Transcript"),
        gr.Textbox(label="Target Transcript"),
        gr.Number(label="Cut-off Second", value=3.01),
        gr.Textbox(label="VoiceCraft Model", value="giga830M.pth"),
        gr.Textbox(label="Encodec File", value="./pretrained_models/encodec_4cb2048_giga.th"),
        gr.Number(label="Top K", value=0),
        gr.Number(label="Top P", value=0.8),
        gr.Number(label="Temperature", value=1),
        gr.Number(label="Stop Repetition", value=1),
        gr.Number(label="KV Cache", value=1),
        gr.Number(label="Codec Audio Sample Rate", value=16000),
        gr.Number(label="Codec Sample Rate", value=50),
        gr.CheckboxGroup(label="Silence Tokens", choices=[1388, 1898, 131], value=[1388, 1898, 131]),
        gr.Number(label="Sample Batch Size", value=4),
    ],
    outputs=[
        gr.Audio(label="Concatenated Audio"),
        gr.Audio(label="Generated Audio"),
    ],
    title="VoiceCraft Voice Generation",
    description="Generate voice using VoiceCraft model with alignment and parameter control.",
)

if __name__ == "__main__":
    iface.launch()
