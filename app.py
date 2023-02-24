from __future__ import annotations
import gradio as gr
import os
import cv2
import numpy as np
from PIL import Image
from moviepy.editor import *
from share_btn import community_icon_html, loading_icon_html, share_js

import pathlib
import shlex
import subprocess

if os.getenv('SYSTEM') == 'spaces':
    with open('patch') as f:
        subprocess.run(shlex.split('patch -p1'), stdin=f, cwd='ControlNet')

base_url = 'https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/'

names = [
    'body_pose_model.pth',
    'dpt_hybrid-midas-501f0c75.pt',
    'hand_pose_model.pth',
    'mlsd_large_512_fp32.pth',
    'mlsd_tiny_512_fp32.pth',
    'network-bsds500.pth',
    'upernet_global_small.pth',
]

for name in names:
    command = f'wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/{name} -O {name}'
    out_path = pathlib.Path(f'ControlNet/annotator/ckpts/{name}')
    if out_path.exists():
        continue
    subprocess.run(shlex.split(command), cwd='ControlNet/annotator/ckpts/')

from model import Model
model = Model()


def controlnet(i, prompt, control_task, seed_in, ddim_steps, scale):
    img= Image.open(i)
    np_img = np.array(img)
    
    a_prompt = "best quality, extremely detailed"
    n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    num_samples = 1
    image_resolution = 512
    detect_resolution = 512
    eta = 0.0
    low_threshold = 100
    high_threshold = 200
    value_threshold = 0.1
    distance_threshold = 0.1
    bg_threshold = 0.4
    
    if control_task == 'Canny':
        result = model.process_canny(np_img, prompt, a_prompt, n_prompt, num_samples,
                image_resolution, ddim_steps, scale, seed_in, eta, low_threshold, high_threshold)
    elif control_task == 'Depth':
        result = model.process_depth(np_img, prompt, a_prompt, n_prompt, num_samples,
            image_resolution, detect_resolution, ddim_steps, scale, seed_in, eta)
    elif control_task == 'Hed':
        result = model.process_hed(np_img, prompt, a_prompt, n_prompt, num_samples,
            image_resolution, detect_resolution, ddim_steps, scale, seed_in, eta)
    elif control_task == 'Hough':
        result = model.process_hough(np_img, prompt, a_prompt, n_prompt, num_samples,
            image_resolution, detect_resolution, ddim_steps, scale, seed_in, eta, value_threshold,
                      distance_threshold)
    elif control_task == 'Normal':
        result = model.process_normal(np_img, prompt, a_prompt, n_prompt, num_samples,
            image_resolution, detect_resolution, ddim_steps, scale, seed_in, eta, bg_threshold)
    elif control_task == 'Pose':
        result = model.process_pose(np_img, prompt, a_prompt, n_prompt, num_samples,
            image_resolution, detect_resolution, ddim_steps, scale, seed_in, eta)
    elif control_task == 'Scribble':
        result = model.process_scribble(np_img, prompt, a_prompt, n_prompt, num_samples,
            image_resolution, ddim_steps, scale, seed_in, eta)
    elif control_task == 'Seg':
        result = model.process_seg(np_img, prompt, a_prompt, n_prompt, num_samples,
            image_resolution, detect_resolution, ddim_steps, scale, seed_in, eta)
    
    #print(result[0])
    processor_im = Image.fromarray(result[0])
    processor_im.save("process_" + control_task + "_" + str(i) + ".jpeg")
    im = Image.fromarray(result[1])
    im.save("your_file" + str(i) + ".jpeg")
    return "your_file" + str(i) + ".jpeg", "process_" + control_task + "_" + str(i) + ".jpeg"


def get_frames(video_in):
    frames = []
    #resize the video
    clip = VideoFileClip(video_in)
    
    #check fps
    if clip.fps > 30:
        print("vide rate is over 30, resetting to 30")
        clip_resized = clip.resize(height=512)
        clip_resized.write_videofile("video_resized.mp4", fps=30)
    else:
        print("video rate is OK")
        clip_resized = clip.resize(height=512)
        clip_resized.write_videofile("video_resized.mp4", fps=clip.fps)
    
    print("video resized to 512 height")
    
    # Opens the Video file with CV2
    cap= cv2.VideoCapture("video_resized.mp4")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("video fps: " + str(fps))
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('kang'+str(i)+'.jpg',frame)
        frames.append('kang'+str(i)+'.jpg')
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()
    print("broke the video into frames")
    
    return frames, fps


def convert(gif):
    if gif != None:
        clip = VideoFileClip(gif.name)
        clip.write_videofile("my_gif_video.mp4")
        return "my_gif_video.mp4"
    else:
        pass


def create_video(frames, fps, type):
    print("building video result")
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(type + "_result.mp4", fps=fps)
    
    return type + "_result.mp4"


def infer(prompt,video_in, control_task, seed_in, trim_value, ddim_steps, scale, gif_import):
    print(f"""
    ———————————————
    {prompt}
    ———————————————""")
    
    # 1. break video into frames and get FPS
    break_vid = get_frames(video_in)
    frames_list= break_vid[0]
    fps = break_vid[1]
    n_frame = int(trim_value*fps)
    
    if n_frame >= len(frames_list):
        print("video is shorter than the cut value")
        n_frame = len(frames_list)
    
    # 2. prepare frames result arrays
    processor_result_frames = []
    result_frames = []
    print("set stop frames to: " + str(n_frame))
    
    for i in frames_list[0:int(n_frame)]:
        controlnet_img = controlnet(i, prompt,control_task, seed_in, ddim_steps, scale)
        #images = controlnet_img[0]
        #rgb_im = images[0].convert("RGB")
  
        # exporting the image
        #rgb_im.save(f"result_img-{i}.jpg")
        processor_result_frames.append(controlnet_img[1])
        result_frames.append(controlnet_img[0])
        print("frame " + i + "/" + str(n_frame) + ": done;")

    processor_vid = create_video(processor_result_frames, fps, "processor")
    final_vid = create_video(result_frames, fps, "final")

    files = [processor_vid, final_vid]
    if gif_import != None:
        final_gif = VideoFileClip(final_vid)
        final_gif.write_gif("final_result.gif")
        final_gif = "final_result.gif"

        files.append(final_gif)
    print("finished !")
    
    return final_vid, gr.Accordion.update(visible=True), gr.Video.update(value=processor_vid, visible=True), gr.File.update(value=files, visible=True), gr.Group.update(visible=True)


def clean():
    return gr.Accordion.update(visible=False),gr.Video.update(value=None, visible=False), gr.Video.update(value=None), gr.File.update(value=None, visible=False), gr.Group.update(visible=False)

title = """
    <div style="text-align: center; max-width: 700px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
        "
        >
        <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
            ControlNet Video
        </h1>
        </div>
        <p style="margin-bottom: 10px; font-size: 94%">
        Apply ControlNet to a video 
        </p>
    </div>
"""

article = """
    
    <div class="footer">
        <p>
        Follow <a href="https://twitter.com/fffiloni" target="_blank">Sylvain Filoni</a> for future updates 🤗
        </p>
    </div>
    <div id="may-like-container" style="display: flex;justify-content: center;flex-direction: column;align-items: center;margin-bottom: 30px;">
        <p>You may also like: </p>
        <div id="may-like-content" style="display:flex;flex-wrap: wrap;align-items:center;height:20px;">
            
            <svg height="20" width="148" style="margin-left:4px;margin-bottom: 6px;">       
                 <a href="https://huggingface.co/spaces/fffiloni/Pix2Pix-Video" target="_blank">
                    <image href="https://img.shields.io/badge/🤗 Spaces-Pix2Pix_Video-blue" src="https://img.shields.io/badge/🤗 Spaces-Pix2Pix_Video-blue.png" height="20"/>
                 </a>
            </svg>
            
        </div>
    
    </div>
    
"""

with gr.Blocks(css='style.css') as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        with gr.Row():
            with gr.Column():
                video_inp = gr.Video(label="Video source", source="upload", type="filepath", elem_id="input-vid")
                video_out = gr.Video(label="ControlNet video result", elem_id="video-output")
                with gr.Accordion("Detailed results", visible=False) as detailed_result:
                    prep_video_out = gr.Video(label="Preprocessor video result", visible=False, elem_id="prep-video-output")
                    files = gr.File(label="Files can be downloaded ;)", visible=False)
                with gr.Group(elem_id="share-btn-container", visible=False) as share_group:
                    community_icon = gr.HTML(community_icon_html)
                    loading_icon = gr.HTML(loading_icon_html)
                    share_button = gr.Button("Share to community", elem_id="share-btn")
            with gr.Column():
                #status = gr.Textbox()
                
                prompt = gr.Textbox(label="Prompt", placeholder="enter prompt", show_label=True, elem_id="prompt-in")
                with gr.Row():
                    control_task = gr.Dropdown(label="Control Task", choices=["Canny", "Depth", "Hed", "Hough", "Normal", "Pose", "Scribble", "Seg"], value="Pose", multiselect=False, elem_id="controltask-in")
                    seed_inp = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, value=123456, elem_id="seed-in")
                with gr.Row():
                    
                    trim_in = gr.Slider(label="Cut video at (s)", minimun=1, maximum=5, step=1, value=1)
                with gr.Accordion("Advanced Options", open=False):
                    
                    ddim_steps = gr.Slider(label='Steps',
                                       minimum=1,
                                       maximum=100,
                                       value=20,
                                       step=1)
                    scale = gr.Slider(label='Guidance Scale',
                                  minimum=0.1,
                                  maximum=30.0,
                                  value=9.0,
                                  step=0.1)
                    
                    gif_import = gr.File(label="import a GIF instead", file_types=['.gif'])
                    gif_import.change(convert, gif_import, video_inp, queue=False)
                
                submit_btn = gr.Button("Generate ControlNet video")
                
                gr.HTML("""
                <a style="display:inline-block" href="https://huggingface.co/spaces/fffiloni/ControlNet-Video?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a> 
                work with longer videos / skip the queue: 
                """, elem_id="duplicate-container")
        
        inputs = [prompt,video_inp,control_task, seed_inp, trim_in, ddim_steps, scale, gif_import]
        outputs = [video_out, detailed_result, prep_video_out, files, share_group]
        #outputs = [status]
        
        
        gr.HTML(article)

    submit_btn.click(clean, inputs=[], outputs=[detailed_result, prep_video_out, video_out, files, share_group], queue=False)
    submit_btn.click(infer, inputs, outputs)
    share_button.click(None, [], [], _js=share_js)

    
    
demo.queue(max_size=12).launch()