import tkinter as tk
import customtkinter as ctk 

from PIL import ImageTk
from authtoken import auth_token
#from accelerate import Accelerator
#accelerator = Accelerator()  
#4device = accelerator.device{}
import torch
from diffusers import StableDiffusionPipeline 

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud") 
ctk.set_appearance_mode("dark") 

# Entry for the prompt
# Entry for the prompt (removed text_font argument)
prompt = ctk.CTkEntry(master=app, height=40, width=512, text_color="black", fg_color="white", font=("Arial", 20))
prompt.place(x=10, y=10)

prompt.place(x=10, y=10)

# Label to display the generated image
lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

# Load the Stable Diffusion model on the CPU
modelid = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(modelid, use_auth_token=auth_token)
pipe.to("cpu")  # Explicitly set the device to CPU

# Generate function
def generate(): 
    # Generate the image using CPU
    image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    # Save and display the generated image
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img) 

# Button to trigger image generation
trigger = ctk.CTkButton(master=app, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate") 
trigger.place(x=206, y=60) 

# Run the app
app.mainloop()
