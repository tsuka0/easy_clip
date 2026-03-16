import tkinter as tk
import cv2
from PIL import Image, ImageTk


class CanvasView:

    def __init__(self, root):

        self.canvas = tk.Canvas(root, bg="#1e1e1e")
        self.canvas.pack(fill="both", expand=True)

    def draw(self, frame, mask):

        img = frame.copy()

        if mask is not None:

            overlay = img.copy()
            overlay[mask] = (0,255,0)

            img = cv2.addWeighted(img,1,overlay,0.5,0)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)

        img = img.resize((960,640))

        img = ImageTk.PhotoImage(img)

        self.canvas.create_image(20,20,anchor="nw",image=img)

        self.canvas.image = img
