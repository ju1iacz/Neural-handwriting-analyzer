from keras.models import load_model
import numpy as np
from tkinter import *
from tkinter import messagebox
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image

model = load_model('model.h5')

def predict_digit(img):
    img = img.resize((32, 32))
    img = img.convert('L')              #RGB to grey
    img = np.array(img)
    img = img.reshape(1, 32, 32, 1)
    img = img/255                       #black and white
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

def open():
    app = Canvas()

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.canva = tk.Button(self, text = "Start", font = ("Helvetica", 18), command = open)
        self.help = tk.Button(self, text = "Help", font = ("Helvetica", 18), command = self.get_help)
        self.exit = tk.Button(self, text = "Exit", font = ("Helvetica", 18), command = self.go_exit)

        self.canva.grid(row = 0, column = 0, pady = 10, padx = 100)
        self.help.grid(row = 1, column = 0, pady = 10)
        self.exit.grid(row = 2, column = 0, pady = 10)

    def get_help(self):     #'Help' button
        messagebox.showinfo("Help", "To start entering digits, press the 'Start' button."
                                    "\nDraw a digit in the white field and press the 'Recognize' button to recognize the digit."
                                    "\nThe recognized digit will be displayed on the right."
                                    "\nPress the 'Clear' button to clear the field and start over.")

    def go_exit(self):      #'Exit' button
        response = messagebox.askyesno("Exit", "Do you want to exit?")
        if response == 1:
            self.quit()


class Canvas(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        self.canvas = tk.Canvas(self, width = 300, height = 400, bg = "white", cursor = "target")
        self.label = tk.Label(self, text = "...", font = ("Helvetica", 36))
        self.classify = tk.Button(self, text = "Recognize", font = ("Helvetica", 18), command = self.classify_handwriting)
        self.clear = tk.Button(self, text = "Clear", font = ("Helvetica", 18), command = self.clear_all)

        self.canvas.grid(row = 0, column = 0, pady = 2, sticky = W)
        self.label.grid(row = 0, column = 1, pady = 2, padx = 2, sticky = N)
        self.classify.grid(row = 0, column = 1, sticky = S, padx = 10, pady = 10)
        self.clear.grid(row = 0, column = 1)

        self.canvas.bind('<B1-Motion>', self.draw_lines)

    def clear_all(self):        #'Clear' button
        self.canvas.delete('all')
        self.label.configure(text = "...")

    def classify_handwriting(self):     #'Recognize' button
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        im = ImageGrab.grab(rect)

        digit, acc = predict_digit(im)
        self.label.configure(text = str(digit) + '\n' + str(int(acc*100)) + '%')    #displaying the result

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill = 'black')

app = App()
mainloop()