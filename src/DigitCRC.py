import cv2
import numpy as np
import math
import pickle
import pathlib
from tkinter import *
from mls import sigmoid
from scipy import ndimage
import PIL
from PIL import ImageDraw

class DCRCGui(object):
    DEFAULT_PEN_SIZE = 30.0
    DEFAULT_COLOR = 'black'
    DEFAULT_WIDTH = 300
    DEFAULT_HEIGHT = 300

    def __init__(self):
        with open('classifiers/1.model','rb') as f:
            self._lr = pickle.load(f)
        with open('classifiers/2.model','rb') as f:
            self._nn = pickle.load(f)
        self._svm = []
        for i in range(10):
            with open('classifiers/3_'+str(i+1)+'.model','rb') as f:
                self._svm.append(pickle.load(f))

        self.root = Tk()

        self.icon_photo = PhotoImage(file = 'mnist.png')

        self.clear_button = Button(self.root, text="Clear",command=self.clear)
        self.clear_button.grid(row=0, column=0)

        self.choose_size_button = Scale(self.root, from_=1, to=50, orient=HORIZONTAL)
        self.choose_size_button.set(self.DEFAULT_PEN_SIZE)
        self.choose_size_button.grid(row=0, column=1)

        self.clear_button = Button(self.root, text="Predict",command=self.predict)
        self.clear_button.grid(row=0, column=4)

        self.canvas0 = Canvas(self.root, bg='white', width=self.DEFAULT_WIDTH, height=self.DEFAULT_HEIGHT)
        self.canvas0.grid(row=1, columnspan=5)

        self.target = PIL.Image.new("RGB", (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.target)

        self.lr = Label(self.root)
        self.lr.grid(row=2, column=0, columnspan=2)
        self.nn = Label(self.root)
        self.nn.grid(row=3, column=0, columnspan=2)
        self.svm = Label(self.root)
        self.svm.grid(row=4, column=0, columnspan=2)

        self.lr_result = Label(self.root)
        self.lr_result.grid(row=2, column=2, columnspan=2)
        self.nn_result = Label(self.root)
        self.nn_result.grid(row=3, column=2, columnspan=2)
        self.svm_result = Label(self.root)
        self.svm_result.grid(row=4, column=2, columnspan=2)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.root.title("DigitCRC – Aguilar")
        self.root.resizable(False,False)
        self.root.iconphoto(False,self.icon_photo)
        self.old_x = None
        self.old_y = None
        self.line_width =  self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.lr.config(text="AI Algoritm 1:")
        self.nn.config(text="AI Algoritm 2:")
        self.svm.config(text="AI Algoritm 3:")
        self.lr_result.config(text="NA")
        self.nn_result.config(text="NA")
        self.svm_result.config(text="NA")
        self.eraser_on = False
        self.canvas0.bind('<B1-Motion>', self.paint)
        self.canvas0.bind('<ButtonRelease-1>', self.reset)
        if (pathlib.Path.cwd() / 'target').exists() is False:
            path = pathlib.Path("target/")
            path.mkdir(parents=True, exist_ok=True)

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.canvas0.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line([(event.x - 5), (event.y - 5), (event.x + 5), (event.y + 5)],
                               fill="black", width=self.line_width, 
                               joint="curve")
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def clear(self):
        self.canvas0.delete('all')
        self.draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
        self.lr_result.config(text="NA")
        self.nn_result.config(text="NA")
        self.svm_result.config(text="NA")

    def predict(self):
        self.target.save("target/target.png")
        x = self._preproc()
        if x is None:
            self.lr_result.config(text=f"NA")
            self.nn_result.config(text=f"NA")
            self.svm_result.config(text=f"NA")
            return
        x = x[None,:]
        nn_p = self._nn.predict(x)
        lr_p = self._lr.predict_onevsall(x)
        svm_p_raw = np.array([ self._svm[i].predict(x)[:,0] for i in range(10) ]).T
        svm_p = np.zeros(2)[None,:]
        svm_p[0,0] = svm_p_raw.max(axis=1)
        svm_p[0,1] = svm_p_raw.argmax(axis=1)
        self.lr_result.config(text=f"{int(lr_p[0,1])} – {(lr_p[0,0]*100):.0f}%")
        self.nn_result.config(text=f"{int(nn_p[0,1])} – {(nn_p[0,0]*100):.0f}%")
        self.svm_result.config(text=f"{int(svm_p[0,1])} – {(sigmoid(nn_p[0,0])*100):.0f}%")

    def _getBestShift(self, img):
        cy,cx = ndimage.measurements.center_of_mass(img)

        rows,cols = img.shape
        shiftx = np.round(cols/2.0-cx).astype(int)
        shifty = np.round(rows/2.0-cy).astype(int)

        return shiftx,shifty

    def _shift(self, img, sx, sy):
        rows,cols = img.shape
        M = np.float32([[1,0,sx],[0,1,sy]])
        shifted = cv2.warpAffine(img,M,(cols,rows))
        return shifted
    
    def _preproc(self):
        target_gray = cv2.imread("target/target.png",cv2.IMREAD_GRAYSCALE)
        target_gray = cv2.resize(255-target_gray, (28,28))
        try:
            while np.sum(target_gray[0]) == 0:
                target_gray = target_gray[1:]
            while np.sum(target_gray[:,0]) == 0:
                target_gray = np.delete(target_gray,0,1)
            while np.sum(target_gray[-1]) == 0:
                target_gray = target_gray[:-1]
            while np.sum(target_gray[:,-1]) == 0:
                target_gray = np.delete(target_gray,-1,1)
        except:
            return None
        rows,cols = target_gray.shape
        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            target_gray = cv2.resize(target_gray, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            target_gray = cv2.resize(target_gray, (cols, rows))
        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        target_gray = np.lib.pad(target_gray,(rowsPadding,colsPadding),'constant')
        shiftx,shifty = self._getBestShift(target_gray)
        shifted = self._shift(target_gray,shiftx,shifty)
        target_gray = shifted
        cv2.imwrite("target/target-proc.png", target_gray)
        flatten_target_gray = target_gray.flatten() / 255.0

        return flatten_target_gray

if __name__ == '__main__':
    DCRCGui()