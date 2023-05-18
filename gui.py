import os
import cv2
import json

import tkinter as tk 
from PIL import Image, ImageTk

import torch
from torch import nn
from model_training import transform_image

from inception_resnet_v1 import InceptionResnetV1

class App(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        self.title('Face Recognition System')
        self.geometry("700x550")

        self.protocol("WM_DELETE_WINDOW", self.save_database)

        self.camera = MainCamera(self, borderwidth=2, relief="ridge")
        self.camera.pack(side="top", fill='both', expand=True)

        frame = tk.Frame()
        frame.pack(side='top', expand=True)

        button_database = tk.Button(frame, text="Database", width = 15, font=(1), command=lambda: DatabaseFrame(self))
        button_database.pack(side='right', expand=True, padx=75)

        button_scan = tk.Button(frame, text="Scan", width = 15, font=(1), command=self.add_unknown_faces)
        button_scan.pack(side='left', expand=True, padx=75)

    def save_database(self):
        with open('database.json', 'w') as f:
            json.dump(database, f)

        self.destroy()

    def add_unknown_faces(self):
        for image, embedding in self.unknown_faces:
            FaceFrame(self, image, embedding)


    def get_identity(self, vector):
        global threshold

        encoding = vector
        min_dist = 100
        
        for name, data in database.items():
            db_enc = torch.FloatTensor(data["embedding"])
                    
            dist_func = nn.PairwiseDistance()
            dist = dist_func(encoding, db_enc)
            
            if dist < min_dist:
                min_dist = dist.item()
                identity = name
                
        if not min_dist < threshold:
            identity = None
            min_dist = 0
            
        return min_dist, identity 


class FaceFrame(tk.Toplevel):
    def __init__(self, parent, image, vec):
        tk.Toplevel.__init__(self)
        self.wm_title(f"Unknown Face")

        self.image = image
        self.embedding = vec 
        self.geometry("300x350")

        canvas = tk.Canvas(self, width=220, height=220, borderwidth=2, relief="ridge")
        canvas.pack(side='top')

        pillow_image = image.resize((200, 200))
        image =  ImageTk.PhotoImage(master = canvas, image=pillow_image)
        canvas.create_image(115, 115, image = image)

        label = tk.Label(self, text="Enter name: ", font=(1))
        label.pack(side = 'top', pady=5)
        self.entry = tk.Entry(self, bd = 5)
        self.entry.pack(side = 'top', pady=5)
        
        save_button = tk.Button(self, text="Save", width = 15, font=(1), command=self.save_face)
        save_button.pack(side = 'top', pady=5)

        parent.wait_window(self)

    def save_face(self):
        global database

        name = self.entry.get()
        path = f"stored_faces/{name}.jpg"

        database[name] = {'image_path': f"stored_faces/{name}.jpg", 'embedding': self.embedding.tolist()}
        self.image.save(path)
        self.destroy()


class MainCamera(tk.Frame):

    def __init__(self, parent, **kwargs):
        tk.Frame.__init__(self, **kwargs)

        vid = cv2.VideoCapture(0)
        if (vid.isOpened() == False):
            print("Web Camera not detected")

        vid.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
  
        label_widget = tk.Label(self)
        label_widget.pack(expand=True)

        def show_camera():
            _, frame = vid.read()
            self.color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            self.faces = faceCascade.detectMultiScale(self.color_frame, 1.2, 5)

            parent.unknown_faces = []
            parent.faces_names = []

            for (x, y, w, h) in self.faces:
                numpy_image = self.color_frame[y : y + h, x : x + w]
                pillow_image = Image.fromarray(numpy_image)

                vec = model(transform_image(pillow_image).unsqueeze(0))
                similarity, name = parent.get_identity(vec)

                if name == None: parent.unknown_faces.append((pillow_image, vec))
                else: parent.faces_names.append(name)

                text = "{}: {:.2f}".format(name, similarity)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)

                y = y - 10 if y - 10 > 10 else y + 10
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            captured_image = Image.fromarray(color_image)
            photo_image = ImageTk.PhotoImage(image=captured_image)

            label_widget.photo_image = photo_image
            label_widget.configure(image=photo_image)
            label_widget.after(10, show_camera)

        show_camera()


class DatabaseFrame(tk.Toplevel):

    def __init__(self, parent):
        tk.Toplevel.__init__(self)
        self.wm_title(f"Database")

        self.geometry("450x320")

        clear_button = tk.Button(self, text="Clear all", width = 15, font=(1), command=self.clear_all)
        clear_button.pack(side = 'bottom', pady=5, expand=True, fill='both', padx=5)

        frame = tk.Frame(self, borderwidth=2, relief="ridge")
        frame.pack(side='left', padx=5, pady=5)

        self.user_list = tk.Variable(value=list(database.keys()))

        self.listbox = tk.Listbox(frame, listvariable=self.user_list, bd=3, selectmode="single", relief="flat", font=(1), height=15)
        self.listbox.pack(side="left")

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side="left", fill="both")

        self.listbox.config(yscrollcommand = scrollbar.set)
        scrollbar.config(command = self.listbox.yview)

        self.frame_photo = tk.Frame(self)
        self.frame_photo.pack(side='left', padx=5, pady=5)

        self.label_widget = tk.Label(self.frame_photo)
        self.label_widget.pack(expand=True, side='top', pady=5)
        self.listbox.bind("<<ListboxSelect>>", self.callback)

        self.delete_button = tk.Button(self.frame_photo, text="Delete", width = 15, font=(1), command=self.delete_face)

    def clear_all(self):
        global database

        database = {}
        for name in os.listdir('stored_faces'):
            os.remove(f'stored_faces/{name}')
        self.destroy()

    def delete_face(self):
        selection = self.listbox.curselection()
        if selection:
            index = selection[0]
            name = self.listbox.get(index)
            os.remove(f'stored_faces/{name}.jpg')
            database.pop(name)

            self.user_list.set(list(database.keys()))
            self.frame_photo.pack_forget()


    def callback(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            name = event.widget.get(index)

            path  = database[name]['image_path']
            pillow_image = Image.open(path)

            pillow_image = pillow_image.resize((200, 200))
            image = ImageTk.PhotoImage(image=pillow_image)
            self.label_widget.photo_image = image
            self.label_widget.configure(image=image)

            self.delete_button.pack(side = 'top', pady=5)
            

if __name__ == "__main__":
    f = open('database.json')
    database = json.load(f)

    # model = InceptionResnetV1()
    # model.load_model('utils/20180402-114759-vggface2.pt')
    model = torch.load('utils/fine_tuned_model.pt')
    model.eval()

    harcascadePath = "utils/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    
    threshold = 0.75
    app = App()
    app.mainloop()