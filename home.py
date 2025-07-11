import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk
import threading
import shutil
from facerec import *
from register import *
from handler import *
from object_detection import *
from object_handler import *
import time
import csv
import numpy as np
import ntpath
import os

active_page = 0
thread_event = None
left_frame = None
right_frame = None
heading = None
webcam = None
img_label = None
img_read = None
img_list = []
slide_caption = None
slide_control_panel = None
current_slide = -1

root = tk.Tk()
root.geometry("1000x900+200+100")

# create Pages
pages = []
for i in range(8):  # Increased to accommodate new pages
    pages.append(tk.Frame(root, bg="#3E3B3C"))
    pages[i].pack(side="top", fill="both", expand=True)
    pages[i].place(x=0, y=0, relwidth=1, relheight=1)


def goBack():
    global active_page, thread_event, webcam

    if (active_page in [4, 7] and thread_event and not thread_event.is_set()):
        thread_event.set()
        if webcam:
            webcam.release()

    for widget in pages[active_page].winfo_children():
        widget.destroy()

    pages[0].lift()
    active_page = 0


def basicPageSetup(pageNo):
    global left_frame, right_frame, heading

    back_img = tk.PhotoImage(file="back.png")
    back_button = tk.Button(pages[pageNo], image=back_img, bg="#3E3B3C", bd=0, highlightthickness=0,
           activebackground="#3E3B3C", command=goBack)
    back_button.image = back_img
    back_button.place(x=10, y=10)

    heading = tk.Label(pages[pageNo], fg="white", bg="#3E3B3C", font="Arial 20 bold", pady=10)
    heading.pack()

    content = tk.Frame(pages[pageNo], bg="#3E3B3C", pady=20)
    content.pack(expand="true", fill="both")

    left_frame = tk.Frame(content, bg="#3E3B3C")
    left_frame.grid(row=0, column=0, sticky="nsew")

    right_frame = tk.LabelFrame(content, text="Results", bg="#3E3B3C", font="Arial 20 bold", bd=4,
                             foreground="#2ea3ef", labelanchor="n")
    right_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

    content.grid_columnconfigure(0, weight=1, uniform="group1")
    content.grid_columnconfigure(1, weight=1, uniform="group1")
    content.grid_rowconfigure(0, weight=1)


def showImage(frame, img_size):
    global img_label, left_frame

    img = cv2.resize(frame, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    if (img_label == None):
        img_label = tk.Label(left_frame, image=img, bg="#202d42")
        img_label.image = img
        img_label.pack(padx=20)
    else:
        img_label.configure(image=img)
        img_label.image = img


def getNewSlide(control):
    global img_list, current_slide

    if(len(img_list) > 1):
        if(control == "prev"):
            current_slide = (current_slide-1) % len(img_list)
        else:
            current_slide = (current_slide+1) % len(img_list)

        img_size = left_frame.winfo_height() - 200
        showImage(img_list[current_slide], img_size)

        slide_caption.configure(text = "Image {} of {}".format(current_slide+1, len(img_list)))


def selectMultiImage(opt_menu, menu_var):
    global img_list, current_slide, slide_caption, slide_control_panel

    filetype = [("images", "*.jpg *.jpeg *.png")]
    path_list = filedialog.askopenfilenames(title="Choose atleast 3 images", filetypes=filetype)

    if(len(path_list) < 3):
        messagebox.showerror("Error", "Choose atleast 3 images.")
    else:
        img_list = []
        current_slide = -1

        # Resetting slide control panel
        if (slide_control_panel != None):
            slide_control_panel.destroy()

        # Creating Image list
        for path in path_list:
            img_list.append(cv2.imread(path))

        # Creating choices for profile pic menu
        menu_var.set("")
        opt_menu['menu'].delete(0, 'end')

        for i in range(len(img_list)):
            ch = "Image " + str(i+1)
            opt_menu['menu'].add_command(label=ch, command= tk._setit(menu_var, ch))
            menu_var.set("Image 1")


        # Creating slideshow of images
        img_size =  left_frame.winfo_height() - 200
        current_slide += 1
        showImage(img_list[current_slide], img_size)

        slide_control_panel = tk.Frame(left_frame, bg="#202d42", pady=20)
        slide_control_panel.pack()

        back_img = tk.PhotoImage(file="previous.png")
        next_img = tk.PhotoImage(file="next.png")

        prev_slide = tk.Button(slide_control_panel, image=back_img, bg="#202d42", bd=0, highlightthickness=0,
                            activebackground="#202d42", command=lambda : getNewSlide("prev"))
        prev_slide.image = back_img
        prev_slide.grid(row=0, column=0, padx=60)

        slide_caption = tk.Label(slide_control_panel, text="Image 1 of {}".format(len(img_list)), fg="#ff9800",
                              bg="#202d42", font="Arial 15 bold")
        slide_caption.grid(row=0, column=1)

        next_slide = tk.Button(slide_control_panel, image=next_img, bg="#202d42", bd=0, highlightthickness=0,
                            activebackground="#202d42", command=lambda : getNewSlide("next"))
        next_slide.image = next_img
        next_slide.grid(row=0, column=2, padx=60)


def register(entries, required, menu_var):
    global img_list

    # Checking if no image selected
    if(len(img_list) == 0):
        messagebox.showerror("Error", "Select Images first.")
        return

    # Fetching data from entries
    entry_data = {}
    for i, entry in enumerate(entries):
        val = entry[1].get()

        if (len(val) == 0 and required[i] == 1):
            messagebox.showerror("Field Error", "Required field missing :\n\n%s" % (entry[0]))
            return
        else:
            entry_data[entry[0]] = val.lower()


    # Setting Directory
    path = os.path.join('face_samples', "temp_criminal")
    if not os.path.isdir(path):
        os.mkdir(path)

    no_face = []
    for i, img in enumerate(img_list):
        # Storing Images in directory
        id = registerCriminal(img, path, i + 1)
        if(id != None):
            no_face.append(id)

    # check if any image doesn't contain face
    if(len(no_face) > 0):
        no_face_st = ""
        for i in no_face:
            no_face_st += "Image " + str(i) + ", "
        messagebox.showerror("Registration Error", "Registration failed!\n\nFollowing images doesn't contain"
                        " face or Face is too small:\n\n%s"%(no_face_st))
        shutil.rmtree(path, ignore_errors=True)
    else:
        # Storing data in database
        insertData(entry_data)
        rowId=1
        if(rowId >= 0):
            messagebox.showinfo("Success", "Criminal Registered Successfully.")
            shutil.move(path, os.path.join('face_samples', entry_data["Name"]))

            # save profile pic
            profile_img_num = int(menu_var.get().split(' ')[1]) - 1
            if not os.path.isdir("profile_pics"):
                os.mkdir("profile_pics")
            cv2.imwrite("profile_pics/criminal %d.png"%rowId, img_list[profile_img_num])

            goBack()
        else:
            shutil.rmtree(path, ignore_errors=True)
            messagebox.showerror("Database Error", "Some error occured while storing data.")


## update scrollregion when all widgets are in canvas
def on_configure(event, canvas, win):
    canvas.configure(scrollregion=canvas.bbox('all'))
    canvas.itemconfig(win, width=event.width)

## Register Page ##
def getPage1():
    global active_page, left_frame, right_frame, heading, img_label
    active_page = 1
    img_label = None
    opt_menu = None
    menu_var = tk.StringVar(root)
    pages[1].lift()

    basicPageSetup(1)
    heading.configure(text="Register Criminal", bg="#3E3B3C")
    right_frame.configure(text="Enter Details", fg="white", bg="#3E3B3C")

    btn_grid = tk.Frame(left_frame, bg="#3E3B3C")
    btn_grid.pack()

    tk.Button(btn_grid, text="Select Images", command=lambda: selectMultiImage(opt_menu, menu_var), font="Arial 15 bold", bg="#000000",
           fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
           activeforeground="white").grid(row=0, column=0, padx=25, pady=25)


    # Creating Scrollable Frame
    canvas = tk.Canvas(right_frame, bg="#202d42", highlightthickness=0)
    canvas.pack(side="left", fill="both", expand="true", padx=30)
    scrollbar = tk.Scrollbar(right_frame, command=canvas.yview, width=20, troughcolor="#3E3B3C", bd=0,
                          activebackground="#3E3B3C", bg="#000000", relief="raised")
    scrollbar.pack(side="left", fill="y")

    scroll_frame = tk.Frame(canvas, bg="#3E3B3C", pady=20)
    scroll_win = canvas.create_window((0, 0), window=scroll_frame, anchor='nw')

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda event, canvas=canvas, win=scroll_win: on_configure(event, canvas, win))


    tk.Label(scroll_frame, text="* Required Fields", bg="#3E3B3C", fg="yellow", font="Arial 13 bold").pack()
    # Adding Input Fields
    input_fields = ("Name", "Father's Name", "Gender", "DOB(yyyy-mm-dd)", "Crimes Done", "Profile Image")
    ip_len = len(input_fields)
    required = [1, 1, 1, 1, 1, 1]

    entries = []
    for i, field in enumerate(input_fields):
        row = tk.Frame(scroll_frame, bg="#3E3B3C")
        row.pack(side="top", fill="x", pady=15)

        label = tk.Text(row, width=20, height=1, bg="#3E3B3C", fg="#ffffff", font="Arial 13", highlightthickness=0, bd=0)
        label.insert("insert", field)
        label.pack(side="left")

        if(required[i] == 1):
            label.tag_configure("star", foreground="yellow", font="Arial 13 bold")
            label.insert("end", "  *", "star")
        label.configure(state="disabled")

        if(i != ip_len-1):
            ent = tk.Entry(row, font="Arial 13", selectbackground="#90ceff")
            ent.pack(side="right", expand="true", fill="x", padx=10)
            entries.append((field, ent))
        else:
            menu_var.set("Image 1")
            choices = ["Image 1"]
            opt_menu = tk.OptionMenu(row, menu_var, *choices)
            opt_menu.pack(side="right", fill="x", expand="true", padx=10)
            opt_menu.configure(font="Arial 13", bg="#000000", fg="white", bd=0, highlightthickness=0, activebackground="#3E3B3C")
            menu = opt_menu.nametowidget(opt_menu.menuname)
            menu.configure(font="Arial 13", bg="white", activebackground="#90ceff", bd=0)

    tk.Button(scroll_frame, text="Register", command=lambda: register(entries, required, menu_var), font="Arial 15 bold",
           bg="#000000", fg="white", pady=10, padx=30, bd=0, highlightthickness=0, activebackground="#3E3B3C",
           activeforeground="white").pack(pady=25)


def showCriminalProfile(name):
    top = tk.Toplevel(bg="#202d42")
    top.title("Criminal Profile")
    top.geometry("1500x900+%d+%d"%(root.winfo_x()+10, root.winfo_y()+10))

    tk.Label(top, text="Criminal Profile", fg="white", bg="#202d42", font="Arial 20 bold", pady=10).pack()

    content = tk.Frame(top, bg="#202d42", pady=20)
    content.pack(expand="true", fill="both")
    content.grid_columnconfigure(0, weight=3, uniform="group1")
    content.grid_columnconfigure(1, weight=5, uniform="group1")
    content.grid_rowconfigure(0, weight=1)

    # Mock data since retrieveData function is not working
    crim_data = {
        "Name": name,
        "Father's Name": "Unknown",
        "Gender": "Unknown",
        "DOB": "Unknown",
        "Crimes Done": "Unknown"
    }

    # Create a placeholder image if profile pic doesn't exist
    profile_img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.putText(profile_img, "No Image", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    img = cv2.cvtColor(profile_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    img_label = tk.Label(content, image=img, bg="#202d42")
    img_label.image = img
    img_label.grid(row=0, column=0)

    info_frame = tk.Frame(content, bg="#202d42")
    info_frame.grid(row=0, column=1, sticky='w')

    for i, item in enumerate(crim_data.items()):
        tk.Label(info_frame, text=item[0], pady=15, fg="yellow", font="Arial 15 bold", bg="#202d42").grid(row=i, column=0, sticky='w')
        tk.Label(info_frame, text=":", fg="yellow", padx=50, font="Arial 15 bold", bg="#202d42").grid(row=i, column=1)
        val = "---" if (item[1]=="") else item[1]
        tk.Label(info_frame, text=val.capitalize(), fg="white", font="Arial 15", bg="#202d42").grid(row=i, column=2, sticky='w')


def startRecognition():
    global img_read, img_label

    if(img_label == None):
        messagebox.showerror("Error", "No image selected. ")
        return

    crims_found_labels = []
    for wid in right_frame.winfo_children():
        wid.destroy()

    frame = cv2.flip(img_read, 1, 0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coords = detect_faces(gray_frame)

    if (len(face_coords) == 0):
        messagebox.showerror("Error", "Image doesn't contain any face or face is too small.")
    else:
        (model, names) = train_model()
        print('Training Successful. Detecting Faces')
        (frame, recognized) = recognize_face(model, frame, gray_frame, face_coords, names)

        img_size = left_frame.winfo_height() - 40
        frame = cv2.flip(frame, 1, 0)
        showImage(frame, img_size)

        if (len(recognized) == 0):
            messagebox.showerror("Error", "No criminal recognized.")
            return

        for i, crim in enumerate(recognized):
            crims_found_labels.append(tk.Label(right_frame, text=crim[0], bg="orange",
                                            font="Arial 15 bold", pady=20))
            crims_found_labels[i].pack(fill="x", padx=20, pady=10)
            crims_found_labels[i].bind("<Button-1>", lambda e, name=crim[0]:showCriminalProfile(name))


def selectImage():
    global left_frame, img_label, img_read
    for wid in right_frame.winfo_children():
        wid.destroy()

    filetype = [("images", "*.jpg *.jpeg *.png")]
    path = filedialog.askopenfilename(title="Choose a image", filetypes=filetype)

    if(len(path) > 0):
        img_read = cv2.imread(path)

        img_size =  left_frame.winfo_height() - 40
        showImage(img_read, img_size)


## Detection Page ##
def getPage2():
    global active_page, left_frame, right_frame, img_label, heading
    img_label = None
    active_page = 2
    pages[2].lift()

    basicPageSetup(2)
    heading.configure(text="Image Surveillance")
    right_frame.configure(text="Detected Criminals", fg="white")

    btn_grid = tk.Frame(left_frame, bg="#3E3B3C")
    btn_grid.pack()

    tk.Button(btn_grid, text="Select Image", command=selectImage, font="Arial 15 bold", padx=20, bg="#000000",
            fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
            activeforeground="white").grid(row=0, column=0, padx=25, pady=25)
    tk.Button(btn_grid, text="Recognize", command=startRecognition, font="Arial 15 bold", padx=20, bg="#000000",
           fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
           activeforeground="white").grid(row=0, column=1, padx=25, pady=25)


def videoLoop(path, model, names):
    p = path
    q = ntpath.basename(p)
    filenam, file_extension = os.path.splitext(q)
    global thread_event, left_frame, webcam, img_label
    start = time.time()
    webcam = cv2.VideoCapture(p)
    old_recognized = []
    crims_found_labels = []
    img_label = None
    field = ['S.No.', 'Name', 'Time']
    g = filenam + '.csv'
    filename = g
    num = 0
    
    try:
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(field)   
            while not thread_event.is_set():
                while (True):
                    (return_val, frame) = webcam.read()
                    if (return_val == True):
                        break

                frame = cv2.flip(frame, 1, 0)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                face_coords = detect_faces(gray_frame)
                (frame, recognized) = recognize_face(model, frame, gray_frame, face_coords, names)

                recog_names = [item[0] for item in recognized]
                if(recog_names != old_recognized):
                    for wid in right_frame.winfo_children():
                        wid.destroy()
                    del(crims_found_labels[:])

                    for i, crim in enumerate(recognized):
                        num += 1
                        x = time.time() - start
                        crims_found_labels.append(tk.Label(right_frame, text=crim[0], bg="orange",
                                                        font="Arial 15 bold", pady=20))
                        crims_found_labels[i].pack(fill="x", padx=20, pady=10)
                        crims_found_labels[i].bind("<Button-1>", lambda e, name=crim[0]: showCriminalProfile(name))
                        y = crim[0]
                        print(x, y)
                        arr = [num, y, x]
                        csvwriter.writerow(arr)  
                        
                    old_recognized = recog_names

                img_size = min(left_frame.winfo_width(), left_frame.winfo_height()) - 20
                showImage(frame, img_size)

    except RuntimeError:
        print("[INFO]Caught Runtime Error")
    except tk.TclError:
        print("[INFO]Caught Tcl Error")


def getPage4(path):
    p = path
    global active_page, video_loop, left_frame, right_frame, thread_event, heading
    active_page = 4
    pages[4].lift()

    basicPageSetup(4)
    heading.configure(text="Video Surveillance")
    right_frame.configure(text="Detected Criminals")
    left_frame.configure(pady=40)

    btn_grid = tk.Frame(right_frame, bg="#3E3B3C")
    btn_grid.pack()

    (model, names) = train_model()
    print('Training Successful. Detecting Faces')

    thread_event = threading.Event()
    thread = threading.Thread(target=videoLoop, args=(p, model, names))
    thread.start()


def getPage3():
    global active_page, video_loop, left_frame, right_frame, thread_event, heading
    active_page = 3
    pages[3].lift()

    basicPageSetup(3)
    heading.configure(text="Video Surveillance")

    btn_grid = tk.Frame(left_frame, bg="#3E3B3C")
    btn_grid.pack()

    tk.Button(btn_grid, text="Select Video", command=selectvideo, font="Arial 15 bold", padx=20, bg="#000000",
                fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
                activeforeground="white").grid(row=0, column=0, padx=25, pady=25)


def selectvideo():
    global left_frame, img_label, img_read
    for wid in right_frame.winfo_children():
        wid.destroy()

    filetype = [("video", "*.mp4 *.mkv")]
    path = filedialog.askopenfilename(title="Choose a video", filetypes=filetype)
    p = ''
    p = path
    
    if(len(path) > 0):
        getPage4(p)


# Object Detection Functions
def selectObjectImages():
    global img_list, current_slide, slide_caption, slide_control_panel

    filetype = [("images", "*.jpg *.jpeg *.png")]
    path_list = filedialog.askopenfilenames(title="Choose object images (at least 3)", filetypes=filetype)

    if(len(path_list) < 3):
        messagebox.showerror("Error", "Choose at least 3 images of the object.")
    else:
        img_list = []
        current_slide = -1

        if (slide_control_panel != None):
            slide_control_panel.destroy()

        for path in path_list:
            img_list.append(cv2.imread(path))

        img_size = left_frame.winfo_height() - 200
        current_slide += 1
        showImage(img_list[current_slide], img_size)

        slide_control_panel = tk.Frame(left_frame, bg="#202d42", pady=20)
        slide_control_panel.pack()

        back_img = tk.PhotoImage(file="previous.png")
        next_img = tk.PhotoImage(file="next.png")

        prev_slide = tk.Button(slide_control_panel, image=back_img, bg="#202d42", bd=0, highlightthickness=0,
                            activebackground="#202d42", command=lambda : getNewSlide("prev"))
        prev_slide.image = back_img
        prev_slide.grid(row=0, column=0, padx=60)

        slide_caption = tk.Label(slide_control_panel, text="Image 1 of {}".format(len(img_list)), fg="#ff9800",
                              bg="#202d42", font="Arial 15 bold")
        slide_caption.grid(row=0, column=1)

        next_slide = tk.Button(slide_control_panel, image=next_img, bg="#202d42", bd=0, highlightthickness=0,
                            activebackground="#202d42", command=lambda : getNewSlide("next"))
        next_slide.image = next_img
        next_slide.grid(row=0, column=2, padx=60)


def registerObject(entries, required):
    global img_list

    if(len(img_list) == 0):
        messagebox.showerror("Error", "Select object images first.")
        return

    entry_data = {}
    for i, entry in enumerate(entries):
        val = entry[1].get()

        if (len(val) == 0 and required[i] == 1):
            messagebox.showerror("Field Error", "Required field missing :\n\n%s" % (entry[0]))
            return
        else:
            entry_data[entry[0]] = val

    # Save images temporarily
    temp_dir = "temp_object_images"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    image_paths = []
    for i, img in enumerate(img_list):
        img_path = os.path.join(temp_dir, f"object_{i}.jpg")
        cv2.imwrite(img_path, img)
        image_paths.append(img_path)

    # Register object
    success = register_criminal_object(entry_data["Object Name"], image_paths)
    
    if success:
        register_new_object(entry_data)
        messagebox.showinfo("Success", "Criminal object registered successfully.")
        
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)
        goBack()
    else:
        messagebox.showerror("Error", "Failed to register object. Please try again.")
        shutil.rmtree(temp_dir, ignore_errors=True)


def getPage5():  # Register Criminal Object
    global active_page, left_frame, right_frame, heading, img_label
    active_page = 5
    img_label = None
    pages[5].lift()

    basicPageSetup(5)
    heading.configure(text="Register Criminal Object")
    right_frame.configure(text="Enter Object Details", fg="white")

    btn_grid = tk.Frame(left_frame, bg="#3E3B3C")
    btn_grid.pack()

    tk.Button(btn_grid, text="Select Object Images", command=selectObjectImages, font="Arial 15 bold", bg="#000000",
           fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
           activeforeground="white").grid(row=0, column=0, padx=25, pady=25)

    # Creating Scrollable Frame
    canvas = tk.Canvas(right_frame, bg="#202d42", highlightthickness=0)
    canvas.pack(side="left", fill="both", expand="true", padx=30)
    scrollbar = tk.Scrollbar(right_frame, command=canvas.yview, width=20, troughcolor="#3E3B3C", bd=0,
                          activebackground="#3E3B3C", bg="#000000", relief="raised")
    scrollbar.pack(side="left", fill="y")

    scroll_frame = tk.Frame(canvas, bg="#3E3B3C", pady=20)
    scroll_win = canvas.create_window((0, 0), window=scroll_frame, anchor='nw')

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda event, canvas=canvas, win=scroll_win: on_configure(event, canvas, win))

    tk.Label(scroll_frame, text="* Required Fields", bg="#3E3B3C", fg="yellow", font="Arial 13 bold").pack()
    
    input_fields = ("Object Name", "Description", "Danger Level")
    required = [1, 1, 1]

    entries = []
    for i, field in enumerate(input_fields):
        row = tk.Frame(scroll_frame, bg="#3E3B3C")
        row.pack(side="top", fill="x", pady=15)

        label = tk.Text(row, width=20, height=1, bg="#3E3B3C", fg="#ffffff", font="Arial 13", highlightthickness=0, bd=0)
        label.insert("insert", field)
        label.pack(side="left")

        if(required[i] == 1):
            label.tag_configure("star", foreground="yellow", font="Arial 13 bold")
            label.insert("end", "  *", "star")
        label.configure(state="disabled")

        if field == "Danger Level":
            danger_var = tk.StringVar(root)
            danger_var.set("Low")
            danger_menu = tk.OptionMenu(row, danger_var, "Low", "Medium", "High", "Critical")
            danger_menu.pack(side="right", fill="x", expand="true", padx=10)
            danger_menu.configure(font="Arial 13", bg="#000000", fg="white", bd=0, highlightthickness=0)
            entries.append((field, danger_var))
        else:
            ent = tk.Entry(row, font="Arial 13", selectbackground="#90ceff")
            ent.pack(side="right", expand="true", fill="x", padx=10)
            entries.append((field, ent))

    tk.Button(scroll_frame, text="Register Object", command=lambda: registerObject(entries, required), font="Arial 15 bold",
           bg="#000000", fg="white", pady=10, padx=30, bd=0, highlightthickness=0, activebackground="#3E3B3C",
           activeforeground="white").pack(pady=25)


def startObjectDetection():
    global img_read, img_label

    if(img_label == None):
        messagebox.showerror("Error", "No image selected.")
        return

    for wid in right_frame.winfo_children():
        wid.destroy()

    detected_objects = detect_objects_in_frame(img_read)

    if len(detected_objects) == 0:
        tk.Label(right_frame, text="No criminal objects detected", bg="#3E3B3C", fg="white",
                font="Arial 15 bold", pady=20).pack(fill="x", padx=20, pady=10)
    else:
        for obj_name, confidence in detected_objects:
            obj_label = tk.Label(right_frame, text=f"{obj_name}\n(Confidence: {confidence})", bg="red",
                               font="Arial 15 bold", pady=20)
            obj_label.pack(fill="x", padx=20, pady=10)
            
            # Save detection data
            save_object_detection_data(obj_name, time.time(), "Image Detection")


def getPage6():  # Object Detection
    global active_page, left_frame, right_frame, img_label, heading
    img_label = None
    active_page = 6
    pages[6].lift()

    basicPageSetup(6)
    heading.configure(text="Image Surveillance")
    right_frame.configure(text="Detected Objects", fg="white")

    btn_grid = tk.Frame(left_frame, bg="#3E3B3C")
    btn_grid.pack()

    tk.Button(btn_grid, text="Select Image", command=selectImage, font="Arial 15 bold", padx=20, bg="#000000",
            fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
            activeforeground="white").grid(row=0, column=0, padx=25, pady=25)
    tk.Button(btn_grid, text="Detect Objects", command=startObjectDetection, font="Arial 15 bold", padx=20, bg="#000000",
           fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
           activeforeground="white").grid(row=0, column=1, padx=25, pady=25)


def objectVideoLoop(path):
    p = path
    q = ntpath.basename(p)
    filenam, file_extension = os.path.splitext(q)
    global thread_event, left_frame, webcam, img_label
    start = time.time()
    webcam = cv2.VideoCapture(p)
    old_detected = []
    obj_found_labels = []
    img_label = None
    field = ['S.No.', 'Object Name', 'Time']
    g = filenam + '_objects.csv'
    filename = g
    num = 0
    
    try:
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(field)   
            while not thread_event.is_set():
                while (True):
                    (return_val, frame) = webcam.read()
                    if (return_val == True):
                        break

                frame = cv2.flip(frame, 1, 0)
                detected_objects = detect_objects_in_frame(frame)

                detected_names = [obj[0] for obj in detected_objects]
                if(detected_names != old_detected):
                    for wid in right_frame.winfo_children():
                        wid.destroy()
                    del(obj_found_labels[:])

                    for i, (obj_name, confidence) in enumerate(detected_objects):
                        num += 1
                        x = time.time() - start
                        obj_found_labels.append(tk.Label(right_frame, text=f"{obj_name}\n(Conf: {confidence})", bg="red",
                                                        font="Arial 15 bold", pady=20))
                        obj_found_labels[i].pack(fill="x", padx=20, pady=10)
                        print(x, obj_name)
                        arr = [num, obj_name, x]
                        csvwriter.writerow(arr)  
                        
                    old_detected = detected_names

                img_size = min(left_frame.winfo_width(), left_frame.winfo_height()) - 20
                showImage(frame, img_size)

    except RuntimeError:
        print("[INFO]Caught Runtime Error")
    except tk.TclError:
        print("[INFO]Caught Tcl Error")


def getPage7(path):  # Object Video Surveillance
    p = path
    global active_page, video_loop, left_frame, right_frame, thread_event, heading
    active_page = 7
    pages[7].lift()

    basicPageSetup(7)
    heading.configure(text="Object Video Surveillance")
    right_frame.configure(text="Detected Objects")
    left_frame.configure(pady=40)

    thread_event = threading.Event()
    thread = threading.Thread(target=objectVideoLoop, args=(p,))
    thread.start()


def selectObjectVideo():
    filetype = [("video", "*.mp4 *.mkv")]
    path = filedialog.askopenfilename(title="Choose a video", filetypes=filetype)
    
    if(len(path) > 0):
        getPage7(path)


def getObjectVideoPage():
    global active_page, left_frame, right_frame, heading
    active_page = 8  # Temporary page for video selection
    pages[3].lift()  # Reuse page 3 layout

    basicPageSetup(3)
    heading.configure(text="Object Video Surveillance")

    btn_grid = tk.Frame(left_frame, bg="#3E3B3C")
    btn_grid.pack()

    tk.Button(btn_grid, text="Select Video", command=selectObjectVideo, font="Arial 15 bold", padx=20, bg="#000000",
                fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
                activeforeground="white").grid(row=0, column=0, padx=25, pady=25)


######################################## Home Page ####################################
tk.Label(pages[0], text="Face Recognition System for Crime Detection", fg="black", bg="#3E3B3C",
      font="Arial 25 bold", pady=30).pack()

logo = tk.PhotoImage(file = "logo2.png")
tk.Label(pages[0], image=logo, bg="#3E3B3C").pack(side='left')

btn_frame = tk.Frame(pages[0], bg="#3E3B3C", pady=30)
btn_frame.pack()

# Criminal Detection Buttons
tk.Label(btn_frame, text="Criminal Detection", fg="white", bg="#3E3B3C", font="Arial 18 bold").pack(pady=10)
tk.Button(btn_frame, text="Add Criminal Details", command=getPage1)
tk.Button(btn_frame, text="Image Surveillance", command=getPage2)
tk.Button(btn_frame, text="Video Surveillance", command=getPage3)

# Separator
tk.Label(btn_frame, text="", bg="#3E3B3C").pack(pady=10)

# Object Detection Buttons
tk.Label(btn_frame, text="Criminal Object Detection", fg="white", bg="#3E3B3C", font="Arial 18 bold").pack(pady=10)
tk.Button(btn_frame, text="Add Criminal Objects", command=getPage5)
tk.Button(btn_frame, text="Object Image Detection", command=getPage6)
tk.Button(btn_frame, text="Object Video Surveillance", command=getObjectVideoPage)

for btn in btn_frame.winfo_children():
    if isinstance(btn, tk.Button):
        btn.configure(font="Arial 16", width=20, bg="#000000", fg="white",
            pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C", activeforeground="white")
        btn.pack(pady=15)

pages[0].lift()
root.mainloop()