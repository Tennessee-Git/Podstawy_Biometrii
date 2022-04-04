from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import pathlib
import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

from segmentation import segmentation

PROJECT_PATH = pathlib.Path(__file__).parent
PROJECT_UI = PROJECT_PATH / "biometria_gui.ui"

from filters import filter_image


class BiometriaGuiApp:
    def __init__(self, master=None):
        # build ui
        self.original_image_path = None
        self.grey_changed_image_1 = None
        self.grey_original_image = None
        self.changed_image_1 = None
        self.original_image = None
        self.main_window = tk.Tk() if master is None else tk.Toplevel(master)
        self.menu_frame = tk.Frame(self.main_window)
        self.select_image_button = tk.Button(self.menu_frame)
        self.select_image_button.configure(cursor='hand2', text='Select Image')
        self.select_image_button.pack(fill='x', ipadx='10', padx='10', pady='10', side='top')
        self.select_image_button.configure(command=self.select_and_insert_image)
        self.frame1 = tk.Frame(self.menu_frame)
        self.binarize_threshold_scale = tk.Scale(self.frame1)
        self.binarize_threshold_scale.configure(from_='0', label='Threshold', orient='horizontal', to='255')
        self.binarize_threshold_scale.pack(fill='x', side='top')
        self.binarize_threshold_scale.set(120)
        self.__tkvar = tk.StringVar(value='Normal')
        __values = ['Red', 'Green', 'Blue', 'Normal']
        self.binarize_options = tk.OptionMenu(self.frame1, self.__tkvar, 'Normal', *__values,
                                              command=self.set_binarize_option)
        self.set_binarize_option(self.__tkvar.get())
        self.binarize_options.pack(anchor='n', fill='x', side='top')
        self.binarize_image_button = tk.Button(self.frame1)
        self.binarize_image_button.configure(text='Binarize Image')
        self.binarize_image_button.pack(fill='x', side='top')
        self.binarize_image_button.configure(command=self.binarize_and_insert_image)
        self.frame1.configure(height='117', width='200')
        self.frame1.pack(padx='10', pady='10', side='top')
        self.frame1.pack_propagate(0)

        self.algorithms_container = tk.Frame(self.menu_frame)
        self.algorithm_window_size_label = tk.Label(self.algorithms_container)
        self.algorithm_window_size_label.configure(justify='left', text='Window size')
        self.algorithm_window_size_label.pack(fill='x', side='top')
        self.algorithm_window_size_input = ttk.Spinbox(self.algorithms_container)
        self.algorithm_window_size_input.configure(from_='1', increment='2', to='100')
        self.algorithm_window_size_input.pack(fill='x', padx='2', side='top')
        self.algorithm_window_size_input.set(5)
        self.algorithm_k_label = tk.Label(self.algorithms_container)
        self.algorithm_k_label.configure(text='k')
        self.algorithm_k_label.pack(side='top')
        self.algorithm_k_input = ttk.Spinbox(self.algorithms_container)
        self.algorithm_k_input.configure(from_='-100', increment='0.1', to='100')
        self.algorithm_k_input.pack(fill='x', padx='2', side='top')
        self.algorithm_k_input.set(0.2)
        __values2 = ['Sauvola Algorithm', 'Phansalkar Algorithm', 'Brensen Algorithm']
        self.__tkvar2 = tk.StringVar(value='Niblack Algorithm')
        self.algorithm_type = tk.OptionMenu(self.algorithms_container, self.__tkvar2, 'Niblack Algorithm', *__values2,
                                            command=self.set_algorithm_type)
        self.algorithm_type.pack(fill='x', side='top')
        self.set_algorithm_type(self.__tkvar2.get())
        self.calculate_algorithm_button = tk.Button(self.algorithms_container)
        self.calculate_algorithm_button.configure(text='Calculate')
        self.calculate_algorithm_button.pack(fill='x', side='top')
        self.calculate_algorithm_button.configure(command=self.calculate_algorithm_threshold)
        self.algorithms_container.configure(height='138', width='200')
        self.algorithms_container.pack(padx='10', pady='10', side='top')
        self.algorithms_container.pack_propagate(0)

        self.filters_frame = tk.Frame(self.menu_frame)
        self.label1 = tk.Label(self.filters_frame)
        self.label1.configure(justify='left', text='Mask size')
        self.label1.pack(fill='x', side='top')
        self.filters_pixel_size_input = tk.Spinbox(self.filters_frame)
        self.filters_pixel_size_input.configure(from_='1', increment='2', to='100')
        self.filters_pixel_size_input.pack(fill='x', padx='2', side='top')
        __values = ['Median Filter', 'Pixelization', 'Kuwahara Filter', 'Gaussian Filter', 'Sobel Filter']
        self.__tkvar = tk.StringVar(value='Median Filter')
        self.filter_type = tk.OptionMenu(self.filters_frame, self.__tkvar, 'Linear Filter', *__values,
                                         command=self.set_filter_type)
        self.filter_type.pack(fill='x', side='top')
        self.set_filter_type(self.__tkvar.get())
        self.calculate_filter_button = tk.Button(self.filters_frame)
        self.calculate_filter_button.configure(text='Calculate')
        self.calculate_filter_button.pack(fill='x', side='top')
        self.calculate_filter_button.configure(command=self.calculate_filter)
        self.filters_frame.configure(height='98', width='200')
        self.filters_frame.pack(padx='10', pady='10', side='top')
        self.filters_frame.pack_propagate(0)

        self.segmentation_frame = tk.Frame(self.menu_frame)
        self.R_color_scale = tk.Scale(self.segmentation_frame)
        self.R_color_scale.configure(from_='0', label='R', orient='horizontal', to='255')
        self.R_color_scale.pack(fill='x', side='top')
        self.G_color_scale = tk.Scale(self.segmentation_frame)
        self.G_color_scale.configure(from_='0', label='G', orient='horizontal', to='255')
        self.G_color_scale.pack(fill='x', side='top')
        self.B_color_scale = tk.Scale(self.segmentation_frame)
        self.B_color_scale.configure(from_='0', label='B', orient='horizontal', to='255')
        self.B_color_scale.pack(fill='x', side='top')
        self.label5 = tk.Label(self.segmentation_frame)
        self.label5.configure(text='Pixel count')
        self.label5.pack(side='top')
        self.segmentation_pixel_count_input = tk.Spinbox(self.segmentation_frame)
        self.segmentation_pixel_count_input.configure(from_='0', increment='1', to='100')
        self.segmentation_pixel_count_input.pack(fill='x', padx='2', side='top')
        self.segmentation_global_mode_val = tk.IntVar()
        self.segmentation_global_mode = tk.Checkbutton(self.segmentation_frame,
                                                       variable=self.segmentation_global_mode_val)
        self.segmentation_global_mode.configure(text='Global mode')
        self.segmentation_global_mode.pack(fill='both', side='left')
        self.segmentation_fill_mode_val = tk.IntVar()
        self.segmentation_fill_mode = tk.Checkbutton(self.segmentation_frame, variable=self.segmentation_fill_mode_val)
        self.segmentation_fill_mode.configure(text='Fill mode')
        self.segmentation_fill_mode.pack(fill='both', side='left')
        self.segmentation_frame.configure(height='250', width='200')
        self.segmentation_frame.pack(padx='10', pady='10', side='top')
        self.segmentation_frame.pack_propagate(0)

        self.menu_frame.configure(background='#F6AE2D', height='800', width='200')
        self.menu_frame.pack(side='left')
        self.menu_frame.pack_propagate(0)
        self.main_frame = tk.Frame(self.main_window)
        self.original_image_canvas = tk.Canvas(self.main_frame)
        self.original_image_canvas = tk.Canvas(self.main_frame)
        self.original_image_canvas.configure(background='#0e5092', cursor='hand2', height='350', width='350')
        self.original_image_canvas.place(anchor='nw', relx='0.13', rely='0.03', x='0', y='0')
        self.original_image_canvas.bind('<Button-1>', self.run_segmentation)
        self.changed_image_1_canvas = tk.Canvas(self.main_frame)
        self.changed_image_1_canvas.configure(background='#F26419', cursor='hand2', height='350', width='350')
        self.changed_image_1_canvas.place(anchor='ne', relx='0.87', rely='0.03', x='0', y='0')
        self.changed_image_1_canvas.bind('<Button-1>',
                                         lambda _: self.open_image_in_new_window(self.changed_image_1_canvas,
                                                                                 self.changed_image_1))
        self.grey_original_image_canvas = tk.Canvas(self.main_frame)
        self.grey_original_image_canvas.configure(background='#65dc80', height='350', width='350')
        self.grey_original_image_canvas.place(anchor='sw', relx='0.13', rely='0.97', x='0', y='0')
        self.grey_changed_image_1_canvas = tk.Canvas(self.main_frame)
        self.grey_changed_image_1_canvas.configure(background='#8986bb', height='350', width='350')
        self.grey_changed_image_1_canvas.place(anchor='se', relx='0.87', rely='0.97', x='0', y='0')
        self.histogram_1 = tk.Button(self.main_frame)
        self.histogram_1.configure(text='Histogram 1')
        self.histogram_1.place(relx='0.035', rely='0.25', x='0', y='0')
        self.histogram_1.configure(command=lambda: self.generate_and_display_histogram(self.original_image))
        self.histogram_2 = tk.Button(self.main_frame)
        self.histogram_2.configure(text='Histogram 2')
        self.histogram_2.place(anchor='se', relx='0.965', rely='0.25', x='0', y='0')
        self.histogram_2.configure(
            command=lambda: self.generate_and_display_histogram(self.changed_image_1, 'changed_image_1'))
        self.histogram_3 = tk.Button(self.main_frame)
        self.histogram_3.configure(text='Histogram 3')
        self.histogram_3.place(anchor='nw', relx='0.035', rely='0.75', x='0', y='0')
        self.histogram_3.configure(command=self.generate_and_display_histogram)
        self.histogram_3.configure(
            command=lambda: self.generate_and_display_histogram(self.grey_original_image, 'grey_original_image'))
        self.histogram_4 = tk.Button(self.main_frame)
        self.histogram_4.configure(text='Histogram 4')
        self.histogram_4.place(anchor='se', relx='0.965', rely='0.75', x='0', y='0')
        self.histogram_4.configure(
            command=lambda: self.generate_and_display_histogram(self.grey_changed_image_1, 'grey_changed_image_1'))
        self.main_frame.configure(background='#03256C', height='800', width='1166')
        self.main_frame.pack(side='top')
        self.main_frame.pack_propagate(0)
        self.main_window.configure(height='200', width='200')
        self.main_window.geometry('1366x768')
        self.main_window.resizable(False, False)
        self.main_window.title('Biometric Basics')

        # Main widget
        self.mainwindow = self.main_window

    def run(self):
        self.mainwindow.mainloop()

    # https://imagej.net/plugins/auto-local-threshold
    def algorithm_threshold(self, img, w, k, type_algorithm):
        img = img.astype(float)
        img2 = np.square(img)

        ave = cv2.blur(img, (w, w))
        ave2 = cv2.blur(img2, (w, w))

        n = np.multiply(*img.shape)
        std = np.sqrt((ave2 * n - img2) / n / (n - 1))

        if type_algorithm == 'Niblack Algorithm':
            # T = m(x, y) - k * s(x, y)
            t = ave - k * std
        elif type_algorithm == 'Sauvola Algorithm':
            # T = m(x,y) * (1 + k * ((s(x,y) / R) - 1))
            t = ave * (1 + k * ((std / 255) - 1))
        elif type_algorithm == 'Phansalkar Algorithm':
            p = 2
            q = 10
            t = ave * (1 + p * np.exp(-q * ave) + k * ((std / 255) - 1))
        elif type_algorithm == 'Brensen Algorithm':
            pass

        binary = np.zeros(img.shape)
        binary[img >= t] = 255
        return binary

    def calculate_algorithm_threshold(self):
        if not self.original_image:
            self.message_popup('No image selected', 'Please select an image first', 'info')
            return
        window_size = int(self.algorithm_window_size_input.get())
        k = float(self.algorithm_k_input.get())
        type_algorithm = self.algorithm_type
        original = cv2.imread(self.original_image_path)
        scaled = cv2.resize(original, None, fx=1, fy=1)
        scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        inverted = 255 - scaled
        binary_inv = self.algorithm_threshold(inverted, window_size, k, type_algorithm)
        binary = 255 - binary_inv
        cv2.imshow(type_algorithm, binary)

    def set_algorithm_type(self, algorithm_type):
        self.algorithm_type = algorithm_type

    def gray_scale_image(self, image):
        image = image.convert('L')
        return image

    def select_and_insert_image(self):
        filename = filedialog.askopenfilename(title='Select an image')
        if not filename:
            self.message_popup('No image selected', 'No image selected', 'warning')
            return
        img = Image.open(filename)
        self.original_image = img
        self.original_image_path = filename
        self.insert_image(img, self.original_image_canvas)
        grey_image = self.gray_scale_image(img)
        self.grey_original_image = grey_image.copy()
        self.insert_image(grey_image, self.grey_original_image_canvas)

    def change_image_size(self, img, width=350, height=350):
        return img.resize((width, height), Image.ANTIALIAS)

    def open_image_in_new_window(self, event, img):
        if not img:
            self.message_popup('Image', 'You need to select an image first', 'info')
        else:
            img.show()

    def message_popup(self, title, text, type_message='info'):
        if type_message == 'info':
            tk.messagebox.showinfo(title, text)
        elif type_message == 'warning':
            tk.messagebox.showwarning(title, text)
        elif type_message == 'error':
            tk.messagebox.showerror(title, text)

    def binarize_grey_image(self, image):
        threshold = self.binarize_threshold_scale.get()
        binarize_option = self.binarize_option
        for x in range(image.width):
            for y in range(image.height):
                pixel = image.getpixel((x, y))
                if pixel < threshold:
                    image.putpixel((x, y), 0)
                else:
                    image.putpixel((x, y), 255)
        image.save('binarized_grey_image.png')
        self.grey_changed_image_1 = image
        return image

    def binarize_image(self, image):
        if image.mode == 'L':
            return self.binarize_grey_image(image)
        threshold = self.binarize_threshold_scale.get()
        binarize_option = self.binarize_option
        for x in range(image.width):
            for y in range(image.height):
                pixel = image.getpixel((x, y))
                if binarize_option == 'Normal':
                    pixel = int((pixel[0] + pixel[1] + pixel[2]) / 3)
                elif binarize_option == 'Red':
                    pixel = pixel[0]
                elif binarize_option == 'Green':
                    pixel = pixel[1]
                elif binarize_option == 'Blue':
                    pixel = pixel[2]
                if pixel < threshold:
                    image.putpixel((x, y), (0, 0, 0))
                else:
                    image.putpixel((x, y), (255, 255, 255))
        self.changed_image_1 = image
        image.save('binarized_image.png')
        return image

    def set_binarize_option(self, option):
        self.binarize_option = option

    def binarize_and_insert_image(self):
        if not self.original_image:
            self.message_popup('Original Image', 'You need to select an image first', 'info')
        else:
            original_image = self.original_image.copy()
            binarized_image = self.binarize_image(original_image)
            self.insert_image(binarized_image, self.changed_image_1_canvas)
            grey_image = self.gray_scale_image(original_image)
            binarized_image = self.binarize_image(grey_image)
            self.insert_image(binarized_image, self.grey_changed_image_1_canvas)

    def insert_image(self, img, canvas):
        img = self.change_image_size(img)
        img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, image=img, anchor='nw')
        canvas.image = img
        canvas.configure(height=img.height(), width=img.width())

    def generate_histogram(self, image):
        histogram = np.zeros(256)
        for x in range(image.width):
            for y in range(image.height):
                pixel = image.getpixel((x, y))
                if type(pixel) == tuple:
                    pixel = int((pixel[0] + pixel[1] + pixel[2]) / 3)
                histogram[pixel] += 1
        return histogram

    def generate_and_display_histogram(self, image, type_histogram='original'):
        if not image:
            self.message_popup('Image', 'You need to select an image first', 'info')
            return
        histogram = self.generate_histogram(image)
        plt.figure()
        plt.bar(range(256), histogram)
        plt.title('Histogram of ' + type_histogram + ' image')
        plt.xlabel('Pixel value')
        plt.ylabel('Number of pixels')
        plt.grid(True, which='major', axis='y')
        plt.ylim(0, max(histogram) + 100)
        plt.show()

    def set_filter_type(self, filter_type):
        self.filter_type = filter_type

    def calculate_filter(self):
        if not self.original_image:
            self.message_popup('No image selected', 'Please select an image first', 'info')
            return
        pixel_size = self.filters_pixel_size_input.get()
        filter_type = self.filter_type
        pixelate_image_result = filter_image(self.original_image_path, pixel_size, filter_type)
        if not filter_type == 'Sobel Filter':
            if filter_type == 'Gaussian Filter':
                pixelate_image_result = pixelate_image_result.astype(np.uint8)
            cv2.imshow(filter_type, pixelate_image_result)

    def run_segmentation(self, event):
        R_value = int(self.R_color_scale.get())
        G_value = int(self.G_color_scale.get())
        B_value = int(self.B_color_scale.get())
        pixel_count = int(self.segmentation_pixel_count_input.get())
        global_mode = int(self.segmentation_global_mode_val.get())
        fill_mode = int(self.segmentation_fill_mode_val.get())

        file_name = "segmentation"
        if global_mode == '1':
            file_name = "global segmentation"
        elif fill_mode == '1':
            file_name = "fill segmentation"
        else:
            file_name = "global segmentation"

        segmented = segmentation(self.original_image_path, R_value, G_value, B_value, pixel_count, global_mode, fill_mode, event.x, event.y)
        cv2.imshow(file_name, segmented)


if __name__ == '__main__':
    app = BiometriaGuiApp()
    app.run()
