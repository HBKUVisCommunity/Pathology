# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2020 Koninklijke Philips N.V. All Rights Reserved.

# A copyright license is hereby granted for redistribution and use of the
# Software in source and binary forms, with or without modification, provided
# that the following conditions are met:
# • Redistributions of source code must retain the above copyright notice, this
#   copyright license and the following disclaimer.
# • Redistributions in binary form must reproduce the above copyright notice,
#   this copyright license and the following disclaimer in the documentation
#   and/ or other materials provided with the distribution.
# • Neither the name of Koninklijke Philips N.V. nor the names of its
#   subsidiaries may be used to endorse or promote products derived from the
#   Software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Tkinter Display
"""
try:
    from tkinter import Label
    import numpy as np
    from PIL import Image, ImageTk
    import matplotlib.pyplot as plt
    from skimage import measure

    from six.moves import tkinter as tk
    from constants import Constants
    from tile_processor import TileProcessor
    import os
    from skimage import measure
    from matplotlib.figure import Figure 
    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
    NavigationToolbar2Tk) 
    
except ImportError as exp:
    print(exp)


class TkinterDisplay:
    """
    This class is responsible for display of tiles.
    """
    myrect=None
    display_list1=[]

    def __init__(self, input_file, args_level, args_backend, args_display_view):
        """
        Constructor initializes the tkinter display with the IsyntaxFilePath.
        :param input_file: IsyntaxFilePath
        :param args_level: The requested level
        :param args_backend: The requested backend option
        """
        self._mouse_wheel_down, self._mouse_wheel_up = Constants.get_mouse_wheel()
        self._window = tk.Tk()
        if args_display_view:
            self._title = "Python iSyntax Viewer                File: {}                View: " \
                          "Display ".format(input_file)
        else:
            self._title = "Python iSyntax Viewer                File: {}                View: " \
                          "Source ".format(input_file)
        self._window.title(self._title)
        # Configuring self.windows geometry as primary monitors height and width
        self._window.geometry("{}x{}+200+100".format(str(self._window.winfo_screenwidth()),
                                                     str(self._window.winfo_screenheight())))
        self._canvas = tk.Canvas(self._window, background='black')
        
        self._canvas.pack(side='left', expand='yes', fill='both')

        self.menu = tk.Menu(self._window)
        self.menu.add_command(label="Thumbnail", command=self.show_macro_label_image)
        self.menu.add_command(label="Properties", command=self.show_properties)
        self._process_tiles = TileProcessor(input_file, args_level, args_backend, args_display_view)
        self.display_canvas()
        self.register_mouse_events()
        self._window.config(menu=self.menu)
        self._window.mainloop()  # Event processing loop
        self.macro_image_data = None
        self.label_image_data = None

        
        ######
        
        self.start=(0,0)
        self.end=(0,0)

    def click_event(self, event): 
        #print(self.myrect)
        canvas_x, canvas_y = self.get_canvas_coordinate(event.x, event.y)

        self.start=(canvas_x,canvas_y)
    def generateimage(self,imageCoord,image_tiles):
        print("len(image_tiles): ",len(image_tiles))

        xpositions=[int(self.start[0]-imageCoord[0]),int(self.end[0]-imageCoord[0])]
        ypositions=[int(self.start[1]-imageCoord[1]),int(self.end[1]-imageCoord[1])]
        #image= np.empty([xpositions[1]-xpositions[0],ypositions[1]-ypositions[0],3],dtype=np.uint8)
        size=[int(imageCoord[3]-imageCoord[1]+Constants.tile_height),int(imageCoord[2]-imageCoord[0]+Constants.tile_width),3]
        image= np.empty(size,dtype=np.uint8)
        
        
        for tile in image_tiles:
                tile[0]= tile[0]-imageCoord[0]
                tile[1]= tile[1]-imageCoord[1]

                for i in range(tile[0],tile[0]+Constants.tile_width):
                    for j in range(tile[1],tile[1]+Constants.tile_height):
                        pix=tile[2]._PhotoImage__photo.get(i-tile[0],j-tile[1])
                        
                        image[j,i,0]= pix[0]
                        image[j,i,1]= pix[1]
                        image[j,i,2]= pix[2]
                        
        print("xpositions,ypositions : ",xpositions,ypositions)
        print("image.shape:", image.shape)
        
        return image[ypositions[0]:ypositions[1],xpositions[0]:xpositions[1],:]                

    
    
    
        
    def release_event(self, event):  
        
        if self.myrect is not None:
            self._canvas.delete(self.myrect) 
        canvas_x, canvas_y = self.get_canvas_coordinate(event.x, event.y)
        self.end=(canvas_x,canvas_y)
        print("self.start:",self.start)
        print('self.end:',self.end)
        
        self.display_canvas(int((self.start[0]+self.end[0])/2),int((self.start[1]+self.end[1])/2),True)



        xStartTile=int(self.start[0]/Constants.tile_width)*Constants.tile_width
        xEndTile=int(self.end[0]/Constants.tile_width)*Constants.tile_width
        yStartTile=int(self.start[1]/Constants.tile_height)*Constants.tile_height
        yEndTile=int(self.end[1]/Constants.tile_height)*Constants.tile_height
        
        minn=min(xStartTile,xEndTile)
        maxx=max(xStartTile,xEndTile)
        xStartTile=minn
        xEndTile=maxx
        
        minn=min(yStartTile,yEndTile)
        maxx=max(yStartTile,yEndTile)
        yStartTile=minn
        yEndTile=maxx
        #print(xStartTile,yStartTile,xEndTile,yEndTile)
        #print(0,0,xEndTile-xStartTile,yEndTile-yStartTile)
        imageCoord=[xStartTile,yStartTile,xEndTile,yEndTile]
        
        print("len(self.display_list1): ",len(self.display_list1))

        image_tiles=[]
        for can in self.display_list1:      
            print('x , y = ',can[0], can[1])
            if can[0]>=xStartTile and can[1]>=yStartTile and can[0]<=xEndTile and can[1]<=yEndTile:
                image_tiles=image_tiles+[can]
        image= self.generateimage(imageCoord,image_tiles) 
        im = Image.fromarray(image)
        
        #im.save("your_file.jpeg")
        window = tk.Toplevel()

        window.title("Join")
        window.geometry(str(image.shape[0])+'x'+str(image.shape[1]))
        window.configure(background='grey')
        img = ImageTk.PhotoImage(im)

        panel = tk.Label(window, image = img)
        
        #The Pack geometry manager packs widgets in rows or columns.
        panel.pack(side = "bottom", fill = "both", expand = "yes")
        

        window.geometry(str(img.width())+'x'+str(img.width()))

        window.mainloop()

        
    def motion_event(self, event): 
        if self.myrect is not None:
            self._canvas.delete(self.myrect) 
        canvas_x, canvas_y = self.get_canvas_coordinate(event.x, event.y)

        self.myrect=self._canvas.create_rectangle(self.start[0],self.start[1],canvas_x,canvas_y)

    def show_macro_label_image(self):
        """
        Method to display macro and label image.
        :return: None
        """
        macro_label_win = tk.Toplevel()
        macro_label_win.title("Macro and Label Image")
        macro_label_win_width = int(self._window.winfo_screenwidth()
                                    *Constants.macro_label_win_size_ratio[0])
        macro_label_win_height = int(self._window.winfo_screenheight()
                                     *Constants.macro_label_win_size_ratio[1])
        macro_label_win.geometry("{}x{}+200+100".format(str(macro_label_win_width),
                                                        str(macro_label_win_height)))
        macro_label_canvas = tk.Canvas(macro_label_win, background='white')
        macro_label_canvas.pack(side='left', expand='yes', fill='both')
        macro_image_width = macro_label_win_width * Constants.macro_image_size_ratio[0]
        macro_image_height = macro_label_win_height * Constants.macro_image_size_ratio[1]
        self.macro_image_data = self._process_tiles.get_sub_image("MACROIMAGE",
                                                                  (int(macro_image_width),
                                                                   int(macro_image_height)))
        label_image_width = macro_label_win_width * Constants.label_image_size_ratio[0]
        label_image_height = macro_label_win_height * Constants.label_image_size_ratio[1]
        self.label_image_data = self._process_tiles.get_sub_image("LABELIMAGE",
                                                                  (int(label_image_width),
                                                                   int(label_image_height)))
        macro_label_canvas.create_image(0, 0, image=self.macro_image_data, anchor=tk.NW)
        macro_label_canvas.create_image(int(macro_image_width), 0, image=self.label_image_data,
                                        anchor=tk.NW)
        macro_label_canvas.update_idletasks()

    def show_properties(self):
        """
        This method displays image properties over new window.
        :return: None
        """
        prop_win = tk.Toplevel()
        prop_win.title("iSyntax Properties")
        prop_win_width = self._window.winfo_screenwidth() * Constants.property_win_size_ratio[0]
        prop_win_height = self._window.winfo_screenheight() * Constants.property_win_size_ratio[1]
        prop_win.geometry("{}x{}+200+100".format(str(int(prop_win_width)),
                                                 str(int(prop_win_height))))
        prop_canvas = tk.Canvas(prop_win)
        scroll_y = tk.Scrollbar(prop_win, orient='vertical', command=prop_canvas.yview)
        scroll_x = tk.Scrollbar(prop_win, orient='horizontal', command=prop_canvas.xview)
        frame = tk.Frame(prop_canvas)
        row = 0
        for prop in Constants.property_list:
            # Dynamically create Label Widgets for each property
            # Structure them in a grid
            tk.Label(frame, text=prop[0], foreground='black',
                     font="Calibri 12").grid(sticky="w", row=row)
            tk.Label(frame, text=":", font="Calibri 14").grid(sticky="w", row=row, column=1)
            tk.Label(frame, text=str(prop[1]), foreground='blue',
                     font="Calibri 14").grid(sticky="w", row=row, column=2)
            row += 1
        prop_canvas.create_window(0, 0, anchor='nw', window=frame)
        prop_canvas.update_idletasks()
        scroll_y.pack(fill='y', side='right')
        scroll_x.pack(fill='x', side='bottom')
        prop_canvas.configure(scrollregion=prop_canvas.bbox('all'), yscrollcommand=scroll_y.set,
                              xscrollcommand=scroll_x.set)
        prop_canvas.pack(fill='both', expand=True, side='left')


    def register_mouse_events(self):
        """
        This method binds events to canvas.
        If an event matching the event description occurs in the widget, the given handler is
        called with an object describing the event
        :return: None
        """
        self._canvas.update_idletasks()
        self._canvas.focus_set()
        self._canvas.bind('<Key>', self.key)
        self._canvas.bind('<ButtonPress-1>', self.left_click)
        self._canvas.bind('<B1-Motion>', self.left_click_motion)
        self._canvas.bind('<ButtonRelease-1>', self.left_drop)
        self._canvas.bind('<MouseWheel>', self.mouse_wheel)
        self._canvas.bind('<Button-4>', self.mouse_wheel)
        self._canvas.bind('<Button-5>', self.mouse_wheel)
        self._canvas.bind('<Double-Button-1>', self.double_click_left)
        #self._canvas.bind('<Double-Button-3>', self.double_click_right)
        self._canvas.update_idletasks()
        #####
        self._canvas.bind("<Button-3>", self.click_event) 
        self._canvas.bind("<ButtonRelease-3>", self.release_event) 
        self._canvas.bind("<B3-Motion>", self.motion_event) 
        
    def display_canvas(self, canvas_x=0, canvas_y=0,flg=False):
        """
        This method displays new tiles by creating a bounding box around the desired location.
        Bounding box is then passed to processing tiles for fetching new tiles and removing
        unwanted tiles.
        :param canvas_x: canvasX coordinate.
        :param canvas_y: canvasY coordinate.
        :return: None
        """
        # Calculating current row and column of x_coord and y_coord
        #print('canvas_x:',canvas_x ,'canvas_y: ',canvas_y)

        current_col = int(canvas_y / Constants.tile_height)
        current_row = int(canvas_x / Constants.tile_width)
        #print('current_row:',current_row ,'current_col: ',current_col)

        # Creating a bounding box of current row and column
        bounding_box_start_row = current_row - Constants.tile_cache_horizontal
        bounding_box_start_col = current_col - Constants.tile_cache_vertical
        # Ensuring bounding box start does not lies in negative
        if bounding_box_start_col <= 0:
            bounding_box_start_col = 0
        if bounding_box_start_row <= 0:
            bounding_box_start_row = 0
        bounding_box_end_col = current_col + Constants.tile_cache_vertical
        bounding_box_end_row = current_row + Constants.tile_cache_horizontal
        display_list = self._process_tiles.processing_tiles([bounding_box_start_row,
                                                             bounding_box_start_col,
                                                             bounding_box_end_row,
                                                             bounding_box_end_col],
                                                            [current_row, current_col],flg)
        if display_list == 0:
            return
        self.display_list1=display_list
        for can in display_list:
            print('called : x, y =',can[0], can[1])
            self._canvas.create_image(can[0], can[1], image=(can[2]), anchor=tk.NW)
        self._canvas.update_idletasks()
        print('called:',len(display_list))

    def get_canvas_coordinate(self, x_coord, y_coord):
        """
        This method returns canvas coordinates of x_coord and y_coord
        :param x_coord: The X coordinate value for which the canvasX value is calculated.
        :param y_coord: The Y coordinate value for which the canvasY value is calculated.
        :return: canvas coordinates(canvasX,canvasY) for x_coord and y_coord
        """
        return self._canvas.canvasx(x_coord), self._canvas.canvasy(y_coord)

    def left_click(self, event):
        """
        Method for left click
        This method is called whenever a left click event occurs.
        Tkinter window title is updated as per the current mouse location.
        It displays current level and pixel indices of isyntax file.
        :param event: Contains details about the event that is occurred.
        :return: None
        """
        canvas_x, canvas_y = self.get_canvas_coordinate(event.x, event.y)
        self._canvas.scan_mark(event.x, event.y)
        pixel_indices = [canvas_x * 2 ** self._process_tiles.get_level(),
                         canvas_y * 2 ** self._process_tiles.get_level()]
        self._window.title("{}                Current Level: {}                "
                           "Pixel Indices: {}".format(self._title,
                                                      str(self._process_tiles.get_level()),
                                                      str(pixel_indices)))

    def left_drop(self, event):
        """
        Method for left click drop
        This method calls display_canvas() with the current canvas location and fetches new tiles if
        required.
        :param event: Contains details about the event that is occurred.
        :return: None
        """
        canvas_x_end, canvas_y_end = self.get_canvas_coordinate(
            event.x,
            event.y
        )
        self.display_canvas(canvas_x_end, canvas_y_end)

    def left_click_motion(self, event):
        """
        Method for left click motion
        This method constantly updates the title bar as the mouse left click(hold) is in motion.
        Also, it drags the canvas to current mouse location.
        :param event: Contains details about the event that is occurred.
        :return: None
        """
        canvas_x_end, canvas_y_end = self.get_canvas_coordinate(event.x, event.y)
        self._canvas.scan_dragto(event.x, event.y, Constants.pixel_pan)
        pixel_indices = [canvas_x_end * 2 ** self._process_tiles.get_level(),
                         canvas_y_end * 2 ** self._process_tiles.get_level()]
        self._window.title("{}                Current Level: {}                "
                           "Pixel Indices: {}".format(self._title,
                                                      str(self._process_tiles.get_level()),
                                                      str(pixel_indices)))

    def clear_canvas(self, level):
        """
        Method to clear canvas in case of level shift
        As soon as a valid mouse wheel event comes a level shift will happen,
        thus the previous level tiles are not required. Hence, they are removed from canvas.
        :param level: The previous level for which tiles are not required.
        :return: None
        """
        level_info = self._process_tiles.get_level_info_list()
        for row_col in level_info[level][3]:
            tile = level_info[level][4][row_col]
            tile.set_image(None)
        self._canvas.update_idletasks()

    def mouse_wheel(self, event):
        """
        Method for mouse wheel interaction and level shift
        :param event: Contains details about the event that is occurred.
        :return: On valid mouse wheel event it calls display_canvas() else returns to event
        processing loop.
        """
        canvas_x, canvas_y = self.get_canvas_coordinate(event.x, event.y)
        if event.delta == self._mouse_wheel_down or event.num == self._mouse_wheel_down:
            if self._process_tiles.get_level() + Constants.level_shift_factor >= \
                    self._process_tiles.get_max_levels():
                return
            shift_factor, x_coord, y_coord = self.zoom_out(event, canvas_x, canvas_y)
        elif event.delta == self._mouse_wheel_up or event.num == self._mouse_wheel_up:
            if self._process_tiles.get_level() - Constants.level_shift_factor < \
                    Constants.level_zero:
                return
            shift_factor, x_coord, y_coord = self.zoom_in(event, canvas_x, canvas_y)
        else:
            return
        self.level_shift([x_coord, y_coord], shift_factor, canvas_x, canvas_y)

    def zoom_out(self, event, canvas_x, canvas_y):
        """
        Method to zoom out
        :param event: Contains details about the event that is occurred.
        :param canvas_x: canvasX coordinate.
        :param canvas_y: canvasY coordinate.
        :return: shift_factor, x_coord, y_coord
        """
        shift_factor = 1
        x_coord = int(canvas_x / 2)
        y_coord = int(canvas_y / 2)

        self._canvas.scan_mark(int(event.x), int(event.y))
        self._canvas.scan_dragto(int(event.x + canvas_x / 2), int(event.y + canvas_y / 2),
                                 Constants.pixel_pan)
        return shift_factor, x_coord, y_coord

    def zoom_in(self, event, canvas_x, canvas_y):
        """
        Method to zoom in
        :param event: Contains details about the event that is occurred.
        :param canvas_x: canvasX coordinate.
        :param canvas_y: canvasY coordinate.
        :return: shift_factor, x_coord, y_coord
        """
        shift_factor = -1
        x_coord = int(canvas_x * 2)
        y_coord = int(canvas_y * 2)
        self._canvas.scan_mark(int(event.x), int(event.y))
        self._canvas.scan_dragto(int(event.x - canvas_x),
                                 int(event.y - canvas_y),
                                 Constants.pixel_pan)
        return shift_factor, x_coord, y_coord


    def double_click_left(self, event):
        """
        Method for left double click
        :param event: Contains details about the event that is occurred.
        :return: None
        """
        canvas_x, canvas_y = self.get_canvas_coordinate(event.x, event.y)
        if self._process_tiles.get_level() - Constants.level_shift_factor < Constants.level_zero:
            return
        shift_factor, x_coord, y_coord = self.zoom_in(event, canvas_x, canvas_y)
        self.level_shift([x_coord, y_coord], shift_factor, canvas_x, canvas_y)

    def double_click_right(self, event):
        """
        Method for right double click
        :param event: Contains details about the event that is occurred.
        :return: None
        """
        canvas_x, canvas_y = self.get_canvas_coordinate(event.x, event.y)
        if self._process_tiles.get_level() + Constants.level_shift_factor >= \
                self._process_tiles.get_max_levels():
            return
        shift_factor, x_coord, y_coord = self.zoom_out(event, canvas_x, canvas_y)
        self.level_shift([x_coord, y_coord], shift_factor, canvas_x, canvas_y)

    def level_shift(self, coord, shift_factor, canvas_x, canvas_y):
        """
        Method for level shift
        :param x_coord:  The X coordinate value for which the canvasX value is calculated.
        :param y_coord:  The Y coordinate value for which the canvasY value is calculated.
        :param shift_factor: The value by which the current level should be changed
        :param canvas_x: canvasX coordinate.
        :param canvas_y: canvasY coordinate.
        :return: None
        """
        self._process_tiles.set_level(self._process_tiles.get_level() + shift_factor)
        self._process_tiles.set_prev_tile_list([])
        self.display_canvas(coord[0], coord[1])
        self.clear_canvas(self._process_tiles.get_level() - shift_factor)
        pixel_indices = [canvas_x * 2 ** self._process_tiles.get_level(),
                         canvas_y * 2 ** self._process_tiles.get_level()
                         ]
        self._window.title("{}                Current Level: {}                "
                           "Pixel Indices: {}".format(self._title,
                                                      str(self._process_tiles.get_level()),
                                                      str(pixel_indices)))

    def key(self, event):
        """
        Method for key strokes
        :param event: Contains details about the event that is occurred.
        :return: None
        """
        canvas_x, canvas_y = self.get_canvas_coordinate(event.x, event.y)
        if event.char == "-":
            if self._process_tiles.get_level() + Constants.level_shift_factor >= \
                    self._process_tiles.get_max_levels():
                return
            shift_factor, x_coord, y_coord = self.zoom_out(event, canvas_x, canvas_y)
            self.level_shift([x_coord, y_coord], shift_factor, canvas_x, canvas_y)
        elif event.char == "+":
            if self._process_tiles.get_level() - Constants.level_shift_factor < \
                    Constants.level_zero:
                return
            shift_factor, x_coord, y_coord = self.zoom_in(event, canvas_x, canvas_y)
            self.level_shift([x_coord, y_coord], shift_factor, canvas_x, canvas_y)
