#!/usr/bin/env python

#python imports
import threading
import re
import sys
import time
import Queue
import os

#ROS imports
import rospy

#Other library imports
from Tkinter import *

#connor code imports
import connor_util as cutil

#SHARE_PATH = os.path.dirname(os.path.abspath(__file__))
#while not "share" in [f for f in os.listdir(SHARE_PATH) if not os.path.isfile(os.path.join(SHARE_PATH, f))]:
#    SHARE_PATH = os.path.dirname(SHARE_PATH)
#SHARE_PATH = os.path.join(SHARE_PATH, "share/")
#SHARE_PATH = cutil.findFile("share", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#if SHARE_PATH is None:
#    raise Exception("Please make sure there is a share directory for the GUI to write a logfile to.")
#SHARE_PATH = SHARE_PATH + "/"
#IMAGE_DIRECTORY = SHARE_PATH + "images/"
#LOG_FILE = SHARE_PATH + "gui_framework_stdout.log"

# http://tkinter.unpythonic.net/wiki/VerticalScrolledFrame

class VerticalScrolledFrame(Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling

    """
    def __init__(self, parent, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)            

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
        canvas = Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set, height=600)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())
        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)

class GUIFramework():
    def __init__(self, title="", buttons=[], divert_console=True):
    
        self.realStdOut = sys.stdout
        sys.stdout = self
    
        self.root = Tk()

        self.frame = Frame(self.root)
        self.frame.pack()
        
        self.menu_frame = VerticalScrolledFrame(self.frame, padx=5, pady=5, height=1000)
        self.menu_frame.grid(row=0, column=0, sticky=N+S+E+W)
        self.buttons = []
        self.menu_title = Label(self.menu_frame.interior, text="Menu")
        self.buttons_enabled = True
        
        self.current_title = None
        self.current_menu = None
        
        self.divert_console = divert_console
        if divert_console:
            self.console_scrollbar = Scrollbar(self.frame, orient=VERTICAL, width=10)
            self.console = Text(self.frame, yscrollcommand=self.console_scrollbar.set, bg = "black", fg = "gray", width=200, height=50, state=DISABLED)
            self.console_scrollbar.config(command=self.console.yview)
            self.console.grid(row=1, rowspan=4, column=0, sticky=N+S+E+W)
            self.console_scrollbar.grid(row=1, rowspan=4, column=1, sticky=W+N+S)
        
        self.log_file = open(LOG_FILE, "a")
        self.log_file.write("\n========================" + time.strftime("%d/%m/%Y") + ": " + time.strftime("%H:%M:%S") + "==============================\n")
        self.log_file.flush()
        
        self.update_lock = threading.Lock()
        self.queue_lock = threading.Lock()
        self.process_queue = Queue.Queue()
        self.gui_lock = threading.Lock()
        self.gui_queue = Queue.Queue()
        
        self.exit_flag = False
        self.root.protocol("WM_DELETE_WINDOW", self.__on_quit)
        
        self.__addMenu(title, buttons)
        
    def __on_quit(self):
        self.exit_flag = True
        
    def processJob(self, job):
        def job_callback():
            self.enableButtons(enable=False)
            job()
            self.enableButtons(enable=True)
        
        job_thread = threading.Thread(target=job_callback, args=())
        job_thread.daemon = True
        job_thread.start()
        
    def __wait_for_call(self, func):
        called = [False]
        def callback():
            func()
            called[0] = True
        
        self.gui_lock.acquire()
        self.gui_queue.put(callback)
        self.gui_lock.release()
        
        while not called[0]:
            rospy.sleep(0.01)
            
    def enableButtons(self, enable=True):
        self.__wait_for_call(cutil.make_closure(self.__enableButtons, [enable]))
        
    def __enableButtons(self, enable=True):
        self.buttons_enabled = enable
        for b in self.buttons:
            if enable:
                b.config(state=NORMAL)
            else:
                b.config(state=DISABLED)
                
    def areButtonsEnabled(self):
        return self.buttons_enabled
        
    def __destroy_menu(self):
        new_frame = VerticalScrolledFrame(self.frame, padx=5, pady=5, height=1000)
        new_frame.grid(row=0, column=0, sticky=N+S+E+W)
        self.menu_frame.destroy()
        del self.menu_frame
        self.menu_frame = new_frame
        for b in self.buttons:
            b.destroy()
            del b
        
        
    def addMenu(self, title, menu):
        self.__wait_for_call(cutil.make_closure(self.__addMenu, [title, menu]))
        
    def __addMenu(self, title, menu):
        self.__destroy_menu()
        
        self.current_title = title
        self.current_menu = menu
        
        self.menu_title = Label(self.menu_frame.interior, text=title, font=("Times", 30, "bold"))
        self.menu_title.pack()
        self.buttons = [self.menu_title]
        
        for title, func in menu:
            b = Button(self.menu_frame.interior, text=title, command=cutil.make_closure(self.__wait_for_button, [func]), font=("Times", 20, "normal"))
            b.pack()
            self.buttons.append(b)
            
    def __wait_for_button(self, func, args=()):
#        self.enableButtons(enable=False)
#    
#        thread = threading.Thread(target=func, args=args)
#        thread.daemon = True
#        thread.start()
#        
#        r = rospy.Rate(500)
#        while thread.isAlive():
#            self.root.update()
#            r.sleep()
#            
#        self.enableButtons(enable=True)
        self.queue_lock.acquire()
        self.process_queue.put(cutil.make_closure(func, args))
        self.queue_lock.release()
        
        
    def addEntryPrompt(self, title, okay_func, cancel_func=None):
        self.__wait_for_call(cutil.make_closure(self.__addEntryPrompt, [title, okay_func, cancel_func]))
        
    def __addEntryPrompt(self, title, okay_func, cancel_func=None):
        self.__destroy_menu()
        
        self.current_title = title
        self.current_menu = None
        
        self.menu_title = Label(self.menu_frame.interior, text=title, font=("Times", 30, "bold"))
        self.menu_title.pack()
        self.buttons = [self.menu_title]
        
        entry = Entry(self.menu_frame.interior, font=("Times", 20, "normal"))
        entry.pack()
        self.buttons.append(entry)
        def okay_callback():
            self.__wait_for_button(func=cutil.make_closure(okay_func, [entry.get()]))
        def entry_callback(event):
            okay_callback()
        entry.bind('<Return>', entry_callback)
        entry.bind('<KP_Enter>', entry_callback)
        
            
        okay = Button(self.menu_frame.interior, text="Okay", command=okay_callback, font=("Times", 20, "normal"))
        okay.pack()
        self.buttons.append(okay)
        
        if cancel_func is not None:
            cancel = Button(self.menu_frame.interior, text="Cancel", command=cutil.make_closure(self.__wait_for_button, [cancel_func]), font=("Times", 20, "normal"))
            cancel.pack()
            self.buttons.append(cancel)
            
        entry.focus_set()

    
    def cleanup(self):
        sys.stdout = self.realStdOut
        self.log_file.close()
        
    def runMainLoop(self):
        #self.root.mainloop()
        while not self.exit_flag:
            self.update_lock.acquire()
            self.root.update()
            self.update_lock.release()
            
            job = None
            self.queue_lock.acquire()
            if not self.process_queue.empty():
                job = self.process_queue.get()
            self.queue_lock.release()
            if job is not None:
                self.processJob(job)
                
            while True:
                self.gui_lock.acquire()
                if self.gui_queue.empty():
                    self.gui_lock.release()
                    break
                else:
                    job = self.gui_queue.get()
                    self.gui_lock.release()
                    job()
            
        self.cleanup()
        
        
    def printLine(self, val):
        self.printString(val + "\n")
        
    def printString(self, val):
        self.__wait_for_call(cutil.make_closure(self.__printString, [val]))
        
    def __printString(self, val):
        self.log_file.write(val)
        self.log_file.flush()
        
        if self.divert_console:
            self.console.config(state=NORMAL)
            while val.find('\b') >= 0:
                self.console.insert(END, val[0:val.find('\b')])
                val = val[val.find('\b'):]
                i = re.search('[^\b]', val)
                if i is None:
                    i = len(val)
                else:
                    i = i.start()
                self.console.delete("%s-%d chars" % (END, i+1), END)
                val = val[i:]
            self.console.insert(END, val)
                    
            self.console.config(state=DISABLED)
            self.console.yview(END)
        else:
            self.realStdOut.write(val)
            self.realStdOut.flush()
        
    def write(self, val):
        self.printString(val)
        
    def flush(self):
        pass
    
    
def display_menu(title, menu, include_back=True):
    global gui
    previous_title = gui.current_title
    previous_menu = gui.current_menu
    previous_enable = gui.areButtonsEnabled()
    
    if previous_menu is not None:
        def back():
            global gui
            gui.addMenu(previous_title, previous_menu)
            gui.enableButtons(enable=previous_enable)
        if include_back:
            menu.append(("Back", back))
    else:
        back = None
        
    gui.addMenu(title, menu)
    return back
    
    
def select_from_list(title, options, cancel=True):
    print(title)
    menu = []
    locks = []
    def lock_callback(to_lock):
        to_lock.acquire()
    
    for i in range(len(options)):
        lock = threading.Lock()
        locks.append(lock)
        menu.append((options[i], cutil.make_closure(lock_callback, [lock])))
        
    cancel_lock = threading.Lock()
    if cancel:
        menu.append(("Cancel", cutil.make_closure(lock_callback, [cancel_lock])))
    
    global gui
    previous_menu = gui.current_menu
    previous_title = gui.current_title
    previous_enable = gui.areButtonsEnabled()
    gui.addMenu(title, menu)
    
    r = rospy.Rate(500)
    ret = None
    while ret is None:
        for i in range(len(locks)):
            if not locks[i].acquire(False):
                ret = i
            else:
                locks[i].release()
        if not cancel_lock.acquire(False):
            ret = -1
        else:
            cancel_lock.release()
        r.sleep()
    
    gui.addMenu(previous_title, previous_menu)
    gui.enableButtons(enable=previous_enable)
    if ret >= 0:
        print("Selected " + options[ret])
    else:
        print("Selected Cancel")
    return ret
    
def toggle_button(title, button1, button2):
    global gui
    previous_menu = gui.current_menu
    previous_title = gui.current_title
    previous_enable = gui.areButtonsEnabled()
    
    def back():
        global gui
        gui.addMenu(previous_title, previous_menu)
        gui.enableButtons(enable=previous_enable)
    
    def action1():
        global gui
        gui.addMenu(title, [(button2[0], button2[1]), ("Back", back)])
        button1[1]()
    
    def action2():
        global gui
        gui.addMenu(title, [(button1[0], button1[1]), ("Back", back)])
        button2[1]()
        
    gui.addMenu(title, [(button1[0], button1[1]), ("Back", back)])
    
    
def continue_or_cancel(title="Continue or Cancel"):
    if select_from_list(title, ["Continue"]) >= 0:
        return True
    else:
        return False
        
def click_to_continue(title="Press button to continue"):
    select_from_list(title, ["Continue"], cancel=False)
    
def getKeyboardLine(title, input_type=str, cancel=True):
    print(title)
    okay_lock = threading.Lock()
    value = []
    def okay_callback(text):
        try:
            input_type(text)
            value.append(input_type(text))
            okay_lock.acquire()
        except ValueError:
            print("ERROR: Please input a value of type " + str(input_type))
        
    cancel_lock = threading.Lock()
    def cancel_callback():
        cancel_lock.acquire()
    
    global gui
    previous_menu = gui.current_menu
    previous_title = gui.current_title
    previous_enable = gui.areButtonsEnabled()
    
    if cancel:
        gui.addEntryPrompt(title, okay_callback, cancel_callback)
    else:
        gui.addEntryPrompt(title, okay_callback)
    
    r = rospy.Rate(500)
    ret = None
    while True:
        if not cancel_lock.acquire(False):
            ret = None
            break
        else:
            cancel_lock.release()
        if not okay_lock.acquire(False):
            ret = value[0]
            break
        else:
            okay_lock.release()
        r.sleep()
    
    gui.addMenu(previous_title, previous_menu)
    gui.enableButtons(enable=previous_enable)
    if ret is not None:
        print(">" + str(ret))
    else:
        print(">")
    return ret
    
    
class MenuMonitor():
    def __init__(self, title, buttons):
        global gui
        menu = [(x, cutil.make_closure(self.__button_callback, [y])) for x, y in zip(buttons, range(len(buttons)))]
        self.counts = [0]*len(buttons)
        self.count_lock = threading.Lock()
        self.buttons = buttons
        
        self.previous_menu = gui.current_menu
        self.previous_title = gui.current_title
        self.previous_enable = gui.areButtonsEnabled()
        
        gui.addMenu(title, menu)
        print(title)
        
    def __button_callback(self, x):
        self.count_lock.acquire()
        self.counts[x] = self.counts[x] + 1
        self.count_lock.release()
        print("Selected " + self.buttons[x])
        
    def buttonPresses(self, x):
        self.count_lock.acquire()
        ret = self.counts[x]
        self.counts[x] = 0
        self.count_lock.release()
        return ret
        
    def numButtons(self):
        return len(self.buttons)
        
    def cleanup(self):
        global gui
        gui.addMenu(self.previous_title, self.previous_menu)
        gui.enableButtons(enable=self.previous_enable)
        self.previous_menu = None
        
    def __del__(self):
        if self.previous_menu is not None:
            self.cleanup()

    
gui = None
def init_gui(main_menu, divert_console=True):
    global gui
    gui = GUIFramework("Main Menu", main_menu, divert_console)
    gui.runMainLoop()
    
def exit_gui():
    global gui
    gui.exit_flag = True
    gui.root.destroy()
    
    
#def test(v):
#    v2 = getKeyboardLine("What kind of menu you want this menu to make?", cancel=False)
#    display_menu("Menu: " + v, [("button", cutil.make_closure(test, [v2]))])
#    
#def waist_time():
#    sec = getKeyboardLine("How long would you like to countdown for? (s)", input_type=int)
#    if sec is None:
#        return
#    click_to_continue("Begin the countdown!")
#    for i in range(sec):
#        print(str(i) + " seconds")
#        rospy.sleep(1.0)
#    
#    
#rospy.init_node('gui_framework', disable_signals=True)
#init_gui([("a button", cutil.make_closure(test, ["LOL"])), ("another button", waist_time)])
#rospy.signal_shutdown("finished.")





















