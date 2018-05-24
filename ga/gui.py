# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:32:06 2015

@author: roberto
"""

# -*- coding: utf-8 -*-
"""
to do:
- comment out parameter controls on page 2.
- see if the functions could just be left alone, but are just not active
during runs. the controls could also just be disabled.
- can also comment out the reading and writting of parameters.
- same with all regarding the settings csv.
- change fitness_history.csv readout to read now from pyevolve
generated file.
"""
from PyQt4 import QtGui, QtCore
import sys
# import hdf2py
import pandas as pd
import numpy as np
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg \
     import NavigationToolbar2QT as NavigationToolbar
#import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import time
import os
import traceback
# import matplotlib.pyplot as plt
#import types
#import matplotlib.pyplot as plt
#import copy

# the following is for using latex fonts in the plots:
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# # getting the temp file (assuming this file is in a subfolder of ga.py's directory:
# current_directory = os.path.abspath(os.path.curdir)
# parent_directory = os.path.dirname(current_directory)


def ga_folder():
    # import sys
    # import os
    this_file_path = sys.modules[__name__].__file__
    return os.path.dirname(this_file_path)


temp_file_default =\
    os.path.join(ga_folder(), '.__ga_temp__.h5')
    # os.path.join(os.path.dirname(os.getcwd()), "ga__temp__.h5")
# r"/home/roberto/Documents/Projects/ND_HEOD_Spectroscopy/data/2015-05-13/ga__temp__.h5"


class gui(QtGui.QMainWindow):
    def __init__(self,
                 temp_file_default = temp_file_default):
        QtGui.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("client gui for ga")
        
        self.temp_file = temp_file_default
        
        self.build_menu_bar()

        self.tabs = tabs = QtGui.QTabWidget() 
        page1 = self.build_page1()
        page2 = self.build_page2()
        page3 = self.build_page3()
        
        tabs.addTab(page1, "settings")
        tabs.addTab(page2, "optimization progress")
        # tabs.addTab(page3, "current measurements")
        
        # this is important for rendering and autosizing somehow...
#        self.main_widget.setFocus()
        self.setCentralWidget(tabs)
        
        self.set_initial_widget_values()
        
        
        # set the gui to update every second:
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_gui)
        timer.start(1000)
        
    def build_page1(self):
        # define widgets:
        page = QtGui.QWidget()
        
        # self.temp_file_path_control =\
        #     create_widget("select file")
        # tmp_file = labeled_widget(self.temp_file_path_control,
        #                           "path to ga temp database")
        #
        # self.current_measurement_control = \
        #     create_widget("select file")
        # curr_meas_file = labeled_widget(self.current_measurement_control,
        #                           "path to current measurement file")
        #
        # self.ga_running_control = \
        #     create_widget("select file")
        # ga_running_file = labeled_widget(self.ga_running_control,
        #                                 "path to ga running file")

        self.stop_control = \
            create_widget("select file")
        stop_file = labeled_widget(self.stop_control,
                                        "path to stop file")

        self.fitness_history_file_control = \
            create_widget("select file")
        fitness_history_control = labeled_widget(self.fitness_history_file_control,
                                        "path to fitness history file")
                                  
#        self.update_settings_button = btn1 =\
#        create_widget(widget_type = "button",
#                      action = self.update_settings_button_pressed,
#                      description = "update\nsettings")
        
        # self.scan_settings_indicator =\
        # create_widget("multiline text")
        # sc_settings = labeled_widget(self.scan_settings_indicator,
        #                              "scan settings")
        
        # place widgets on main widget:
        self.place_widgets \
            ([[stop_file],
              [fitness_history_control]],
             page)
        # self.place_widgets\
        # ([[tmp_file],
        #   [curr_meas_file],
        #   [ga_running_file],
        #   [stop_file],
        #   [fitness_history_control],
        #   [sc_settings]],
        # page)
        
        return page
        
    def build_page2(self):
        # define widgets:
        page = QtGui.QWidget()
                      
        
#        ind1 = labeled_widget(self.proj_to_plot_widget,
#                              "index of\nprojection\nto plot")
        
        self.fitness_history_plot =\
        MyMplCanvas(page, width=5, height=10, dpi=100)
        #create_widget("matplotlib canvas")
        snav = MplCanvasWithNav(self.fitness_history_plot)
        hist = labeled_widget(snav, "fitness history")
        
        # right column with controls and stop button:
        r_col = create_widget("container")
        
        self.par_mut_prob_widget = create_widget("numeric")
        par_mut = labeled_widget(self.par_mut_prob_widget, 
                                 "parameter mutation probability")
                                 
        self.mut_prob_widget = create_widget("numeric")
        mut_prob = labeled_widget(self.mut_prob_widget, 
                                 "mutation probability")

        self.mut_sigma_widget = create_widget("numeric")
        mut_sigma = labeled_widget(self.mut_sigma_widget,
                                 "mutation sigma")

        self.mild_mut_sigma_widget = create_widget("numeric")
        mild_mut_sigma = labeled_widget(self.mild_mut_sigma_widget,
                                   "mild mutation sigma")

                                 
        self.x_prob_widget = create_widget("numeric")
        x_prob = labeled_widget(self.x_prob_widget, 
                                 "crossover probability")
                                 
        self.n_elit_widget = create_widget("numeric")
        n_elit = labeled_widget(self.n_elit_widget, 
                                 "number of elitsts")
                                 
        self.refresh_opt_pars_button = refsh =\
        create_widget(widget_type = 'button', 
                      description = "refresh",
                      action = self.refresh_opt_pars_button_pressed)
                                 
        self.upload_opt_pars_button = updt =\
        create_widget(widget_type = 'button', 
                      description = "upload changes",
                      action = self.upload_opt_pars_button_pressed)
        
        self.stop_button = stp =\
        create_widget(widget_type = 'button', 
                      description = "stop ga",
                      action = self.stop_button_pressed)
                      
        #place widgets on right column:
        self.place_widgets\
        ([[par_mut],
          [mut_prob],
          [mut_sigma],
          [mild_mut_sigma],
          [x_prob],
          [n_elit],
          [refsh],
          [updt],
          [stp]], r_col)
        
        # place widgets on main widget:
        self.place_widgets \
            ([[hist], [stp]],
             page)
        # self.place_widgets\
        # ([[hist, r_col]],
        # page)
        
        return page
    
    def build_page3(self):
        """
        current measurements along with the corresponding slm masks.
        """
        # define widgets:
        page = QtGui.QWidget()
        
        self.current_index_widget = create_widget("text")
        sig = labeled_widget(self.current_index_widget, "current index")
        
        self.curr_params_plot = create_widget("matplotlib canvas")
        slmnav = MplCanvasWithNav(self.curr_params_plot)
        params_plot = labeled_widget(slmnav, "current parameters")
        
        # place widgets on main widget:
        self.place_widgets\
        ([[sig],[params_plot]],
        page)
        
        return page 
        
    
    def update_gui(self):
        """
        update the plots, status bar, and progress bar. will update each 
        tab only when its active.
        """
        # # page 1 active:
        # if self.tabs.currentIndex() == 0:
        #     self.get_ga_settings()
        
        # page 2 active:
        if self.tabs.currentIndex() == 1:
            self.get_ga_progress()
            self.refreshStopButtonState()
#            self.check_optimization_parameters()
            
        # # page 3 active:
        # elif self.tabs.currentIndex() == 2:
        #     # self.get_current_signal()
        #     self.get_current_ga_parameters()
        #
        # self.get_engine_status()
        
    def place_widgets(self, widgets_list = [], 
                      parent = None):
        l = QtGui.QVBoxLayout(parent)
        for widget_row in widgets_list:
            row = QtGui.QWidget()
            row_layout = QtGui.QHBoxLayout(row)
            for widget in widget_row:
                row_layout.addWidget(widget)
            l.addWidget(row)
#        """
#        recursively place the widgets in widgets_list.
#        """
#        l = QtGui.QVBoxLayout(parent)
#        for widget in widgets_list:
##            row = QtGui.QWidget(parent = parent)
##            row_layout = QtGui.QHBoxLayout(row)
#            if isinstance(widget, types.ListType):
#                row = QtGui.QWidget()
#                self.place_widgets(widget, row)
#                l.addWidget(row)
#            else:
#                l.addWidget(widget)
            
    def set_initial_widget_values(self):
        try:
            # fname_settings = os.path.join(ga_folder(), '.__ga_settings__.csv')
            # set_widget_value(self.temp_file_path_control, fname_settings)
            #
            # fname = os.path.join(ga_folder(), '.__current_measurement__.csv')
            # set_widget_value(self.current_measurement_control, fname)
            #
            # fname = os.path.join(ga_folder(), '.__ga_running__')
            # set_widget_value(self.ga_running_control, fname)

            fname = os.path.join(ga_folder(), '.stop')
            set_widget_value(self.stop_control, fname)

            fname = os.path.join(ga_folder(), '.__fitness_history__.csv')
            set_widget_value(self.fitness_history_file_control, fname)

            # self.get_ga_settings()
            # self.get_engine_status()
            self.statusBar().showMessage("initializing gui", 2000)

            # # now initialize the parameters that can be changed during the
            # # optimization:
            # df = pd.read_csv(fname_settings)
            # s = df.iloc[-1].copy()
            #
            # addr = "parameter_mutation_probability"
            # value = s[addr]
            # set_widget_value(self.par_mut_prob_widget, value)
            #
            # addr = "mutation_probability"
            # value = s[addr]
            # set_widget_value(self.mut_prob_widget, value)
            #
            # addr = "mutation_sigma"
            # value = s[addr]
            # set_widget_value(self.mut_sigma_widget, value)
            #
            # addr = "mild_mutation_sigma"
            # value = s[addr]
            # set_widget_value(self.mild_mut_sigma_widget, value)
            #
            # addr = "crossover_probability"
            # value = s[addr]
            # set_widget_value(self.x_prob_widget, value)
            #
            # addr = "n_elitists"
            # value = s[addr]
            # set_widget_value(self.n_elit_widget, int(value))
        except:
            t = time.strftime("at %H:%M:%S on %Y/%m/%d")
            print "couldn't set_initial_widget_values {}".format(t)
            traceback.print_exc()
    
    def get_ga_settings(self):
        try:
            fname = get_widget_value(self.temp_file_path_control)
            # fname = os.path.join(ga_folder(), '.__ga_settings__.csv')
            df = pd.read_csv(fname)
            s = df.iloc[-1].copy()

            keys = s.index
            values = s.values

            # read all the attributes on the root group:
            text = ""
            for key, value in zip(keys, values):
                text += key + ": " + str(value) + "\n"

            set_widget_value(self.scan_settings_indicator, text)

        except:
            t = time.strftime("at %H:%M:%S on %Y/%m/%d")
            print "couldn't update scan settings {}".format(t)
            traceback.print_exc()

    def get_current_ga_parameters(self):
        try:
            # hdf = self.temp_file
            # addr = "current_parameters"
            # params_exist = hdf2py.is_object_in_hdf(addr, hdf)
            fname = get_widget_value(self.current_measurement_control)
            # fname = os.path.join(ga_folder(), '.__current_measurement__.csv')
            s = pd.Series.from_csv(fname)

            if self.engine_on:
                params = s['parameters']
                params = params.replace('[','').replace(']','')
                params = params.split(',')
                params = [float(par) for par in params]

                x = np.arange(np.size(params))
                self.curr_params_plot.axes.bar(x, params)
                # self.curr_params_plot.axes.plot(params)

                # get the current index in the generation:
                ind_indx = s['index']
                set_widget_value(self.current_index_widget,
                                 str(ind_indx))

                # add the signal as an annotation to the plot:
                sig = s['fitness']
                # ax = plt.gca()
                ax = self.curr_params_plot.axes
                text = 'current signal = {}'.format(sig)
                # text += ', %s'%ind_indx
                ax.annotate(text, xy=(0., 1.02), xycoords='axes fraction')

                self.curr_params_plot.draw()
        except:
            t = time.strftime("at %H:%M:%S on %Y/%m/%d")
            print "couldn't get current ga parameters {}".format(t)
            traceback.print_exc()
            
    # def get_current_signal(self):
    #     sig = 0
    #     try:
    #         if self.engine_on:
    #             fname = os.path.join(ga_folder(), '.__current_measurement__.csv')
    #             s = pd.Series.from_csv(fname)
    #             sig =s['fitness']
    #             set_widget_value(self.current_signal_widget, str(sig))
    #
    #     except:
    #         t = time.strftime("at %H:%M:%S on %Y/%m/%d")
    #         print "couldn't get current signal {}".format(t)
    #         traceback.print_exc()
    #
    #     return sig

    def get_current_individual_index(self):
        sig = ""
        try:
            if self.engine_on:
                fname = get_widget_value(self.current_measurement_control)
                # fname = os.path.join(ga_folder(), '.__current_measurement__.csv')
                s = pd.Series.from_csv(fname)
                sig = s['index']

        except:
            t = time.strftime("at %H:%M:%S on %Y/%m/%d")
            print "couldn't get current individual index {}".format(t)
            traceback.print_exc()

        return sig
        
    def get_ga_progress(self):
        try:
            fname = os.path.join(ga_folder(), '.__fitness_history__.csv')

            # get the cumulated signals:
            if not hasattr(self, "fitness_history_shape"):
                self.fitness_history_shape = ()

            if os.path.exists(fname):
                df = pd.read_csv(fname)
                cols = [x for x in df.columns]
                data = df.values#[:, 1:]
                shape = np.shape(data)

                if np.size(shape) == 2 and shape != self.fitness_history_shape:
                    self.fitness_history_plot.axes.plot(data)
                    self.fitness_history_plot.axes.legend(cols,
                                                          loc='best')
                    # self.fitness_history_plot.axes.xlabel('generation #')
                    self.fitness_history_plot.draw()
                    self.fitness_history_shape = shape

        except:
            t = time.strftime("at %H:%M:%S on %Y/%m/%d")
            print "couldn't get ga progress {}".format(t)
            traceback.print_exc()
        
    def get_engine_status(self):
        try:
            # hdf = self.temp_file
            # addr = "ga_running"
            # engine_on = self.engine_on = hdf2py.get_hdf_attribute(addr, hdf)
            fname = get_widget_value(self.ga_running_control)
            # fname = os.path.join(ga_folder(), '.__ga_running__')
            engine_on = self.engine_on = os.path.exists(fname)
            stp_btn = self.stop_button
            
            if engine_on:
                stp_btn.setEnabled(True)
                text = "ga running"
                # text += ", current scan row: " +\
                # str( hdf2py.get_hdf_dataset(hdf, "current_scan_row") )
                self.statusBar().showMessage(text, 2000)
            else:
                stp_btn.setEnabled(False)
                self.stop_button.setText("stop ga")
                set_widget_value(self.stop_button, False)
                text = "ga not running"            
                self.statusBar().showMessage(text, 2000)
        except:
            t = time.strftime("at %H:%M:%S on %Y/%m/%d")
            print "couldn't get the engine status {}".format(t)
            traceback.print_exc()
        # show in status bar, e.g.: 
        # current scan row: (-1000, 780), scan program running: True
#        self.statusBar().showMessage("All hail matplotlib!", 2000)
            
    def refresh_optimization_parameters(self):
        try:
            fname = os.path.join(ga_folder(), '.__ga_settings__.csv')
            df = pd.read_csv(fname)
            s = df.iloc[-1].copy()

            addr = "parameter_mutation_probability"
            value = s[addr]
            set_widget_value(self.par_mut_prob_widget, value)

            addr = "mutation_probability"
            value = s[addr]
            set_widget_value(self.mut_prob_widget, value)

            addr = "mutation_sigma"
            value = s[addr]
            set_widget_value(self.mut_sigma_widget, value)

            addr = "mild_mutation_sigma"
            value = s[addr]
            set_widget_value(self.mild_mut_sigma_widget, value)

            addr = "crossover_probability"
            value = s[addr]
            set_widget_value(self.x_prob_widget, value)

            addr = "n_elitists"
            value = s[addr]
            set_widget_value(self.n_elit_widget, int(value))
        except:
            print traceback.print_exc()
            
    def save_changes_to_optimization_parameters(self):
        # now initialize the parameters that can be changed during the 
        # optimization:
        try:
            fname = os.path.join(ga_folder(), '.__ga_settings__.csv')
            df = pd.read_csv(fname)
            s = df.iloc[-1].copy()
            
            addr = "parameter_mutation_probability"
            value = get_widget_value(self.par_mut_prob_widget)
            old_value = s[addr]
            if old_value != value:
                s[addr] = value
                
            addr = "mutation_probability"
            value = get_widget_value(self.mut_prob_widget)
            old_value = s[addr]
            if old_value != value:
                s[addr] = value
            
            addr = "mutation_sigma"
            value = get_widget_value(self.mut_sigma_widget)
            old_value = s[addr]
            if old_value != value:
                s[addr] = value

            addr = "mild_mutation_sigma"
            value = get_widget_value(self.mild_mut_sigma_widget)
            old_value = s[addr]
            if old_value != value:
                s[addr] = value
            
            addr = "crossover_probability"
            value = get_widget_value(self.x_prob_widget)
            old_value = s[addr]
            if old_value != value:
                s[addr] = value
            
            addr = "n_elitists"
            value = get_widget_value(self.n_elit_widget)
            old_value = s[addr]
            if old_value != value:
                s[addr] = value

            append_row_to_csv(s.values, fname)

        except:
            t = time.strftime("at %H:%M:%S on %Y/%m/%d")
            print "couldn't check_optimization_parameters {}".format(t)
            traceback.print_exc()
    
    def stop_scan(self):
        # folder = os.path.dirname(temp_file_default)
        folder = ga_folder()
        stop_file = os.path.join(folder, '.stop')
        create_file(stop_file)
        # hdf = self.temp_file
        # addr = "stop"
        # hdf2py.set_hdf_attribute(1, addr, hdf, True)
    
    def unstop_scan(self):
        # folder = os.path.dirname(temp_file_default)
        folder = ga_folder()
        stop_file = os.path.join(folder, '.stop')
        delete_file(stop_file)
        # hdf = self.temp_file
        # addr = "stop"
        # hdf2py.set_hdf_attribute(0, addr, hdf, True)
    
    def stop_button_pressed(self):
        # hdf = self.temp_file
        # addr = "stop"
        # stop = not( hdf2py.get_hdf_attribute(addr, hdf) )
        fname = os.path.join(ga_folder(), '.stop')
        stop = not os.path.exists(fname)
#        self.stop_signal = not(self.stop_signal)
        
        if stop:
            self.stop_scan()
            self.stop_button.setText("stopping...")
            set_widget_value(self.stop_button, True)
        else:
            self.unstop_scan()
            self.stop_button.setText("stop ga")
            set_widget_value(self.stop_button, False)

    def refreshStopButtonState(self):
        fname = os.path.join(ga_folder(), '.stop')
        stop = not os.path.exists(fname)
        if not stop:
            self.stop_button.setText("stopping...")
            set_widget_value(self.stop_button, True)
        else:
            self.stop_button.setText("stop ga")
            set_widget_value(self.stop_button, False)
            
    def update_field_button_pressed(self, *args):
        self.get_field()
        set_widget_value(self.update_field_button, False)
        
    def refresh_opt_pars_button_pressed(self, *args):
        self.refresh_optimization_parameters()
        set_widget_value(self.refresh_opt_pars_button, False)
        
    def upload_opt_pars_button_pressed(self, *args):
        self.save_changes_to_optimization_parameters()
        set_widget_value(self.upload_opt_pars_button, False)
        
#    def update_settings_button_pressed(self, *args):
#        self.get_ga_settings()
#        set_widget_value(self.update_settings_button, False)
    
    def build_menu_bar(self):
        self.file_menu = QtGui.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtGui.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtGui.QMessageBox.about(self, "About",
"""embedding_in_qt4.py example
Copyright 2005 Florent Rougon, 2006 Darren Dale

This program is a simple example of a Qt4 application embedding matplotlib
canvases.

It may be used and modified with no restriction; raw copies as well as
modified versions may be distributed without limitation."""
)

#### using the matplotlibt backend to qt:
class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig = fig
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class MplCanvasWithNav(QtGui.QWidget):
    def __init__(self, mpl_canvas_widget):
        QtGui.QWidget.__init__(self)
        vbox = QtGui.QVBoxLayout(self)
        nav = NavigationToolbar(mpl_canvas_widget, self)
        vbox.addWidget(mpl_canvas_widget)
        vbox.addWidget(nav)
        
def plotyy_in_canvas(canvas, x, y1, y2):
    canvas.axes.hold(False)
    canvas.axes.plot(x, y1, 'b')
    if not hasattr(canvas, "axes2"):
        canvas.axes2 =\
        canvas.axes.twinx()
    canvas.axes2.hold(False)
    canvas.axes2.plot(x, y2, 'g')
    canvas.draw()

#### qt gui components:
def start_application(title = "qt application",
                      minimum_width = 500,
                      minimum_height = 500):
    ## Always start by initializing Qt (only once per application)
#    qApp = QtGui.QApplication(sys.argv)
    app = QtGui.QApplication([title])
#    app = QtGui.QApplication(sys.argv)
#    aw = ApplicationWindow()
#    aw.setWindowTitle("%s" % progname)
    
    ## Define a top-level widget to hold everything
    main_window = QtGui.QWidget()
    main_window.setMinimumHeight(minimum_height)
    main_window.setMinimumWidth(minimum_width)
#    main_window.setSizePolicy(QtGui.QSizePolicy.Expanding,
#                              QtGui.QSizePolicy.Expanding)
#    main_window.updateGeometry()
    return main_window, app
    
#def place_widgets(widgets = [[]], parent_window = None):
#    """
#    widgets is a 2D list of widgets. the widgets are placed in the gui in 
#    the order indicated in the list (i.e., on the same indicated rows).
#    Can be used to create nested widgets by having the parent_window be 
#    a container widget (see create_widget below).
#    """
#    if not parent_window:
#        parent_window = start_application()
##    layout = QtGui.QGridLayout()
#    cont = QtGui.QWidget(parent = parent_window)
#    layout = QtGui.QVBoxLayout(cont)
##    parent_window.setLayout(layout)
#    for widget_row in widgets:
#        row = QtGui.QWidget()
#        row_layout = QtGui.QHBoxLayout(row)
##        row.setLayout(row_layout)
#        for widget in widget_row:
#            row_layout.addWidget(widget)
#        layout.addWidget(row)

def create_widget(widget_type = 'button', 
                  value = "",
                  action = None,
                  description = ""):
    """
    implemented widgets: container, button, numeric, text, list, plot,
    3D plot, progress bar.
    """
    wn = widget_type
    
    def run_action(*args, **kwargs):
        #run_on_thread(action, *args, **kwargs)
        # this was removed because it's not possible to use threads with qt 
        # (or can't call the qt objects from child threads). The solution 
        # is to call processEvents() everywhere there's a long process that 
        # should/can be stopped.
        action()
    
    if wn == "container":
        w = QtGui.QWidget()
    elif wn == 'button':
        w = QtGui.QPushButton(description)
        w.setCheckable(True)
        if action:        
            w.clicked.connect(run_action)
    elif wn == "numeric":
        w = QtGui.QDoubleSpinBox(value=0.0, decimals=4)
#        w = QtGui.QSpinBox(value=0.0)
    elif wn == "text":
        w = QtGui.QLineEdit(value)
    elif wn == "label":
        w = QtGui.QLabel(value)
    elif wn == "list":
        w = QtGui.QListWidget()
    elif wn == "matplotlib canvas":
        w = MyMplCanvas()
    elif wn == "progress bar":
        w = QtGui.QProgressBar()
    elif wn == "select file":
        w = select_file_widget()
    elif wn == "multiline text":
        w = multiline_text_widget()
    else:
        print "widget {} isn't supported yet...".format(wn)

    w.widget_type = widget_type
    return w

def labeled_widget(widget, label = ""):
    box = QtGui.QWidget()
    layout = QtGui.QGridLayout()
    box.setLayout(layout)
    label_object = QtGui.QLabel(label)
    layout.addWidget(label_object)
    layout.addWidget(widget)
    box.widget = widget
    box.label = label_object
    return box

def change_label(labeled_widget, new_label = ""):
    labeled_widget.label.setText(new_label)

def set_widget_value(widget, value):
    try:
        wt = widget.widget_type
    except:
        wt = "not supported"
    if wt == "numeric":
        widget.setValue( float(value) )
    elif wt == "text":
        widget.setText( str(value) )
    elif wt == "multiline text":
        widget.setValue( str(value) )
#        widget.insertPlainText( str(value) )
    elif wt == "button":
        widget.setDown( bool(value) )#presed down is True
        widget.setChecked( bool(value) )
    elif wt == "progress bar":
        widget.setValue( float(value) )#from 0 to 100
    else:
        try:
            widget.setValue(value)
        except:
            print "widget type: {} is not supported yet :(".format(wt)

def get_widget_value(widget):
    try:
        # widget.valueChanged()
        wt = widget.widget_type
    except:
        wt = "not supported"
    if wt == "numeric":
        return widget.value()
    elif wt == "text":
        # return widget.text()
        return widget.displayText()
    elif wt == "multiline text":
        return widget.toPlainText()
    elif wt == "button":
        return widget.isDown()
    elif wt == "progress bar":
        return widget.value()
    else:
        try:
            return widget.value()
        except:
            print "widget type {} is not supported yet :(".format(wt)

def disable_widget(widget):
    widget.widget.setEnabled(False)
    
def enable_widget(widget):
    widget.widget.setEnabled(True)

def display_window(main_window, app):
    main_window.show()
#    sys.exit(app.exec_())
    app.exec_()
    #have to keep somehow main_window in memory, otherwise the window 
    #will disappear (garbage colected):
    return main_window

#def process_events():
#    """
#    important to include this somewere in a main loop to process any events 
#    in gui (e.g., pressing the stop button), since Qt objects can't be 
#    passed to children threads.
#    """
#    pg.QtGui.QApplication.processEvents()
    
def resize_widget(widget, new_size = (300, 300)):
    widget.widget.setMinimumHeight(new_size[0])
    widget.widget.setMinimumWidth(new_size[1])
    
def yes_no_dialog(question = "stop?", 
                  parent = None):
#    msg = QtGui.QMessageBox(None, 'Message', question,
#                            QtGui.QMessageBox.Yes,
#                            QtGui.QMessageBox.No)
    msg = QtGui.QMessageBox(parent = parent)
    msg.setText(question)
    msg.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
    msg.show()
    reply = msg.exec_()#this doesn't seem to cause trouble here somehow...
    if reply == QtGui.QMessageBox.Yes:
        answer = True
    else:
        answer = False
    return answer
    
def file_dialog(initial_directory = "", 
                window_title = "Choose file",
                parent = None):
    """
    opens a window that allows to choose an existing file or can type the 
    name of a new file. If initial directory is left as "" it will default 
    to the current working directory (at least in linux). Returns the 
    absolute file path.
    """
    Qstr = QtGui.QFileDialog.getOpenFileName(parent, window_title,
                                             initial_directory)
    return str(Qstr)
    
class select_file_widget(QtGui.QWidget):
    """
    like the labview file path control, allows choosing a file.
    """
    def __init__(self, 
                 initial_directory = "", 
                 dialog_window_title = "Choose file",
                 parent = None):
        QtGui.QWidget.__init__(self)
        
        self.initial_directory = initial_directory
        self.dialog_window_title = dialog_window_title
        self.parent = parent
        self.file_path = ""
        
        self.file_path_control = create_widget("text")
        self.browse_button = create_widget("button",
                                           description = "browse",
                                           action = self.on_click)
        l = QtGui.QHBoxLayout(self)
        l.addWidget(self.file_path_control)
        l.addWidget(self.browse_button)
    
    def on_click(self, *args):
        file_path = file_dialog\
        (initial_directory = self.initial_directory, 
         window_title = self.dialog_window_title,
         parent = self.parent)
        if file_path:#in case user hits cancel...
            self.file_path = file_path
        set_widget_value(self.browse_button, False)
        set_widget_value(self.file_path_control,
                         self.file_path)
                         
    def setValue(self, value):
        self.file_path = value
        set_widget_value(self.file_path_control, value)
        
    def value(self):
        # return self.file_path
        return str(get_widget_value(self.file_path_control))
                         
class multiline_text_widget(QtGui.QTextEdit):
    """
    like the labview file path control, allows choosing a file.
    """
    def __init__(self):
        QtGui.QTextEdit.__init__(self)
        self.setAcceptRichText(True)
        self.setReadOnly(False)
#        self.setLineWrapMode(QtGui.QTextEdit.NoWrap)
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())
        
    def setValue(self, text):
        self.setPlainText( str(text) )

#def gui_example():
#    w, app = start_application(minimum_width = 600,
#                               minimum_height = 800,
#                               title = "gui example")
#    btn = create_labeled_widget(widget_type = "button", 
#                                description = "click to print hi",
#                                label = "hi button",
#                                action = print_hi_and_wait)
#    btn2 = create_labeled_widget(widget_type = "button", 
#                                 description = "click to write on file",
#                                 label = "write file button",
#                                 action = write_to_test_file)
#    ind1 = create_labeled_widget(widget_type = "numeric", 
#                                 label = "indicator")
#    txt = create_labeled_widget(widget_type = "text", 
#                                 label = "text box",
#                                 value = "enter text")
#    cont = create_labeled_widget(widget_type = "container", 
#                                 label = "container")
#    plotw = create_labeled_widget(widget_type = "matplotlib canvas", 
#                                  label = "plot")
#    progress = create_labeled_widget(widget_type = "progress bar", 
#                                     label = "progress")
#    place_widgets([[btn, btn2, ind1],
#                   [txt],
#                   [cont],
#                   [plotw],
#                   [progress]],
#                   w)
#
#    w = display_window(w, app)
#    
#    
#    
##    w.updateGeometry()
##    w.resize(w.sizeHint())
##    w.resize(w.sizeHint().width(), w.sizeHint().height())
#    
#    x = np.linspace(0,1)
#    y = x**2
#    plotw.widget.axes.plot(x, y)
#    change_label(plotw, "nice plot")
#    w.updateGeometry()
#    
#    set_widget_value(ind1, 8)
#    print get_widget_value(ind1)
#    QtGui.QApplication.processEvents()
##    ind1.widget.setValue(5)
##    print ind1.widget.value()
##    
##    set_widget_value(txt, "jajajaja")
##    print get_widget_value(txt)
###    txt.widget.setText("jojo")
###    print txt.widget.displayText()
##    
##    set_widget_value(btn2, True)
##    print get_widget_value(btn2)
##    
##    set_widget_value(progress, 50)
##    print get_widget_value(progress)
##    
##    disable_widget(btn)
#    
#    timer = QtCore.QTimer(app)
#    timer.timeout.connect(print_hi)
#    timer.start(1000)
#    
#    return w #have to return to prevent the garbage collector from
#    # removing the window and its variables.
#
#class gui_example2(QtGui.QMainWindow):
#    def __init__(self):
#        QtGui.QMainWindow.__init__(self)
#        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
#        self.setWindowTitle("gui example as class")
#        self.main_widget = QtGui.QWidget(self)
#        self.main_widget.setFocus()
#        self.setCentralWidget(self.main_widget)
    
### other:
#can't use threads with qt...
# instead, have to call ProcessEvents() often...
def run_on_thread(target, *args, **kwargs):
    from threading import Thread
    worker = Thread(target = target)
    #worker.setDaemon(True)
    worker.start()  

def print_hi():
    print "hi"

def print_hi_and_wait():
    import time
    time.sleep(10)
    print "hi" 

def write_to_test_file():
    fp = open("test", 'a')
    fp.write("alo alooo")
    fp.close()
    
def do_nothing():
    pass


class EmptyClass(): pass


def create_file(file_name = "name", data=""):
    dir_name = os.path.dirname(file_name)
    if not does_directory_exist(dir_name):
        os.makedirs(dir_name)
    fp = open(file_name, "w+")
    fp.write(data)
    fp.close()


def does_directory_exist(directory_name = 'folder/'):
    """
    relative folder starts at the current working directory (i.e., the
    directory where the application resides).
    """
    import os.path
    folder = os.getcwd()
#    folder = os.path.join(folder, relative_folder)
    dir_path = os.path.join(folder, directory_name)
    return os.path.isdir(dir_path)


def delete_file(file_name = "name"):
    if os.path.exists(file_name):
        os.remove(file_name)
    else:
        print "could not delete, file {} doesn't exist".format(file_name)


def create_directory_if_needed(fname="scraps/asins/test.csv"):
    dir_name = os.path.dirname(fname)
    if dir_name and not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def initialize_csv(fname='', column_labels=[]):
    create_directory_if_needed(fname)
    df = pd.DataFrame(columns=column_labels)
    df.to_csv(fname, index=False)


def append_row_to_csv(row=[], fname=''):
    srow = pd.DataFrame([row])
    with open(fname, 'a') as f:
        srow.to_csv(f, header=False, index=False)
    
    
if __name__ == "__main__":
    qApp = QtGui.QApplication([])
    aw = gui()
    aw.show()
#    sys.exit(qApp.exec_())
    qApp.exec_()
