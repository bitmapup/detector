##!/usr/bin/env python
"""
Contains the methods that allows to build a Graphical User Interface, by which the distances will be computed defining the parameters of Initial Trajectory, Folder of Trajectories, Nº of trajectories, Initial Date, Final Date. Also, through the use of the GUI, the trajectories could be plotted on a cartography.
"""
import datetime
import numpy as np
import json
import os
import sys
import time
import webbrowser
import wx
from distances import *
from query import *
from geojson import MultiLineString,Feature,FeatureCollection

class visualization(wx.Frame):

    def __init__(self,parent,id):
	"""Constructor of the GUI class"""
        wx.Frame.__init__(self,parent,id,'Visualization',size=(970,500))
        self.panel=wx.Panel(self)

	wx.StaticText(self.panel, -1 ,"Folder",pos=(10,10))
        self.carp=wx.TextCtrl(self.panel, -1, value=" ",pos=(90,10))

        wx.StaticText(self.panel, -1 ,"Trajectory",pos=(200,10))
        self.tray=wx.TextCtrl(self.panel, -1, value=" ",pos=(280,10))

        wx.StaticText(self.panel, -1 ,"N° Trayectories",pos=(380,10))
        self.lim=wx.TextCtrl(self.panel, -1 ,pos=(490,10))

        wx.StaticText(self.panel, -1 ,"Initial Date",pos=(600,10))
        self.ini=wx.TextCtrl(self.panel, -1,pos=(690,10))

        wx.StaticText(self.panel, -1 ,"Final Date",pos=(790,10))
        self.fi=wx.TextCtrl(self.panel, -1,pos=(870,10))        
        
 	wx.StaticText(self.panel, -1,"Date Format:", pos=(600,60))
	wx.StaticText(self.panel, -1,"dd/mm/aaaa HH:MM",pos=(600,80))
	
        self.button=wx.Button(self.panel,label="Generate Map",pos=(200,50), size=(130,60))
        self.button.Bind(wx.EVT_BUTTON,self.generate_html)    

	self.button2=wx.Button(self.panel,label="Calculate",pos=(90,50), size=(70,60))
	self.button2.Bind(wx.EVT_BUTTON,self.calculates)

	self.button3=wx.Button(self.panel,label="New",pos=(890,50), size=(70,60))
	self.button3.Bind(wx.EVT_BUTTON,self.clean)

	self.style = wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL
	self.log=wx.TextCtrl(self.panel, -1,pos=(10,150), size=(950,300),style=self.style)
	
	sys.stdout=self.log

    def calculates(self,event):
	"""Executes the calculate method from distances.py file"""
	trayectory=self.tray.GetValue()
	folder=self.carp.GetValue()
	limit=self.lim.GetValue()
	results=calculate(trayectory,folder,limit)
	return resultados

    def clean(self,event):
	"""Cleans all fields in the GUI"""
	self.tray.SetValue("")
	self.lim.SetValue("")
	self.carp.SetValue("")
	self.ini.SetValue("")
	self.fi.SetValue("")
	self.log.SetValue("")

    def generate_html(self,event):
	"""Generates the html file that contains the map visualization on Google Maps"""
        trayectory=self.tray.GetValue()
        folder=self.carp.GetValue()
        initial=int(time.mktime(datetime.strptime(
            self.ini.GetValue(), "%d/%m/%Y %H:%M").timetuple()))
        final=int(time.mktime(datetime.strptime(
            self.fi.GetValue(), "%d/%m/%Y %H:%M").timetuple()))
        antenne= getAntennasLocation()
        lista=list()
        for i in listdir(folder):
            tray= "{0}/".format(folder)+i
            if tray==trayectory:
                    matriz=getMobilityTracesMatrix(tray,antenne)
                    matriz = matriz[ (final>matriz[:,0]) & (matriz[:,0]>initial) ]
                    #print matriz
                    [matriz[:,:2]/(pow(10,9),1)]+lista
            else:
                    matriz=getMobilityTracesMatrix(tray,antenne)
                    matriz = matriz[ (final>matriz[:,0]) & (matriz[:,0]>initial) ]
                    lista.append(matriz[:,:2]/(pow(10,9),1))

        a=[]
        color=["black","red","orange","yellow","green","blue","indigo","violet"]

        for i in range(0,len(lista)-1):
            lon = lista[i][:,0]
            lat = lista[i][:,1]
            cord = list()

            for j in range(0,len(lon)-1):
                cord.append([lon[j],lat[j]])

            coordenadas= [cord]
            coordenadas= MultiLineString(coordenadas)
            visual= Feature(geometry=coordenadas,properties={"color": color[i]})
            a.append(visual)

        output = FeatureCollection(a)
        json_file = open('vis.geojson','w')
        json.dump(output,json_file)

        viewer = open('viewer.html','w')
        viewer.write('<!DOCTYPE html>\n')
        viewer.write('<html>\n')
        viewer.write('<head>\n')
        viewer.write('<meta name="viewport" content="initial-scale=1.0, user-scalable=no">\n')
        viewer.write('<meta charset="utf-8">\n')
        viewer.write('<title>Simple Polylines</title>\n')
        viewer.write('<style>\n')
        viewer.write('html, body {\n')
        viewer.write('height: 100%;\n')
        viewer.write('margin: 0;\n')
        viewer.write('padding: 0;\n')
        viewer.write('}\n')
        viewer.write('#map {\n')
        viewer.write('height: 100%;\n')
        viewer.write('}\n')
        viewer.write('</style>\n')
        viewer.write('</head>\n')
        viewer.write('<body>\n')
        viewer.write('<div id="map"></div>\n')
        viewer.write('<script>\n')
        viewer.write('function initMap() {\n')
        viewer.write("var map = new google.maps.Map(document.getElementById('map'), {\n")
        viewer.write('zoom: 10,\n')
        viewer.write('center: {lat: 48, lng: 2},\n')
        viewer.write('mapTypeId: google.maps.MapTypeId.TERRAIN\n')
        viewer.write('});\n')
        viewer.write("map.data.loadGeoJson('vis.geojson');\n")
        viewer.write('map.data.setStyle(function(feature) {\n')
        viewer.write('return ({\n')
        viewer.write("strokeColor: feature.getProperty('color'),\n")
        viewer.write('strokeWeight: 1\n')
        viewer.write('});\n')
        viewer.write('});\n')
        viewer.write('}\n')
        viewer.write('</script>\n')
        viewer.write('<script async defer\n')
        viewer.write('\tsrc="https://maps.googleapis.com/maps/api/js?key=AIzaSyAKKl3Jdqe-mV9Q1gU2GoPSO9GtCIh2uE8&callback=initMap">\n')
        viewer.write('</script>\n')
        viewer.write('</body>\n')
        viewer.write('</html>\n')
        viewer.close()

        filename='file:///'+os.getcwd()+'/'+'viewer.html'

        webbrowser.open_new_tab(filename)


if __name__=='__main__':
    app=wx.App()
    frame=visualization(parent=None,id=-1)
    frame.Show(True)
    app.MainLoop()

__author__ = "Isaias Hoyos"
__copyright__ = "Copyright 2017, Detector Project"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Isaias Hoyos"
__email__ = "i.hoyoslopez@up.edu.pe"
__status__ = "Prototype"
