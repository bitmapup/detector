##!/usr/bin/env python
"""
Contains the methods that allows to extract the information of antennas and to represent it as Mobility Traces.
"""
import csv
import logging
from mmc.mobilitytrace import MobilityTrace
import glob
import numpy
from os import listdir
import os
from os.path import isfile, join
import collections
from datetime import datetime
from datetime import date
import numpy as np
import warnings


LABELS=["type","date","event","msisdn","imsi","imei","tac","cgi","imsiMcc","imsiMnc","imsiOc"]

def getAntennasLocation (path="cell_2014_07_29.csv"):

    labels = ["mcc","mnc","lac","ci","radius","lon","lat"]
    dic_antennas = dict()
    with open(path,'rb') as tsvin:
        logging.info("Open: {0}".format(path))
        tsvin = csv.reader(tsvin, delimiter=';')
        tsvin.next()

        for row in tsvin:
            #print (row)
            mcc = row[labels.index("mcc")]
            mnc = row[labels.index("mnc")]
            lac = row[labels.index("lac")]
            ci = row[labels.index("ci")]
            cgi = "{0}-{1}-{2}-{3}".format(mcc,mnc,lac,ci)
            lon = float(row[labels.index("lon")])/1000000
            lat = float(row[labels.index("lat")])/1000000
            latlon = [lat,lon]

            dic_antennas[cgi] = latlon

    return (dic_antennas)
#end getLanLon

def getAntennas (path="sfr-cells-juin2015.csv"):

    labels =  ["mcc","mnc","lac","ci","radius","longitude","latitude","angle_start","angle_end","dept","zip_code",";city","address","cell_kind","cuid","start_ts","end_ts"]
    dic_antennas = dict()
    with open(path,'rb') as tsvin:
        logging.info("Open: {0}".format(path))
        tsvin = csv.reader(tsvin, delimiter=';')
        tsvin.next()

        for row in tsvin:
            #print (row)
            mcc = row[labels.index("mcc")]
            mnc = row[labels.index("mnc")]
            lac = row[labels.index("lac")]
            ci = row[labels.index("ci")]
            cgi = "{0}-{1}-{2}-{3}".format(mcc,mnc,lac,ci)
            cuid = row[labels.index("cuid")]
            lon = float(row[labels.index("longitude")])/1000000
            lat = float(row[labels.index("latitude")])/1000000
            latlon = [lat,lon]
            dic_antennas[str(cuid)] = latlon
               

    return (dic_antennas)
#end getAntennas

def getMobilityTraces(inputFilePath,antenneDict):

    trailmt = list()
    labels=LABELS
    with open(inputFilePath,'rb') as tsvin:
        logging.info("Open: {0}".format(inputFilePath))
        tsvin = csv.reader(tsvin, delimiter=';')
        for row in tsvin:
            imsi = row[labels.index("imsi")]
            date = row[labels.index("date")]
            cgi = row[labels.index("cgi")]
            event = row[labels.index("event")]
            lat = 0
            lon = 0

            if cgi in antenneDict:
               lat = antenneDict[cgi][0] 
               lon = antenneDict[cgi][1] 
               
            if ((lat!=0) and (lon!=0) and (event == "CELL_ENTER")):
                aux_mt = MobilityTrace(
                        int(date),
                        cgi,#arr_id"
                        lat,
                        lon,
                        "gsm",
                        userid=imsi
                        )
                trailmt.append(aux_mt)
    return trailmt

#end getMobilityTraces():


def getMobilityTracesMatrix(inputFilePath,antenneDict):
    trailmt = list()
    labels=LABELS
    if os.stat(inputFilePath).st_size > 0:
        with open(inputFilePath,'rb') as tsvin:
            logging.info("Open: {0}".format(inputFilePath))
            tsvin = csv.reader(tsvin, delimiter=';')
            for row in tsvin:
                imsi = row[labels.index("imsi")]
                date = row[labels.index("date")]
                cgi = row[labels.index("cgi")]
                event = row[labels.index("event")]
                lat = 0
                lon = 0

                if cgi in antenneDict:
                   lat = antenneDict[cgi][0]
                   lon = antenneDict[cgi][1]
           #print lat,lon
     
                   
                if ((lat!=0) and (lon!=0) and (event == "CELL_ENTER")):
                    aux_mt = [int(date),lat,lon]
            #print aux_mt
                    trailmt.append(aux_mt)
        #print np.array(trailmt)
    else:
        warnings.warn(
            "Tried to process an empty file inside getMobilityTraceMatrix."
            ,RuntimeWarning
            )
    return  np.array(trailmt)

def getMobilityTracesDistanceModified(inputFilePath,antenneDict,distanceToMove):
    move=distanceToMove*0.009
    trailmt = list()
    labels=LABELS
    if os.stat(inputFilePath).st_size > 0:
        with open(inputFilePath,'rb') as tsvin:
            logging.info("Open: {0}".format(inputFilePath))
            tsvin = csv.reader(tsvin, delimiter=';')
            for row in tsvin:
                imsi = row[labels.index("imsi")]
                date = row[labels.index("date")]
                cgi = row[labels.index("cgi")]
                event = row[labels.index("event")]
                lat = 0
                lon = 0

                if cgi in antenneDict:
                   lat = antenneDict[cgi][0]+move
                   lon = antenneDict[cgi][1]
		   #print lat,lon
	 
                   
                if ((lat!=0) and (lon!=0) and (event == "CELL_ENTER")):
                    aux_mt = [int(date),lat,lon]
		    #aux_mt = [lat,lon]
                    trailmt.append(aux_mt)
		#print np.array(trailmt)
    else:
        warnings.warn(
            "Tried to process an empty file inside getMobilityTraceMatrix."
            ,RuntimeWarning
            )
    return  np.array(trailmt)
    #print np.array(trailmt)

#end getMobilityTracesCgi:

def getMobilityTracesToCsvMap(basePath,listUsers,outputFile):
    labels=LABELS
    antenneDict = getAntennasLocation()
    f = open(outputFile,'w')
    f.write("date,latitude,longitude,name\n")
    for oFile in listUsers:
        inputFilePath = basePath + oFile

        if os.stat(inputFilePath).st_size > 0:
            with open(inputFilePath,'rb') as tsvin:
                logging.info("Open: {0}".format(inputFilePath))
                tsvin = csv.reader(tsvin, delimiter=';')
                for row in tsvin:
                    imsi = row[labels.index("imsi")]
                    date = row[labels.index("date")]
                    cgi = row[labels.index("cgi")]
                    event = row[labels.index("event")]
                    lat = 0
                    lon = 0

                    
                    if ((cgi in antenneDict) ):
                       lat = antenneDict[cgi][0] 
                       lon = antenneDict[cgi][1] 
                   
                    if ((lat!=0) and (lon!=0) and (event == "CELL_ENTER")):
                        aux_mt = "{0},{1},{2},{3}".format(int(date),lat,lon,inputFilePath)
                        f.write(aux_mt+'\n') 
        else:
            warnings.warn(
                "Tried to process an empty file inside getMobilityTraceMatrix."
                ,RuntimeWarning
            )
    f.close()

#end  getMobilityTracesTOCsvMap


def getMobilityTracesCgi(inputFilePath,antenneDict):
    trailmt = list()
    labels=LABELS
    with open(inputFilePath,'rb') as tsvin:
        logging.info("Open: {0}".format(inputFilePath))
        tsvin = csv.reader(tsvin, delimiter=';')
        for row in tsvin:
            imsi = row[labels.index("imsi")]
            date = row[labels.index("date")]
            cgi = row[labels.index("cgi")]
            lat = 0
            lon = 0

            if cgi in antenneDict:
               lat = antenneDict[cgi][0] 
               lon = antenneDict[cgi][1] 
               
            if ((lat!=0) and (lon!=0)):
                aux_mt = cgi.split("-")
                trailmt.append(str(aux_mt[-2]+""+aux_mt[-1]))

    return  np.array(trailmt)
    #print np.array(trailmt)

#end getMobilityTracesCgi:
####

def getMobilityTracesCgiModified(inputFilePath,antenneDict,distanceToMove):
    move=distanceToMove*0.009
    trailmt = list()
    labels=LABELS
    with open(inputFilePath,'rb') as tsvin:
        logging.info("Open: {0}".format(inputFilePath))
        tsvin = csv.reader(tsvin, delimiter=';')
        for row in tsvin:
            imsi = row[labels.index("imsi")]
            date = row[labels.index("date")]
            cgi = row[labels.index("cgi")]
            lat = 0
            lon = 0

            if cgi in antenneDict:
               lat = antenneDict[cgi][0]+move 
               lon = antenneDict[cgi][1] 
               
            if ((lat!=0) and (lon!=0)):
                aux_mt = cgi.split("-")
                trailmt.append(str(aux_mt[-2]+""+aux_mt[-1]))

    return np.array(trailmt)
    #print np.array(trailmt)

def getMobilityTraces_timeInterval(inputFilePath,antenneDict,lowerLimit,upperLimit):

    trailmt = dict()
    labels=LABELS
    with open(inputFilePath,'rb') as tsvin:
        logging.info("Open: {0}".format(inputFilePath))
        tsvin = csv.reader(tsvin, delimiter=';')
        for row in tsvin:
            date = datetime.fromtimestamp(float(row[labels.index("date")]))
            day = date.weekday()
            hour = date.hour
            if ((day < 5) and (lowerLimit <= hour) and (hour <= upperLimit)):
                imsi = row[labels.index("imsi")]
                date = row[labels.index("date")]
                cgi = row[labels.index("cgi")]
                lat = 0
                lon = 0

                if cgi in antenneDict:
                   lat = antenneDict[cgi][0] 
                   lon = antenneDict[cgi][1] 
                   
                if ((lat!=0) and (lon!=0)):
                    aux_mt = MobilityTrace(
                            int(date),
                            cgi,#arr_id"
                            lat,
                            lon,
                            "gsm",
                            userid=imsi
                            )
                    if cgi in trailmt:
                        trailmt[cgi][0]+= 1
                    else:
                        trailmt[cgi] = [1,aux_mt]
    return trailmt

#end getMobilityTraces():


def buildSubscribersMmc (locationDictionary,inputFilePath,outputFolder):
    labels = ["user_id","timestamp","arr_id"]
    trailmt = dict()
    processedUsers = list()
    pDaysArray=[False,False,False,False,False,False,False,False,False,True]
    pTimeslices =  1

    minpts = 2
    eps = 3
    key = processedUsers[-1]
    oDjCluster = Djcluster(minpts,eps,trailmt[key])
    #clustering
    oDjCluster.doCluster()
    oDjCluster.post_proccessing()

    oMmc = Mmc(oDjCluster,
            trailmt[key],key,
            daysArray=pDaysArray,
            timeSlices=pTimeslices,
            radius=eps
            )
    oMmc.buildModel()
    #oMmc.export(outputFolder)
    return oMmc
#end buildSubscribersMmc


def buildHeatMap (inputFilePath,event):

    labels = ["timestamp","imei","tac","imsiMcc","imsiMnc","imsiOc","event","imsi","msisdn","cgi","class_id","is_traced"]
    labels = LABELS
    locationDictionary = getAntennasLocation ("global_mmc/cell_2014_07_29.csv")
    events_dict = dict()

    with open(inputFilePath,'rb') as tsvin:
        logging.info("Open: {0}".format(inputFilePath))
        tsvin = csv.reader(tsvin, delimiter=';')
        for row in tsvin:
            idUser = row[labels.index("imsi")]
            idCgi = row[labels.index("cgi")]
            latitude = 0.0
            longitude = 0.0
            if idCgi in locationDictionary:
                latitude = (locationDictionary[idCgi])[0]
                longitude = (locationDictionary[idCgi])[1]

            if ((latitude != 0) & (longitude != 0)):
                if (idCgi in events_dict):
                    events_dict[idCgi][2] += 1
                else:
                    events_dict[idCgi] = [latitude,longitude,1]

    print "id,count,latitude,longitude"

    for idCgi in events_dict:
        print "{0},{1},{2},{3},{4}".format(idCgi,
                events_dict[idCgi][2],
                events_dict[idCgi][0],
                events_dict[idCgi][1],
                event)


def build_traces_fusion (traceFile,antennaLocFile="/home/nunez/labs/studies/roissyexpress/route_detection/global_mmc/cell_2014_07_29.csv"):

    labels = ["timestamp","imei","tac","imsiMcc","imsiMnc","imsiOc","event","imsi","msisdn","cgi","class_id","is_traced"]
    labels = LABELS
    locationDictionary = getAntennasLocation (antennaLocFile)
    events_dict = dict()
    print "imsi,timestamp,cgi,event,latitude,longitude,day,hour,icon"

    with open(traceFile,'rb') as tsvin:
        logging.info("Open: {0}".format(traceFile))
        tsvin = csv.reader(tsvin, delimiter=';')
        for row in tsvin:
            idUser = row[labels.index("imsi")]
            idCgi = row[labels.index("cgi")]
            event = row[labels.index("event")]
            icon = "large_red"
            if event=="DETACH":
                icon = "large_yellow"
            if event=="CELL_LEAVE":
                icon = "large_green"
            if event=="LOST":
                icon = "large_blue"
            if event=="DISAPPEARED":
                icon = "large_purple"


            unixtime = float(row[labels.index("date")])#.split(":")[1]
            timestamp = datetime.fromtimestamp(unixtime)
            day = timestamp.day
            hour = int(timestamp.hour)
            label_hour = ""
            if hour < 6:
                label_hour = "0-6"
            if ((hour >= 6) & (hour < 12)):
                label_hour = "6-12"
            if ((hour >= 12) & (hour < 18)):
                label_hour = "12-18"
            if (hour >= 18):
                label_hour = "18-0"


            latitude = 0.0
            longitude = 0.0
            if idCgi in locationDictionary:
                latitude = (locationDictionary[idCgi])[0]
                longitude = (locationDictionary[idCgi])[1]

            if ((latitude != 0) & (longitude != 0)):
                if (idCgi in events_dict):
                    events_dict[idCgi][2] += 1
                else:
                    events_dict[idCgi] = [latitude,longitude,1]


            print "{0},{1},{2},{3},{4},{5},{6},{7},{8}".format(idUser,
                timestamp,
                idCgi,
                event,
                latitude,
                longitude,
                day,
                label_hour,
                icon)
#end build_traces_fusion


if __name__ == "__main__":
    locationDictionary = "cell.2014-07-29.csv"
    inputFilePath = "data/user_1004640209356311.csv"
    outputFolder = "models/"
    antennes = getAntennasLocation()
    mt = getMobilityTraces_timeInterval(inputFilePath,antennes,0,4)
    print len(mt)
    for key in mt:
        print mt[key]

__author__ = "Miguel Nuñez del Prado"
__copyright__ = "Copyright 2017, Detector Project"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Miguel Nuñez del Prado"
__email__ = "m.nunezdelpradoc@up.edu.pe"
__status__ = "Development"

