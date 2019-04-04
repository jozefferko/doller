import pyodbc
import pandas as pd
import datetime



def getData():
    conn = pyodbc.connect('Driver={SQL Server};Server=92.43.176.119;Database=Indusoft;UID=UCN-DDD01;PWD=rQdj7KmG8p;')
    

    return conn
    
def getNameAndProdID():
    getNameAndProdID = getData()
    result = getNameAndProdID.execute("SELECT PRODTABLE.NAME,JMGSTAMPTRANS.JOBREF,JMGSTAMPTRANS.CORRSTARTTIME, JMGSTAMPTRANS.CORRSTOPTIME,JMGSTAMPTRANS.QTYGOOD,JMGSTAMPTRANS.CORRSTARTDATE,JMGSTAMPTRANS.CORRSTOPDATE from JMGSTAMPTRANS inner join PRODTABLE on PRODTABLE.PRODID=JMGSTAMPTRANS.JOBREF WHERE  PRODTABLE.NAME LIKE '%ST1 :%'")
    return result
        
        
def main():
    return getNameAndProdID()

result = pd.read_sql_query("SELECT PRODTABLE.NAME,JMGSTAMPTRANS.JOBREF,JMGSTAMPTRANS.CORRSTARTTIME, JMGSTAMPTRANS.CORRSTOPTIME,JMGSTAMPTRANS.QTYGOOD,JMGSTAMPTRANS.CORRSTARTDATE,JMGSTAMPTRANS.CORRSTOPDATE from JMGSTAMPTRANS inner join PRODTABLE on PRODTABLE.PRODID=JMGSTAMPTRANS.JOBREF WHERE  PRODTABLE.NAME LIKE '%ST1 :%'",getData())

start_time = result['CORRSTARTTIME']
stop_time = result['CORRSTOPTIME']
result['Start Time'] = start_time.apply(lambda start_time: pd.to_datetime(str(datetime.timedelta(seconds=start_time))))
result['Start Time'] = pd.to_datetime(result['Start Time'],format= '%H:%M:%S' ).dt.time
result['Stop Time'] = stop_time.apply(lambda stop_time: pd.to_datetime(str(datetime.timedelta(seconds=stop_time))))
result['Stop Time'] = pd.to_datetime(result['Stop Time'],format= '%H:%M:%S' ).dt.time

start_date = result['CORRSTARTDATE']
stop_date = result['CORRSTOPDATE']
result['Start Date'] = pd.to_datetime(start_date,format= '%Y/%M/%D' ).dt.date
result['Stop Date'] = pd.to_datetime(stop_date,format= '%H:%M:%S' ).dt.date


result['StartDateTime'] = result.apply(lambda x: pd.datetime.combine(x['Start Date'], x['Start Time']), axis=1)
result['StopDateTime'] = result.apply(lambda y: pd.datetime.combine(y['Stop Date'], y['Stop Time']), axis=1)

result = pd.merge(result.iloc[:, :5], result.iloc[:, 11:], left_index=True, right_index=True)
result = pd.merge(result.iloc[:, :2], result.iloc[:, 5:], left_index=True, right_index=True)



