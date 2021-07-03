import pymssql
import pyodbc
import mysql.connector

'''   
DriverDetails:SQLOLEDB
ServerName:DESKTOP-N33MAFB
databaseName:RMHS_5MT_rakeDetails
UID:sa
Password:Password@123   
192.168.101.79
# Install pyodbc after if it is not working try second line
# pip install pyodbc
# pip install --upgrade pyodbc
if ODBC issue is happen then we ened to follow below link.
https://docs.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver15
'''

# Through this function we can insert data into ou sql database.

# To insert dat into table we need to pass the data in list format
def insert_data(Data):

    try:
        print("Try to connect SQL Server")
        # log.info("Try to connect mssql server")

        # Give sql credentials...
        conn = pyodbc.connect('DRIVER={ODBC Drive  r 17 for SQL Server};'
                             'Server=172.21.25.164;'
                             'Database=cvmlautomation2;'
                             'UID=sa;PWD=admin@123;'
                             'MARS_connection=yes')

        # conn = pymssql.connect(server='172.21.25.164', user='sa', password='admin@123', database='cvml')
        cursor = conn.cursor()
        print("Connection successful")


        if cursor.connection:
            # print("Inside SQL DB")

            # SQl query to insert data.( According to your requirement we need pass the value in list format
            sql = "INSERT INTO wagonumber_1(SLNo, WagonNumber) VALUES ('{}','{}');".format(Data[0],Data[1])

            # Checking the sql inserting data
            print(" Data Insert query {}".format(sql))

            # Code is execute
            cursor.execute(sql)
            conn.commit()

            # After execute code server will close
            cursor.close()
            conn.close()

            print("Data has been inserted")
            logger.info("Data has been insert in server-{}-{}".format(Data[0],Data[1]))

        else:

            cursor.close()
            conn.close()

    except Exception as e:
        print("SQL connect failed", e)

    # log.info("mssql server connection successful")