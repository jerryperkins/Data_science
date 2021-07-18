import pandas as pd
from sqlalchemy import create_engine

school = pd.read_csv('data/School_Details.csv')
facility = pd.read_csv('data/School_Facility_Details.csv')
# print(school)

engine = create_engine('sqlite:///sql_temp.db', echo=True)
conn = engine.connect()

# sqlite_table = "school_details"
# school.to_sql(sqlite_table, conn, if_exists='fail')
# sqlite_table2 = "school_facility_details"
# facility.to_sql(sqlite_table2, conn, if_exists='fail')

print(school.head())
print(school.info())

# What is the average BACDistance (in Km)? Rename the output column to "avg_bac_distance."

sql = """
select avg(`BACDistance (in Km)`) as `avg_bac_distance`
from school_details

"""

temp = pd.read_sql_query(sql, engine)
print(temp)

# What is the average BACDistance (in Km) by School Level? Sort the results from highest average to lowest average.

sql = """
select avg(`BACDistance (in Km)`) as `avg_bac_distance`, SchoolLevel
from school_details
group by SchoolLevel
Having `avg_bac_distance`

"""

temp = pd.read_sql_query(sql, engine)
print(temp)


# Repeat the previous query, but only display results where the average distance is at least 10 km.

sql = """
select avg(`BACDistance (in Km)`) as `avg_bac_distance`, SchoolLevel
from school_details
group by SchoolLevel
Having `avg_bac_distance` > 10

"""

temp = pd.read_sql_query(sql, engine)
print(temp)

# Join the two tables together (school_details and school_facility_details). Start by displaying all of the columns, but limit the resulting rows to just 2.


sql = """
select *
from school_details s
join school_facility_details f
    on s.EstablishmentCode = f.EstablishmentCode
limit 2    
"""

temp = pd.read_sql_query(sql, engine)
print(temp)

# Now, select just the schools where the SchoolLevel is "PS" that do not have libraries

sql = """
select s.EstablishmentName, f.IsLibraryAvailable
from school_details s
join school_facility_details f
    on s.EstablishmentCode = f.EstablishmentCode
where s.SchoolLevel = 'PS'
    AND f.IsLibraryAvailable = 'Not Available'
"""

temp = pd.read_sql_query(sql, engine)
print(temp)

# Build off the previous query, but order the results alphabetically by name.

sql = """
select s.EstablishmentName, f.IsLibraryAvailable
from school_details s
join school_facility_details f
    on s.EstablishmentCode = f.EstablishmentCode
where s.SchoolLevel = 'PS'
    AND f.IsLibraryAvailable = 'Not Available'
order by s.EstablishmentName
"""

temp = pd.read_sql_query(sql, engine)
print(temp)