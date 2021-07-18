import pandas as pd
from sqlalchemy import create_engine

school = pd.read_csv('data/School_Details.csv')
# print(school)

engine = create_engine('sqlite:///sql_temp.db', echo=True)
conn = engine.connect()

# sqlite_table = "school_details"
# school.to_sql(sqlite_table, conn, if_exists='fail')

print(school.head())
print(school.info())

# Get the EstablishmentName, SchoolLevel, and Website columns.
sql = """
select EstablishmentName, SchoolLevel, Website 
from school_details

"""

temp = pd.read_sql_query(sql, engine)
print(temp)


# Display the EstablishmentName and DistrictHQDistance (in Km) for schools where the DistrictHQDistance (in Km) is more than 100 Km.

sql = """
select EstablishmentName, `DistrictHQDistance (in Km)` 
from school_details
where `DistrictHQDistance (in Km)` > 100


"""

temp = pd.read_sql_query(sql, engine)
print(temp)

# Display the EstablishmentName, SchoolLevel, DistrictHQDistance (in Km) for schools where the school level is "JHS."

sql = """
select EstablishmentName, SchoolLevel, `DistrictHQDistance (in Km)` 
from school_details
where SchoolLevel = 'JHS'

"""

temp = pd.read_sql_query(sql, engine)
print(temp)

# Display the EstablishmentName, SchoolLevel, DistrictHQDistance (in Km) for schools where the name of the school contains the words "JUNIOR HIGH."

sql = """
select EstablishmentName, SchoolLevel, `DistrictHQDistance (in Km)` 
from school_details
where EstablishmentName like '%JUNIOR HIGH%'

"""

temp = pd.read_sql_query(sql, engine)
print(temp)

# Display the EstablishmentName, SchoolLevel, DistrictHQDistance (in Km) for schools where the name of the school starts with the letter "C." Sort the results alphabetically by school name and limit the results of the query to 10 rows.

sql = """
select EstablishmentName, SchoolLevel, `DistrictHQDistance (in Km)` 
from school_details
where EstablishmentName like 'C%'
order by EstablishmentName
limit 10
"""

temp = pd.read_sql_query(sql, engine)
print(temp)

# Display the names of schools in urban areas. Sort the results in reverse alphabetical order.

sql = """
select EstablishmentName
from school_details
where LocatedInRuralOrUrban = 'Urban Area'
order by EstablishmentName desc

"""

temp = pd.read_sql_query(sql, engine)
print(temp)


# Repeat the previous query, but rename the column displayed to "name".

sql = """
select EstablishmentName as Name
from school_details
where LocatedInRuralOrUrban = 'Urban Area'
order by EstablishmentName desc

"""

temp = pd.read_sql_query(sql, engine)
print(temp)