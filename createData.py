import duckdb
import random 

from duckdb.typing import *
from faker import Faker

temp="data/flights-3m.csv"
def random_iban(n):
    fake = Faker(int(n*10))
    return fake.iban()

#duckdb.create_function("iban", random_iban, [DOUBLE], VARCHAR)
#res = duckdb.sql("COPY (SELECT iban(random()) as iban FROM generate_series(1, 10)) TO 'data/beneficiaries1.parquet'  (FORMAT 'parquet')")
#res = duckdb.sql("COPY (SELECT iban(random()) as iban FROM generate_series(1, 10)) TO 'data/beneficiaries2.parquet'  (FORMAT 'parquet')")
#res = duckdb.sql("COPY (SELECT * FROM '"+temp+"') TO 'data/beneficiaries1.parquet'  (FORMAT 'parquet')")
#res=duckdb.sql("PRAGMA add_parquet_key('key256', '')")
#res=duckdb.sql("COPY (SELECT * FROM './data/beneficiaries4.parquet') TO './data/beneficiaries-encrypted4.parquet' (ENCRYPTION_CONFIG {footer_key: 'key256'})")
#res = duckdb.sql("INSERT INTO 'data/beneficiaries1.parquet' (SELECT * FROM '"+temp+"')")
res = duckdb.sql("SELECT * FROM read_parquet('data/beneficiaries-encrypted1.parquet', encryption_config = {footer_key: 'key256'})")
print (len(res))
