
# coding: utf-8

# <img src="http://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="35%" align="right" border="0"><br>

# # Python for Finance (2nd ed.)
# 
# **Mastering Data-Driven Finance**
# 
# &copy; Dr. Yves J. Hilpisch | The Python Quants GmbH
# 
# <img src="http://hilpisch.com/images/py4fi_2nd_shadow.png" width="300px" align="left">

# # Input-Output Operations

# ## Basic I/O with Python

# ### Writing Objects to Disk

# In[1]:


from pylab import plt, mpl
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pickle  
import numpy as np
from random import gauss   


# In[3]:


a = [gauss(1.5, 2) for i in range(1000000)]  


# In[4]:


# adjust to path of your choice
path = '/Users/yves/Temp/data/'  


# In[5]:


pkl_file = open(path + 'data.pkl', 'wb')  


# In[6]:


get_ipython().run_line_magic('time', 'pickle.dump(a, pkl_file)')


# In[7]:


pkl_file.close()  


# In[8]:


ll $path*  


# In[9]:


pkl_file = open(path + 'data.pkl', 'rb')  


# In[10]:


get_ipython().run_line_magic('time', 'b = pickle.load(pkl_file)')


# In[11]:


a[:3]


# In[12]:


b[:3]


# In[13]:


np.allclose(np.array(a), np.array(b))  


# In[14]:


pkl_file = open(path + 'data.pkl', 'wb')


# In[15]:


get_ipython().run_line_magic('time', 'pickle.dump(np.array(a), pkl_file)')


# In[16]:


get_ipython().run_line_magic('time', 'pickle.dump(np.array(a) ** 2, pkl_file)')


# In[17]:


pkl_file.close()


# In[18]:


ll $path*  


# In[19]:


pkl_file = open(path + 'data.pkl', 'rb')


# In[20]:


x = pickle.load(pkl_file)  
x[:4]


# In[21]:


y = pickle.load(pkl_file)  
y[:4]


# In[22]:


pkl_file.close()


# In[23]:


pkl_file = open(path + 'data.pkl', 'wb')
pickle.dump({'x': x, 'y': y}, pkl_file)  
pkl_file.close()


# In[24]:


pkl_file = open(path + 'data.pkl', 'rb')
data = pickle.load(pkl_file)  
pkl_file.close()
for key in data.keys():
    print(key, data[key][:4])


# In[25]:


get_ipython().system('rm -f $path*')


# ### Reading and Writing Text Files

# In[26]:


import pandas as pd


# In[27]:


rows = 5000  
a = np.random.standard_normal((rows, 5)).round(4)  


# In[28]:


a  


# In[29]:


t = pd.date_range(start='2019/1/1', periods=rows, freq='H')  


# In[30]:


t  


# In[31]:


csv_file = open(path + 'data.csv', 'w')  


# In[32]:


header = 'date,no1,no2,no3,no4,no5\n'  


# In[33]:


csv_file.write(header)  


# In[34]:


for t_, (no1, no2, no3, no4, no5) in zip(t, a):  
    s = '{},{},{},{},{},{}\n'.format(t_, no1, no2, no3, no4, no5)  
    csv_file.write(s)  


# In[35]:


csv_file.close()


# In[36]:


ll $path*


# In[37]:


csv_file = open(path + 'data.csv', 'r')  


# In[38]:


for i in range(5):
    print(csv_file.readline(), end='')  


# In[39]:


csv_file.close()


# In[40]:


csv_file = open(path + 'data.csv', 'r')  


# In[41]:


content = csv_file.readlines()  


# In[42]:


content[:5]  


# In[43]:


csv_file.close()


# In[44]:


import csv


# In[45]:


with open(path + 'data.csv', 'r') as f:
    csv_reader = csv.reader(f)  
    lines = [line for line in csv_reader]


# In[46]:


lines[:5]  


# In[47]:


with open(path + 'data.csv', 'r') as f:
    csv_reader = csv.DictReader(f)  
    lines = [line for line in csv_reader]


# In[48]:


lines[:3]  


# In[49]:


get_ipython().system('rm -f $path*')


# ### SQL Databases

# In[50]:


import sqlite3 as sq3


# In[51]:


con = sq3.connect(path + 'numbs.db')  


# In[52]:


query = 'CREATE TABLE numbs (Date date, No1 real, No2 real)'  


# In[53]:


con.execute(query)  


# In[54]:


con.commit()  


# In[55]:


q = con.execute  


# In[56]:


q('SELECT * FROM sqlite_master').fetchall()  


# In[57]:


import datetime


# In[58]:


now = datetime.datetime.now()
q('INSERT INTO numbs VALUES(?, ?, ?)', (now, 0.12, 7.3))  


# In[59]:


np.random.seed(100)


# In[60]:


data = np.random.standard_normal((10000, 2)).round(4)  


# In[61]:


get_ipython().run_cell_magic('time', '', "for row in data:  \n    now = datetime.datetime.now()\n    q('INSERT INTO numbs VALUES(?, ?, ?)', (now, row[0], row[1]))\ncon.commit()")


# In[62]:


q('SELECT * FROM numbs').fetchmany(4)  


# In[63]:


q('SELECT * FROM numbs WHERE no1 > 0.5').fetchmany(4)  


# In[64]:


pointer = q('SELECT * FROM numbs')  


# In[65]:


for i in range(3):
    print(pointer.fetchone())  


# In[66]:


rows = pointer.fetchall()  
rows[:3]


# In[67]:


q('DROP TABLE IF EXISTS numbs')  


# In[68]:


q('SELECT * FROM sqlite_master').fetchall()  


# In[69]:


con.close()  


# In[70]:


get_ipython().system('rm -f $path*  ')


# ### Writing and Reading Numpy Arrays

# In[71]:


dtimes = np.arange('2019-01-01 10:00:00', '2025-12-31 22:00:00',
                  dtype='datetime64[m]')  


# In[72]:


len(dtimes)


# In[73]:


dty = np.dtype([('Date', 'datetime64[m]'),
                ('No1', 'f'), ('No2', 'f')])  


# In[74]:


data = np.zeros(len(dtimes), dtype=dty)  


# In[75]:


data['Date'] = dtimes  


# In[76]:


a = np.random.standard_normal((len(dtimes), 2)).round(4)  


# In[77]:


data['No1'] = a[:, 0]  
data['No2'] = a[:, 1]  


# In[78]:


data.nbytes  


# In[79]:


get_ipython().run_line_magic('time', "np.save(path + 'array', data)")


# In[80]:


ll $path*  


# In[81]:


get_ipython().run_line_magic('time', "np.load(path + 'array.npy')")


# In[82]:


get_ipython().run_line_magic('time', 'data = np.random.standard_normal((10000, 6000)).round(4)')


# In[83]:


data.nbytes  


# In[84]:


get_ipython().run_line_magic('time', "np.save(path + 'array', data)")


# In[85]:


ll $path*   


# In[86]:


get_ipython().run_line_magic('time', "np.load(path + 'array.npy')")


# In[87]:


get_ipython().system('rm -f $path*')


# ## I/O with pandas

# In[88]:


data = np.random.standard_normal((1000000, 5)).round(4)


# In[89]:


data[:3]


# ### SQL Database

# In[90]:


filename = path + 'numbers'


# In[91]:


con = sq3.Connection(filename + '.db')


# In[92]:


query = 'CREATE TABLE numbers (No1 real, No2 real,        No3 real, No4 real, No5 real)'  


# In[93]:


q = con.execute
qm = con.executemany


# In[94]:


q(query)


# In[95]:


get_ipython().run_cell_magic('time', '', "qm('INSERT INTO numbers VALUES (?, ?, ?, ?, ?)', data)  \ncon.commit()")


# In[96]:


ll $path*


# In[97]:


get_ipython().run_cell_magic('time', '', "temp = q('SELECT * FROM numbers').fetchall()  \nprint(temp[:3])")


# In[98]:


get_ipython().run_cell_magic('time', '', "query = 'SELECT * FROM numbers WHERE No1 > 0 AND No2 < 0'\nres = np.array(q(query).fetchall()).round(3)  ")


# In[99]:


res = res[::100]  
plt.figure(figsize=(10, 6))
plt.plot(res[:, 0], res[:, 1], 'ro')  
# plt.savefig('../../images/ch09/io_01.png');


# ### From SQL to pandas

# In[100]:


get_ipython().run_line_magic('time', "data = pd.read_sql('SELECT * FROM numbers', con)")


# In[101]:


data.head()


# In[102]:


get_ipython().run_line_magic('time', "data[(data['No1'] > 0) & (data['No2'] < 0)].head()")


# In[103]:


get_ipython().run_cell_magic('time', '', "q = '(No1 < -0.5 | No1 > 0.5) & (No2 < -1 | No2 > 1)'  \nres = data[['No1', 'No2']].query(q)  ")


# In[104]:


plt.figure(figsize=(10, 6))
plt.plot(res['No1'], res['No2'], 'ro');
# plt.savefig('../../images/ch09/io_02.png');


# In[105]:


h5s = pd.HDFStore(filename + '.h5s', 'w')  


# In[106]:


get_ipython().run_line_magic('time', "h5s['data'] = data")


# In[107]:


h5s  


# In[108]:


h5s.close()  


# In[109]:


get_ipython().run_cell_magic('time', '', "h5s = pd.HDFStore(filename + '.h5s', 'r')  \ndata_ = h5s['data']  \nh5s.close()  ")


# In[110]:


data_ is data  


# In[111]:


(data_ == data).all()  


# In[112]:


np.allclose(data_, data)  


# In[113]:


ll $path*  


# ### Data as CSV File

# In[114]:


get_ipython().run_line_magic('time', "data.to_csv(filename + '.csv')")


# In[115]:


ll $path


# In[116]:


get_ipython().run_line_magic('time', "df = pd.read_csv(filename + '.csv')")


# In[117]:


df[['No1', 'No2', 'No3', 'No4']].hist(bins=20, figsize=(10, 6));
# plt.savefig('../../images/ch09/io_03.png');


# ### Data as Excel File

# In[118]:


get_ipython().run_line_magic('time', "data[:100000].to_excel(filename + '.xlsx')")


# In[119]:


get_ipython().run_line_magic('time', "df = pd.read_excel(filename + '.xlsx', 'Sheet1')")


# In[120]:


df.cumsum().plot(figsize=(10, 6));
# plt.savefig('../../images/ch09/io_04.png');


# In[121]:


ll $path*


# In[122]:


rm -f $path*


# ## Fast I/O with PyTables

# In[123]:


import tables as tb  
import datetime as dt


# ### Working with Tables

# In[124]:


filename = path + 'pytab.h5'


# In[125]:


h5 = tb.open_file(filename, 'w')  


# In[126]:


row_des = {
    'Date': tb.StringCol(26, pos=1),  
    'No1': tb.IntCol(pos=2),  
    'No2': tb.IntCol(pos=3),  
    'No3': tb.Float64Col(pos=4),  
    'No4': tb.Float64Col(pos=5)  
    }


# In[127]:


rows = 2000000


# In[128]:


filters = tb.Filters(complevel=0)  


# In[129]:


tab = h5.create_table('/', 'ints_floats',  
                      row_des,  
                      title='Integers and Floats',  
                      expectedrows=rows,  
                      filters=filters)  


# In[130]:


type(tab)


# In[131]:


tab


# In[132]:


pointer = tab.row  


# In[133]:


ran_int = np.random.randint(0, 10000, size=(rows, 2))  


# In[134]:


ran_flo = np.random.standard_normal((rows, 2)).round(4)  


# In[135]:


get_ipython().run_cell_magic('time', '', "for i in range(rows):\n    pointer['Date'] = dt.datetime.now()  \n    pointer['No1'] = ran_int[i, 0]  \n    pointer['No2'] = ran_int[i, 1]  \n    pointer['No3'] = ran_flo[i, 0]  \n    pointer['No4'] = ran_flo[i, 1]  \n    pointer.append()  \ntab.flush()  ")


# In[136]:


tab  


# In[137]:


ll $path*


# In[138]:


dty = np.dtype([('Date', 'S26'), ('No1', '<i4'), ('No2', '<i4'),
                                 ('No3', '<f8'), ('No4', '<f8')])  


# In[139]:


sarray = np.zeros(len(ran_int), dtype=dty)  


# In[140]:


sarray[:4]  


# In[141]:


get_ipython().run_cell_magic('time', '', "sarray['Date'] = dt.datetime.now()  \nsarray['No1'] = ran_int[:, 0]  \nsarray['No2'] = ran_int[:, 1]  \nsarray['No3'] = ran_flo[:, 0]  \nsarray['No4'] = ran_flo[:, 1]  ")


# In[142]:


get_ipython().run_cell_magic('time', '', "h5.create_table('/', 'ints_floats_from_array', sarray,\n                      title='Integers and Floats',\n                      expectedrows=rows, filters=filters)  ")


# In[143]:


type(h5)


# In[144]:


h5  


# In[145]:


h5.remove_node('/', 'ints_floats_from_array')  


# In[146]:


tab[:3]  


# In[147]:


tab[:4]['No4']  


# In[148]:


get_ipython().run_line_magic('time', "np.sum(tab[:]['No3'])")


# In[149]:


get_ipython().run_line_magic('time', "np.sum(np.sqrt(tab[:]['No1']))")


# In[150]:


get_ipython().run_cell_magic('time', '', "plt.figure(figsize=(10, 6))\nplt.hist(tab[:]['No3'], bins=30);  \n# plt.savefig('../../images/ch09/io_05.png');")


# In[151]:


query = '((No3 < -0.5) | (No3 > 0.5)) & ((No4 < -1) | (No4 > 1))'  


# In[152]:


iterator = tab.where(query)  


# In[153]:


get_ipython().run_line_magic('time', "res = [(row['No3'], row['No4']) for row in iterator]")


# In[154]:


res = np.array(res)  
res[:3]


# In[155]:


plt.figure(figsize=(10, 6))
plt.plot(res.T[0], res.T[1], 'ro');
# plt.savefig('../../images/ch09/io_06.png');


# In[156]:


get_ipython().run_cell_magic('time', '', "values = tab[:]['No3']\nprint('Max %18.3f' % values.max())\nprint('Ave %18.3f' % values.mean())\nprint('Min %18.3f' % values.min())\nprint('Std %18.3f' % values.std())")


# In[157]:


get_ipython().run_cell_magic('time', '', "res = [(row['No1'], row['No2']) for row in\n        tab.where('((No1 > 9800) | (No1 < 200)) \\\n                & ((No2 > 4500) & (No2 < 5500))')]")


# In[158]:


for r in res[:4]:
    print(r)


# In[159]:


get_ipython().run_cell_magic('time', '', "res = [(row['No1'], row['No2']) for row in\n        tab.where('(No1 == 1234) & (No2 > 9776)')]")


# In[160]:


for r in res:
    print(r)


# ### Working with Compressed Tables

# In[161]:


filename = path + 'pytabc.h5'


# In[162]:


h5c = tb.open_file(filename, 'w') 


# In[163]:


filters = tb.Filters(complevel=5,  
                     complib='blosc')  


# In[164]:


tabc = h5c.create_table('/', 'ints_floats', sarray,
                        title='Integers and Floats',
                        expectedrows=rows, filters=filters)


# In[165]:


query = '((No3 < -0.5) | (No3 > 0.5)) & ((No4 < -1) | (No4 > 1))'


# In[166]:


iteratorc = tabc.where(query)  


# In[167]:


get_ipython().run_line_magic('time', "res = [(row['No3'], row['No4']) for row in iteratorc]")


# In[168]:


res = np.array(res)
res[:3]


# In[169]:


get_ipython().run_line_magic('time', 'arr_non = tab.read()')


# In[170]:


tab.size_on_disk


# In[171]:


arr_non.nbytes


# In[172]:


get_ipython().run_line_magic('time', 'arr_com = tabc.read()')


# In[173]:


tabc.size_on_disk


# In[174]:


arr_com.nbytes


# In[175]:


ll $path*  


# In[176]:


h5c.close()  


# ### Working with Arrays

# In[177]:


get_ipython().run_cell_magic('time', '', "arr_int = h5.create_array('/', 'integers', ran_int)  \narr_flo = h5.create_array('/', 'floats', ran_flo)  ")


# In[178]:


h5  


# In[179]:


ll $path*


# In[180]:


h5.close()


# In[181]:


get_ipython().system('rm -f $path*')


# ### Out-of-Memory Computations

# In[182]:


filename = path + 'earray.h5'


# In[183]:


h5 = tb.open_file(filename, 'w') 


# In[184]:


n = 500  


# In[185]:


ear = h5.create_earray('/', 'ear',  
                      atom=tb.Float64Atom(),  
                      shape=(0, n))  


# In[186]:


type(ear)


# In[187]:


rand = np.random.standard_normal((n, n))  
rand[:4, :4]


# In[188]:


get_ipython().run_cell_magic('time', '', 'for _ in range(750):\n    ear.append(rand)  \near.flush()')


# In[189]:


ear


# In[190]:


ear.size_on_disk


# In[191]:


out = h5.create_earray('/', 'out',
                      atom=tb.Float64Atom(),
                      shape=(0, n))


# In[192]:


out.size_on_disk


# In[193]:


expr = tb.Expr('3 * sin(ear) + sqrt(abs(ear))')  


# In[194]:


expr.set_output(out, append_mode=True)  


# In[195]:


get_ipython().run_line_magic('time', 'expr.eval()')


# In[196]:


out.size_on_disk


# In[197]:


out[0, :10]


# In[198]:


get_ipython().run_line_magic('time', 'out_ = out.read()')


# In[199]:


out_[0, :10]


# In[200]:


import numexpr as ne  


# In[201]:


expr = '3 * sin(out_) + sqrt(abs(out_))'  


# In[202]:


ne.set_num_threads(1)  


# In[203]:


get_ipython().run_line_magic('time', 'ne.evaluate(expr)[0, :10]')


# In[204]:


ne.set_num_threads(4)  


# In[205]:


get_ipython().run_line_magic('time', 'ne.evaluate(expr)[0, :10]')


# In[206]:


h5.close()


# In[207]:


get_ipython().system('rm -f $path*')


# ## TsTables

# ### Sample Data

# In[208]:


no = 5000000  
co = 3  
interval = 1. / (12 * 30 * 24 * 60)  
vol = 0.2  


# In[209]:


get_ipython().run_cell_magic('time', '', 'rn = np.random.standard_normal((no, co))  \nrn[0] = 0.0  \npaths = 100 * np.exp(np.cumsum(-0.5 * vol ** 2 * interval +\n        vol * np.sqrt(interval) * rn, axis=0))  \npaths[0] = 100  ')


# In[210]:


dr = pd.date_range('2019-1-1', periods=no, freq='1s')


# In[211]:


dr[-6:]


# In[212]:


df = pd.DataFrame(paths, index=dr, columns=['ts1', 'ts2', 'ts3'])


# In[213]:


df.info()


# In[214]:


df.head()


# In[215]:


df[::100000].plot(figsize=(10, 6));
# plt.savefig('../../images/ch09/io_07.png')


# ### Data Storage

# In[216]:


import tstables as tstab


# In[217]:


class ts_desc(tb.IsDescription):
    timestamp = tb.Int64Col(pos=0)  
    ts1 = tb.Float64Col(pos=1)  
    ts2 = tb.Float64Col(pos=2)  
    ts3 = tb.Float64Col(pos=3)  


# In[218]:


h5 = tb.open_file(path + 'tstab.h5', 'w')  


# In[219]:


ts = h5.create_ts('/', 'ts', ts_desc)  


# In[220]:


get_ipython().run_line_magic('time', 'ts.append(df)')


# In[221]:


type(ts)


# In[222]:


ls -n $path


# In[223]:


read_start_dt = dt.datetime(2019, 2, 1, 0, 0)  
read_end_dt = dt.datetime(2019, 2, 5, 23, 59)  #<2>


# In[224]:


get_ipython().run_line_magic('time', 'rows = ts.read_range(read_start_dt, read_end_dt)')


# In[225]:


rows.info()  


# In[226]:


rows.head()  


# In[227]:


h5.close()


# In[228]:


(rows[::500] / rows.iloc[0]).plot(figsize=(10, 6));
# plt.savefig('../../images/ch09/io_08.png')


# In[229]:


import random


# In[230]:


h5 = tb.open_file(path + 'tstab.h5', 'r')


# In[231]:


ts = h5.root.ts._f_get_timeseries()  


# In[232]:


get_ipython().run_cell_magic('time', '', 'for _ in range(100):  \n    d = random.randint(1, 24)  \n    read_start_dt = dt.datetime(2019, 2, d, 0, 0, 0)\n    read_end_dt = dt.datetime(2019, 2, d + 3, 23, 59, 59)\n    rows = ts.read_range(read_start_dt, read_end_dt)')


# In[233]:


rows.info()  


# In[234]:


get_ipython().system('rm $path/tstab.h5')


# <img src="http://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="35%" align="right" border="0"><br>
# 
# <a href="http://tpq.io" target="_blank">http://tpq.io</a> | <a href="http://twitter.com/dyjh" target="_blank">@dyjh</a> | <a href="mailto:training@tpq.io">training@tpq.io</a>
