#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import pandas as pd
star_one = pd.read_csv('lc_J0247.csv')
g_band = star_one.loc[star_one['band'] == 'g']
plt.figure(figsize = (10,2))
plt.scatter(g_band['mjd_obs'], g_band['mag_psf'], color = 'green')
plt.errorbar(g_band['mjd_obs'], g_band['mag_psf'], yerr = g_band['mag_err_psf'], xerr = 0, color = 'yellow', linestyle = 'None')
plt.gca().invert_yaxis()
plt.xlabel('Julian Date(lc_J0247- gband)')
plt.ylabel('Observed Magnitiude of Brightness')
def sin(x, a, z, p, o):
    y = a*(np.sin((2*3.14)*(x/z) + p)) + o
    return y
x_cleaned = pd.notnull(g_band['mjd_obs'])
y_cleaned = pd.notnull(g_band['mag_psf'])
model = scipy.optimize.curve_fit(sin,g_band['mjd_obs'][x_cleaned], g_band['mag_psf'][y_cleaned], p0 = (0.5, 3000, 0, 50)) 
print(model)
print(model[0])
x_array = np.linspace(51000,60000,1000)
y_array = sin(x_array,*model[0])
plt.plot(x_array,y_array, color ='red')
plt.show()
 


# In[9]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import pandas as pd
star_one = pd.read_csv('lc_J0247.csv')
r_band = star_one.loc[star_one['band'] == 'r']
plt.figure(figsize = (10,2))
plt.scatter(r_band['mjd_obs'], r_band['mag_psf'], color = 'red')
plt.errorbar(r_band['mjd_obs'], r_band['mag_psf'], yerr = r_band['mag_err_psf'], xerr = 0, color = 'yellow', linestyle = 'None')
plt.gca().invert_yaxis()
plt.xlabel('Julian Date(lc_J0247- rband)')
plt.ylabel('Observed Magnitiude of Brightness')
def sin(x, a, z, p, o):
    y = a*(np.sin((2*3.14)*(x/z) + p)) + o
    return y
x_cleaned = pd.notnull(r_band['mjd_obs'])
y_cleaned = pd.notnull(r_band['mag_psf'])
model = scipy.optimize.curve_fit(sin,r_band['mjd_obs'][x_cleaned], r_band['mag_psf'][y_cleaned], p0 = (0.5, 3000, 0, 50)) 
print(model)
print(model[0])
x_array = np.linspace(51000,60000,1000)
y_array = sin(x_array,*model[0])
plt.plot(x_array,y_array, color ='red')
plt.show()
    


# In[18]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import pandas as pd
star_one = pd.read_csv('lc_J0246.csv')
g_band = star_one.loc[star_one['band'] == 'g']
plt.figure(figsize = (10,2))
plt.scatter(g_band['mjd_obs'], g_band['mag_psf'], color = 'green')
plt.errorbar(g_band['mjd_obs'], g_band['mag_psf'], yerr = g_band['mag_err_psf'], xerr = 0, color = 'yellow', linestyle = 'None')
plt.gca().invert_yaxis()
plt.xlabel('Julian Date(lc_J0246- gband)')
plt.ylabel('Observed Magnitiude of Brightness')
def sin(x, a, z, p, o):
    y = a*(np.sin((2*3.14)*(x/z) + p)) + o
    return y
x_cleaned = pd.notnull(g_band['mjd_obs'])
y_cleaned = pd.notnull(g_band['mag_psf'])
model = scipy.optimize.curve_fit(sin,g_band['mjd_obs'][x_cleaned], g_band['mag_psf'][y_cleaned], p0 = (2, 3000, 0, 22.5)) 
print(model)
print(model[0])
x_array = np.linspace(51000,60000,1000)
y_array = sin(x_array,*model[0])
plt.plot(x_array,y_array, color ='red')
plt.show()
    


# In[19]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import pandas as pd
star_one = pd.read_csv('lc_J0246.csv')
r_band = star_one.loc[star_one['band'] == 'r']
plt.figure(figsize = (10,2))
plt.scatter(r_band['mjd_obs'], r_band['mag_psf'], color = 'red')
plt.errorbar(r_band['mjd_obs'], r_band['mag_psf'], yerr = r_band['mag_err_psf'], xerr = 0, color = 'yellow', linestyle = 'None')
plt.gca().invert_yaxis()
plt.xlabel('Julian Date(lc_J0246- rband)')
plt.ylabel('Observed Magnitiude of Brightness')
def sin(x, a, z, p, o):
    y = a*(np.sin((2*3.14)*(x/z) + p)) + o
    return y
x_cleaned = pd.notnull(r_band['mjd_obs'])
y_cleaned = pd.notnull(r_band['mag_psf'])
model = scipy.optimize.curve_fit(sin,r_band['mjd_obs'][x_cleaned], r_band['mag_psf'][y_cleaned], p0 = (2, 3000, 0, 22.5)) 
print(model)
print(model[0])
x_array = np.linspace(51000,60000,1000)
y_array = sin(x_array,*model[0])
plt.plot(x_array,y_array, color ='red')
plt.show


# In[15]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import pandas as pd
star_one = pd.read_csv('lc_J0249.csv')
g_band = star_one.loc[star_one['band'] == 'g']
plt.figure(figsize = (10,2))
plt.scatter(g_band['mjd_obs'], g_band['mag_psf'], color = 'green')
plt.errorbar(g_band['mjd_obs'], g_band['mag_psf'], yerr = g_band['mag_err_psf'], xerr = 0, color = 'yellow', linestyle = 'None')
plt.gca().invert_yaxis()
plt.xlabel('Julian Date(lc_J0249- gband)')
plt.ylabel('Observed Magnitiude of Brightness')
def sin(x, a, z, p, o):
    y = a*(np.sin((2*3.14)*(x/z) + p)) + o
    return y
x_cleaned = pd.notnull(g_band['mjd_obs'])
y_cleaned = pd.notnull(g_band['mag_psf'])
model = scipy.optimize.curve_fit(sin,g_band['mjd_obs'][x_cleaned], g_band['mag_psf'][y_cleaned], p0 = (2, 3000, 0, 22.5)) 
print(model)
print(model[0])
x_array = np.linspace(51000,60000,1000)
y_array = sin(x_array,*model[0])
plt.plot(x_array,y_array, color ='red')
plt.show


# In[20]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import pandas as pd
star_one = pd.read_csv('lc_J0246.csv')
r_band = star_one.loc[star_one['band'] == 'r']
plt.figure(figsize = (10,2))
plt.scatter(r_band['mjd_obs'], r_band['mag_psf'], color = 'red')
plt.errorbar(r_band['mjd_obs'], r_band['mag_psf'], yerr = r_band['mag_err_psf'], xerr = 0, color = 'yellow', linestyle = 'None')
plt.gca().invert_yaxis()
plt.xlabel('Julian Date(lc_J0246- rband)')
plt.ylabel('Observed Magnitiude of Brightness')
def sin(x, a, z, p, o):
    y = a*(np.sin((2*3.14)*(x/z) + p)) + o
    return y
x_cleaned = pd.notnull(r_band['mjd_obs'])
y_cleaned = pd.notnull(r_band['mag_psf'])
model = scipy.optimize.curve_fit(sin,r_band['mjd_obs'][x_cleaned], r_band['mag_psf'][y_cleaned], p0 = (2, 3000, 0, 22.5)) 
print(model)
print(model[0])
x_array = np.linspace(51000,60000,1000)
y_array = sin(x_array,*model[0])
plt.plot(x_array,y_array, color ='red')
plt.show


# In[21]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import pandas as pd
star_one = pd.read_csv('lc_J0254.csv')
g_band = star_one.loc[star_one['band'] == 'g']
plt.figure(figsize = (10,2))
plt.scatter(g_band['mjd_obs'], g_band['mag_psf'], color = 'green')
plt.errorbar(g_band['mjd_obs'], g_band['mag_psf'], yerr = g_band['mag_err_psf'], xerr = 0, color = 'yellow', linestyle = 'None')
plt.gca().invert_yaxis()
plt.xlabel('Julian Date(lc_J0254- gband)')
plt.ylabel('Observed Magnitiude of Brightness')
def sin(x, a, z, p, o):
    y = a*(np.sin((2*3.14)*(x/z) + p)) + o
    return y
x_cleaned = pd.notnull(g_band['mjd_obs'])
y_cleaned = pd.notnull(g_band['mag_psf'])
model = scipy.optimize.curve_fit(sin,g_band['mjd_obs'][x_cleaned], g_band['mag_psf'][y_cleaned], p0 = (2, 3000, 0, 22.5)) 
print(model)
print(model[0])
x_array = np.linspace(51000,60000,1000)
y_array = sin(x_array,*model[0])
plt.plot(x_array,y_array, color ='red')
plt.show


# In[22]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import pandas as pd
star_one = pd.read_csv('lc_J0254.csv')
r_band = star_one.loc[star_one['band'] == 'r']
plt.figure(figsize = (10,2))
plt.scatter(r_band['mjd_obs'], r_band['mag_psf'], color = 'red')
plt.errorbar(r_band['mjd_obs'], r_band['mag_psf'], yerr = r_band['mag_err_psf'], xerr = 0, color = 'yellow', linestyle = 'None')
plt.gca().invert_yaxis()
plt.xlabel('Julian Date(lc_J0254- rband)')
plt.ylabel('Observed Magnitiude of Brightness')
def sin(x, a, z, p, o):
    y = a*(np.sin((2*3.14)*(x/z) + p)) + o
    return y
x_cleaned = pd.notnull(r_band['mjd_obs'])
y_cleaned = pd.notnull(r_band['mag_psf'])
model = scipy.optimize.curve_fit(sin,r_band['mjd_obs'][x_cleaned], r_band['mag_psf'][y_cleaned], p0 = (2, 3000, 0, 22.5)) 
print(model)
print(model[0])
x_array = np.linspace(51000,60000,1000)
y_array = sin(x_array,*model[0])
plt.plot(x_array,y_array, color ='red')
plt.show

