import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import pandas as pd

def light_curve(starname: str, band_filter: str):
     #starname and band_filter must be strings, with the starname being a .csv file.
    star = pd.read_csv(starname + '.csv')
    band = star_one.loc[star_one['band'] == band_filter]

    plt.figure(figsize = (10,2))
    plt.scatter(band['mjd_obs'],band['mag_psf'], color = 'green')
    plt.errorbar(band['mjd_obs'], band['mag_psf'], yerr = band['mag_err_psf'], xerr = 0, color = 'yellow', linestyle = 'None')
    plt.gca().invert_yaxis()
    plt.xlabel('Julian Date' + ' ' + starname + ' ' + band_filter + 'band')
    plt.ylabel('Observed Magnitiude of Brightness')
    def sin(x, a, z, p, o):
        y = a*(np.sin((2*3.14)*(x/z) + p)) + o
        return y
    x_cleaned = pd.notnull(band['mjd_obs'])
    y_cleaned = pd.notnull(band['mag_psf'])
    model = scipy.optimize.curve_fit(sin,band['mjd_obs'][x_cleaned], band['mag_psf'][y_cleaned], p0 = (0.5, 3000, 0, 50)) 
    print(model)
    print(model[0])
    x_array = np.linspace(51000,60000,1000)
    y_array = sin(x_array,*model[0])
    plt.plot(x_array,y_array, color ='red')
    plt.show()

