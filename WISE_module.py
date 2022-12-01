# Many imports, should check they are all necessary

import numpy as np
import math
import os
from astropy import stats
from astropy import units as u
import astropy.time
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import (MinMaxInterval, SqrtStretch,
                                   ImageNormalize,LogStretch)
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.coordinates import SkyCoord
import dateutil.parser
import scipy
from scipy import stats
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import unumpy as unp
from uncertainties import ufloat_fromstr
import pandas as pd
import pyphot
import pyphot.ezunits.pint as pint
import scipy.constants as const
import extinction #https://extinction.readthedocs.io/en/latest/
from extinction import ccm89, remove, apply
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from cycler import cycler
from astroquery.skyview import SkyView
from astropy.modeling.models import BlackBody
from astropy.modeling import models, fitting
import emcee
import corner
from scipy.optimize import curve_fit
from lmfit import Model

# Various helping functions
# Check they are all documented

def mjday(day):
    "Convert observation date such as 20171131 into Julian day"
    return (astropy.time.Time(dateutil.parser.parse(day)).jd - 2400000.5,day)

def fluxdens_to_mag(flux, f0):
    "Convert flux density to magnitude"
    return -2.5 * unp.log10(flux / f0)

def mag_to_fluxdens(mag, f0):
    "Convert magnitude to flux density"
    return f0 * 10**(-mag/2.5)

def flambda(flux_jy, wavelength_micron):
    # Convert flux in jy to flux_lambda (cgs)
    # F(lambda = (3 x 10^18 F(nu)) / lambda^2), where lambda is in angstroms and F(nu) is in erg/(cm2 sec Hz)
    # from https://www.stsci.edu/itt/APT_help20/P2PI_ENG/ch03_targ_fixed11.html
    return flux_jy * 3e-5/((wavelength_micron*1e4)**2)

def lamflam(flux_jy, wavelength_micron, distance_mpc):
    # Luminosity in erg/s
    return flambda(flux_jy, wavelength_micron) * 4*np.pi*(distance_mpc*3.086e24)**2 * wavelength_micron*1e4

def mag_unc_to_flux_unc(mag_err) :
    "Convert mag error to flux error"
    return (10**(0.4*mag_err)) -1

def flux_unc_to_mag_unc(flux_err) :
    "Convert flux error to mag error"
    return np.log10(flux_err + 1) * 2.5

def weighted_mean(array):
    """
    Input array should be Nx2 measurements and errs.
    Taken from Sokolovsky, 2017
    """
    array = list(zip(*array))
    #print(array[0])
    
    x = np.nansum(np.array([array[0][i]/(array[1][i]**2) for i in range(0, len(array[0]))]))
    y = np.nansum(np.array([1/array[1][i]**2 for i in range(0, len(array[0]))]))
    weighted_mean =  x/y
    
    a = y**2 - np.nansum(np.array([1/array[1][i]**4 for i in range(0, len(array[0]))]))
    b = np.nansum(np.array([((array[0][i]-weighted_mean)**2)/(array[1][i]**2) for i in range(0, len(array[0]))]))
    weighted_sig = np.sqrt((y/a)*b)
    #print(x,y,a,b)
    return(weighted_mean,weighted_sig)

def SEM(array):
    """
    stat function for errors. Sigma clip with 5 sigma to remove big outliers
    """
    test = [np.isnan(i) for i in array]
    if np.all(test) == True:
        return np.nan
    clipped_array = astropy.stats.sigma_clip(array,sigma=3,masked=False)
    if len(array) < 3 :
        return(np.std(clipped_array))
    return np.std(clipped_array) / np.sqrt(len(clipped_array))

def bins_W3_W4(data,filt,bins):
    """filt: w1mag,w2mag,w1flux,w2flux"""
    filts = {'w3mag':'w3sig',"w4mag":"w4sig","w3flux":"w3fluxsig","w4flux":"w4fluxsig"}
    dates,bin_edges,binnumber = scipy.stats.binned_statistic(data['mjd'].values,data['mjd'].values,\
                statistic=np.mean, bins=bins, range=None)
    median_mags = []; SEM_errs = []
    for n in range(1,np.max(binnumber)+1):  # Iterate over each bin
        pairs = zip(data[filt].values,data[filts[filt]].values,binnumber)
        in_bin_mags = [x for x,y,nbin in pairs if nbin == n]
        pairs = zip(data[filt].values,data[filts[filt]].values,binnumber)
        in_bin_errs = [y for x,y,nbin in pairs if nbin == n]
        if in_bin_mags == []:
            #print("no_data")
            median_mags.append(np.nan)
            SEM_errs.append(np.nan)
            continue
        median_mags.append(np.median(in_bin_mags))
        SEM_errs.append(SEM(in_bin_errs ))
        #print(weighted_mags)
    return np.array(median_mags), np.array(SEM_errs)


# Now the big class
# WISE data class, has attributes: pandas table with info, galaxy?
# methods: Position and mag diagnostics


class WISE_Data:
    """
    A class used to contain, process and analyse WISE data tables (i.e. .tbl files)

    Attributes
    ----------
    source : str
        Name of source
    datatable : pandas table
        The original input data from .tbl file, in a pandas table
    data : pandas table
        Output from filter_data method, data will be filtered and some new columns are added
    binned_data : pandas table
        Output from bin_data method, binned data, required for plotting and output (for now)
    allowed_sep : float
        The seperation used by the filter_data method to determine good/bad data
    binned : str
        A flag that indicates if the bin_data method has been run
    filtered : str
        A flag that indicates if the filter_data method has been run
    f0_wise_3_4/self.f0_wise_4_6  :float 
        zp values for these bands, determined from pyphot
    baddata : str
        A flag that switchs on if filtering removes all data
                
    """
    def __init__(self, file, file_2 = None, pos=None,source='galname?', allowed_sep=1,dist=None,WISE_name=""):
        """
        File parameter is IRSA data table
        If you want more than one file, set file_2 = filename
        """
        #process NEOWISE data
        neowise_header = []
        for idx,line in enumerate(open(file).readlines()):
            if line.startswith('|'):
                skiprows = idx + 4 # skip over comments and header info
                for i in line.split('|')[1:-1]:
                    neowise_header.append(i.lstrip().rstrip())
                break
        # Read in data in Pandas dataframe
        neowise_read_df = pd.read_fwf(file, skiprows = skiprows, header = None, names = neowise_header)
        ra_median = np.median(neowise_read_df['ra'])
        dec_median = np.median(neowise_read_df['dec'])
        if pos == None :
            pos_median = SkyCoord(ra_median*u.deg,dec_median*u.deg, frame='icrs')
        else :
            pos_median =SkyCoord(pos[0]*u.deg,pos[1]*u.deg, frame='icrs')
        neowise_read_df['sep'] = pos_median.separation(
            SkyCoord(neowise_read_df['ra']*u.deg,neowise_read_df['dec']*u.deg,frame='icrs')).arcsec
        self.NEOWISE_datatable = neowise_read_df
        #process ALLWISE data
        if file_2 != None :
            neowise_header = []
            for idx,line in enumerate(open(file_2).readlines()):
                if line.startswith('|'):
                    skiprows = idx + 4 # skip over comments and header info
                    for i in line.split('|')[1:-1]:
                        neowise_header.append(i.lstrip().rstrip())
                    break
            # Read in data in Pandas dataframe
            ALLWISE_read_df = pd.read_fwf(file_2, skiprows = skiprows, header = None, names = neowise_header)
            # ALLWISE does not have these flags, so force passing the check
            ALLWISE_read_df['ph_qual'] = '-'
            ALLWISE_read_df['qual_frame'] = 1
            #df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
            ALLWISE_read_df = ALLWISE_read_df.rename(
                columns={'w1mpro_ep': 'w1mpro', 'w2mpro_ep': 'w2mpro','w1sigmpro_ep': 'w1sigmpro',
                         'w2sigmpro_ep': 'w2sigmpro','w3mpro_ep': 'w3mpro', 'w4mpro_ep': 'w4mpro',
                         'w3sigmpro_ep': 'w3sigmpro','w4sigmpro_ep': 'w4sigmpro'})
            ra_median = np.median(ALLWISE_read_df['ra'])
            dec_median = np.median(ALLWISE_read_df['dec'])
            if pos == None :
                pos_median = SkyCoord(ra_median*u.deg,dec_median*u.deg, frame='icrs')
            else :
                pos_median =SkyCoord(pos[0]*u.deg,pos[1]*u.deg, frame='icrs')
            ALLWISE_read_df['sep'] = pos_median.separation(
                SkyCoord(ALLWISE_read_df['ra']*u.deg,ALLWISE_read_df['dec']*u.deg,frame='icrs')).arcsec
            self.ALLWISE_datatable = ALLWISE_read_df
            neowise_read_df = neowise_read_df.append(ALLWISE_read_df)
        else :
            self.ALLWISE_datatable = pd.DataFrame()

        self.datatable = neowise_read_df
        self.allowed_sep = allowed_sep
        self.filtered = 'no'
        self.binned = 'no'
        lib = pyphot.get_library()
        self.f0_wise_3_4 = lib['WISE_RSR_W1'].Vega_zero_Jy.magnitude#*1e8
        self.f0_wise_4_6 = lib['WISE_RSR_W2'].Vega_zero_Jy.magnitude#*1e8
        self.f0_W3 = lib['WISE_RSR_W3'].Vega_zero_Jy.magnitude#*1e8
        self.f0_W4 = lib['WISE_RSR_W4'].Vega_zero_Jy.magnitude#*1e8
        self.source = source
        self.WISE_name= WISE_name
        if pos == None: #If the user gave no coords for the query, use the median of the located source
            pos = [pos_median.ra.value,pos_median.dec.value]
        if dist != None :
            self.dist = dist
        self.pos = pos
        self.baddata ='unk'
        # The below is used for some BB fit plotting later
        self.simple_BB = None
        #self.author = "Erik Kool and Tom Reynolds"
        
    def position_diag(self,cut=None,contrast =[-200,500],overlay='yes',set_source=None,rad=10,
                     diff = 2,download='yes',save='no'):
    
        """
        
        NB: Diagnostic only, no processing is done here
        
        Makes a plot showing the position of all data in the table
        To make a cropped/zoomed image, cut=[[x1,x2][y1,y2]]
        contrast = (lower,upper) to change scaling
        
        # Diff: radius of image in ARCSEC
        
        """
        #convert diff to arcminutes

#         def round_to_int(x):
#             return int(np.round(x,decimals=0))
#         diff = diff * 0.00166667        

        if set_source == None :
            gal_images = glob.glob('./Data/gal_images/*.fits')   
            im_found = 'no'
            for filename in gal_images :
                gal_name = filename.split('_')[-3].split('/')[1]
                if gal_name == self.source :
                    source_image= filename
                    im_found = 'yes'
                    set_source = 'NOTCam'
                elif self.source.split('_')[-1] in ['1','2','N','E','W','S']:
                    if gal_name == self.source.split('_')[-2]:
                        source_image= filename
                        im_found = 'yes'  
                    set_source = 'NOTCam'
            if im_found == 'no' and download == 'yes':
                set_source ='W1'
                print("gal_name mismatch:", self.source, ' , will use WISE W1')
            elif im_found == 'no':
                set_source='skip'
                print("gal_name mismatch:", self.source, " , won't show image")
                
        else :
            source_image = set_source         
        #print(source_image)
        
        if set_source != 'skip' :
            fail = 'no'
            pos = SkyCoord(self.pos[0]*u.deg,self.pos[1]*u.deg, frame='icrs')
            
            if set_source == '2MASS' :
                image_2MASS = SkyView.get_images(pos,survey=['2MASS-K'],radius=diff*u.arcsec)
                if image_2MASS == []:
                    fail = 'yes'
                else :
                    wcs = WCS(image_2MASS[0][0].header)
                    image_data = image_2MASS[0][0].data    
            if set_source == 'W1' or fail == 'yes' :
                image_W1 = SkyView.get_images(pos,survey=['WISE 3.4'],radius=diff*u.arcsec)            
                wcs = WCS(image_W1[0][0].header)
                image_data = image_W1[0][0].data 
            elif set_source == 'W2' :
                image_W2 = SkyView.get_images(pos,survey=['WISE 4.6'],radius=diff*u.arcsec)
                wcs = WCS(image_W2[0][0].header)
                image_data = image_W2[0][0].data  
            elif set_source != '2MASS' :
                hdu = fits.open(source_image)
                wcs = WCS(hdu[0].header)
                image_data = hdu[0].data
                
            #---
            size = u.Quantity(diff, u.arcsec)
            cutout = Cutout2D(image_data, pos, size, wcs=wcs)
            image_data = cutout.data; wcs = cutout.wcs
              

            fig=plt.figure(figsize=(10,10))
            ax = fig.add_subplot(1, 1, 1, projection=wcs)
            if set_source in ['2MASS'] :
                ax.imshow(image_data,clim=[0.8*np.mean(image_data),1.2*np.mean(image_data)],cmap='gray_r')
            elif set_source in ['W1','W2'] :
                norm = ImageNormalize(image_data, interval=MinMaxInterval(), stretch=LogStretch())
                ax.imshow(image_data,cmap='gray_r',norm=norm,aspect='equal')#,vmin=0.001*np.mean(image_data), vmax=2*np.max(image_data))
            else :
                ax.imshow(image_data,clim=contrast,cmap='gray_r')
            for ra, dec, sep in zip(self.datatable['ra'], self.datatable['dec'], self.datatable['sep']):
                if sep > self.allowed_sep: 
                        #print(ra,dec)
                    ax.plot(ra, dec, color = 'red', marker = 'o', markersize=3,transform=ax.get_transform('world'))
                else:
                    ax.plot(ra, dec, color = 'blue', marker = 'o', markersize=3,transform=ax.get_transform('world'))
                    
                    
                c = SphericalCircle((self.pos[0] * u.deg, self.pos[1] * u.deg), rad * u.arcsec,
                                    vertex_unit= u.deg,
                     edgecolor='yellow', facecolor='none',
                     transform=ax.get_transform('fk5'))

                ax.add_patch(c)
                ax.plot(self.pos[0], self.pos[1],marker='x',color='xkcd:cyan',markersize=8,
                    transform=ax.get_transform('world'))
                
                ax.coords[0].grid(color='white', ls='dotted')
                ax.coords[0].set_axislabel('RA')
                ax.coords[0].set_ticks(spacing=0.5* u.arcmin)
                ax.coords[0].set_major_formatter('dd:mm:ss')

                ax.coords[1].grid(color='white', ls='dotted')
                ax.coords[1].set_axislabel('DEC')
                ax.coords[1].set_ticks(spacing=0.5* u.arcmin)
                ax.coords[1].set_major_formatter('dd:mm:ss')
                ax.set_title(self.source + '\n' + self.WISE_name + " " +  set_source)

#         else :
#             fig=plt.figure(figsize=(10,10))
#             ax = fig.add_subplot(1, 1, 1)                

        if save == 'yes' :
            fig.savefig("/home/treynolds/data/LIRGS/WISE/WISE_analysis/Data/WISE_position_plots/" +\
                        self.source + "_" + self.WISE_name + ".pdf",
                       bbox_inches='tight')
        
        plt.show()
        
    def phot_diag(self,SAA='yes',Moon_mask='yes'):
        """
        Makes a plot showing the magnitudes of the data points and colors them according to flags
        NB: Diagnostic only, no processing is done here
        """
        
        fig, ax = plt.subplots(1,1)
        ax.errorbar(self.datatable['w1mpro'], self.datatable['w2mpro'],\
                    xerr = self.datatable['w1sigmpro'], yerr = self.datatable['w2sigmpro'],\
                   linestyle = '',color='xkcd:navy blue',marker='x',markersize = 6)
        # Poor quality
        qual_mask = self.datatable['qual_frame'] == 0
        ax.errorbar(self.datatable['w1mpro'][qual_mask], self.datatable['w2mpro'][qual_mask],\
                    xerr = self.datatable['w1sigmpro'][qual_mask], yerr = self.datatable['w2sigmpro'][qual_mask],\
                   linestyle = '', color = 'red', marker = 'o', label = 'poor qual')       
        # Flagged as upper limit or no profile-fit
        qual_mask = [('X' in i or 'U' in i) for i in self.datatable['ph_qual']]
        ax.errorbar(self.datatable['w1mpro'][qual_mask], self.datatable['w2mpro'][qual_mask],\
                    xerr = self.datatable['w1sigmpro'][qual_mask], yerr = self.datatable['w2sigmpro'][qual_mask],\
                   linestyle = '', color = 'orange', marker = 'o', label = 'photometry flag')       
        # Close to SAA
        if SAA == 'yes':
            qual_mask = [abs(i) < 5 for i in self.datatable['saa_sep']]
            ax.errorbar(self.datatable['w1mpro'][qual_mask], self.datatable['w2mpro'][qual_mask],\
                    xerr = self.datatable['w1sigmpro'][qual_mask], yerr = self.datatable['w2sigmpro'][qual_mask],\
                   linestyle = '', color = 'purple', marker = 'o', label = 'SAA sep')      
        # Within the moon-mask area
        if Moon_mask == 'yes' :
            qual_mask = self.datatable['moon_masked'] != 0
            ax.errorbar(self.datatable['w1mpro'][qual_mask], self.datatable['w2mpro'][qual_mask],\
                    xerr = self.datatable['w1sigmpro'][qual_mask], yerr = self.datatable['w2sigmpro'][qual_mask],\
                   linestyle = '', color = 'green', marker = 'o', label = 'moon mask')      
        # Flagged as known artifact
        # i == 0 implies it's fine. Various letter codes imply other stuff, could be more careful
        qual_mask = [(i != 0 and i != "0000") for i in self.datatable['cc_flags']]
        ax.errorbar(self.datatable['w1mpro'][qual_mask], self.datatable['w2mpro'][qual_mask],\
                    xerr = self.datatable['w1sigmpro'][qual_mask], yerr = self.datatable['w2sigmpro'][qual_mask],\
                   linestyle = '', color = 'pink', marker = 'o', label = 'artifact flag')
        # Too far offset
        qual_mask = self.datatable['sep'] > self.allowed_sep
        ax.errorbar(self.datatable['w1mpro'][qual_mask], self.datatable['w2mpro'][qual_mask],\
                    xerr = self.datatable['w1sigmpro'][qual_mask], yerr = self.datatable['w2sigmpro'][qual_mask],\
                   linestyle = '', color = 'black', marker = 'o', label = 'offset')      
        ax.legend()
        ax.set_xlabel('W1')
        ax.set_ylabel('W2')
        fig.set_size_inches(7,7)
        plt.show()
        
    def filter_data(self,filters='all',soft='no'):
        """
        Removes data based on a number of criteria. Currently only has strictest setting or the 
        "soft" setting which allows objects with bad cc_flags
        Also checks for saturation and applies non-linearity correction
        """
        
        # If we want to use the aperture_mags

        if soft == 'yes':
            print('Allowing bad cc_flags')
            neowise_mask = [all(constraint) for constraint in zip(
                self.datatable['sep'] < self.allowed_sep,\
                self.datatable['qual_frame'] > 0,\
                self.datatable['qi_fact'] > 0,\
                [('X' not in i and 'U' not in i) for i in self.datatable['ph_qual']],\
                [abs(i) > 5 for i in self.datatable['saa_sep']],\
                self.datatable['moon_masked'] == 0,\
                ~np.isnan(self.datatable['w1mag']),\
                ~np.isnan(self.datatable['w2mag']),\
                ~np.isnan(self.datatable['w1mpro']),\
                ~np.isnan(self.datatable['w2mpro']))]

        else :
            neowise_mask = [all(constraint) for constraint in zip(
                self.datatable['sep'] < self.allowed_sep,\
                self.datatable['qual_frame'] > 0,\
                self.datatable['qi_fact'] > 0,\
                [('X' not in i and 'U' not in i) for i in self.datatable['ph_qual']],\
                [abs(i) > 5 for i in self.datatable['saa_sep']],\
                self.datatable['moon_masked'] == 0,\
                [(i == 0.0 or i == '0000') for i in self.datatable['cc_flags']],\
                ~np.isnan(self.datatable['w1mag']),\
                ~np.isnan(self.datatable['w2mag']),\
                ~np.isnan(self.datatable['w1mpro']),\
                ~np.isnan(self.datatable['w2mpro'])
#                 self.datatable['w1flg'] == 0.0, 
#                 self.datatable['w2flg'] == 0.0
            )]      
            
        neowise_df = pd.DataFrame({})
        neowise_df['mjd'] = self.datatable['mjd'][neowise_mask]
           
        neowise_df['w1mag'] = self.datatable['w1mpro'][neowise_mask]
        neowise_df['w1sig'] = self.datatable['w1sigmpro'][neowise_mask]
        neowise_df['w2mag'] = self.datatable['w2mpro'][neowise_mask]
        neowise_df['w2sig'] = self.datatable['w2sigmpro'][neowise_mask]
        neowise_df['w1flux'] = mag_to_fluxdens(neowise_df['w1mag'], self.f0_wise_3_4)
        neowise_df['w1fluxsig'] =   neowise_df['w1flux'] * mag_unc_to_flux_unc(neowise_df['w1sig'])
        neowise_df['w2flux'] = mag_to_fluxdens(neowise_df['w2mag'], self.f0_wise_4_6)
        neowise_df['w2fluxsig'] =   neowise_df['w2flux'] * mag_unc_to_flux_unc(neowise_df['w2sig'])
            
        neowise_df['w1apmag'] = self.datatable['w1mag'][neowise_mask]
        neowise_df['w1apsig'] = self.datatable['w1sigm'][neowise_mask]
        neowise_df['w2apmag'] = self.datatable['w2mag'][neowise_mask]
        neowise_df['w2apsig'] = self.datatable['w2sigm'][neowise_mask]    
        neowise_df['w1apflux'] = mag_to_fluxdens(neowise_df['w1mag'], self.f0_wise_3_4)
        neowise_df['w1apfluxsig'] =   neowise_df['w1flux'] * mag_unc_to_flux_unc(neowise_df['w1sig'])
        neowise_df['w2apflux'] = mag_to_fluxdens(neowise_df['w2mag'], self.f0_wise_4_6)
        neowise_df['w2apfluxsig'] =   neowise_df['w2flux'] * mag_unc_to_flux_unc(neowise_df['w2sig'])


        print("Length:", len(self.data['w1mag']))
        if len(self.data['w1mag']) == 0:
            self.baddata ='yes'
        else :
            self.baddata ='no'
        
        # Here we also want to extract the data rejected for being too far from the expected coords
        neowise_extras = pd.DataFrame({})
        mask = [all(constraint) for constraint in zip(
                self.datatable['sep'] > self.allowed_sep,\
                self.datatable['qual_frame'] > 0,\
                self.datatable['qi_fact'] > 0,\
                [('X' not in i and 'U' not in i) for i in self.datatable['ph_qual']],\
                [abs(i) > 5 for i in self.datatable['saa_sep']],\
                self.datatable['moon_masked'] == 0,\
                [(i == 0.0 or i == '0000') for i in self.datatable['cc_flags']],\
                ~np.isnan(self.datatable['w1mpro']),\
                ~np.isnan(self.datatable['w2mpro']))]
        
        for i in self.datatable.columns :
            neowise_extras[i] = self.datatable[i][mask]
        self.companion_data = neowise_extras
        
        if self.ALLWISE_datatable.empty != True :
            try :
                if self.ALLWISE_datatable.empty == False :
                    neowise_df['w3mag'] = self.datatable['w3mpro'][neowise_mask]
                    neowise_df['w3sig'] = self.datatable['w3sigmpro'][neowise_mask]
                    neowise_df['w3flux'] = mag_to_fluxdens(neowise_df['w3mag'], self.f0_W3)
                    neowise_df['w3fluxsig'] =   neowise_df['w3flux'] * mag_unc_to_flux_unc(neowise_df['w3sig'])                                
                    neowise_df['w4mag'] = self.datatable['w4mpro'][neowise_mask]
                    neowise_df['w4sig'] = self.datatable['w4sigmpro'][neowise_mask]
                    neowise_df['w4flux'] = mag_to_fluxdens(neowise_df['w4mag'], self.f0_W4)
                    neowise_df['w4fluxsig'] =   neowise_df['w4flux'] * mag_unc_to_flux_unc(neowise_df['w4sig'])
            except KeyError:
                pass
        
        #Apply non-linearity correction
        # THIS NEEDS an external file! Careful
        w1_lincorr = np.loadtxt('/home/treynolds/data/LIRGS/WISE/WISE_analysis/Data/W1_saturation_corr.txt',
                               skiprows=5,unpack='yes')
        w2_lincorr = np.loadtxt('/home/treynolds/data/LIRGS/WISE/WISE_analysis/Data/W2_saturation_corr.txt',
                                skiprows=5,unpack='yes')
        
        w1mags = neowise_df["w1mag"].to_numpy()
        w2mags = neowise_df["w2mag"].to_numpy()
        nonlin_unc_w1 = np.zeros(len(w1mags))
        nonlin_unc_w2 = np.zeros(len(w2mags))
    
        sat_warning=None
        for i in range(0,len(w1mags)) :
            mag = w1mags[i]
            if mag < 8 :
                if sat_warning == None :
                    print("Applying saturation correction for W1")
                    sat_warning = 'yes'
                find_mag = [abs(mag-i) for i in w1_lincorr[0]]
                index = np.where(find_mag == np.min(find_mag))
                mag_corr = mag +  w1_lincorr[1][index]
                corr_err = np.max(np.array([w1_lincorr[2][index],w1_lincorr[3][index]]))
                nonlin_unc_w1[np.where(w1mags == mag)] = corr_err
                w1mags[np.where(w1mags == mag)] = mag_corr[0]
                
        sat_warning=None
        for i in range(0,len(w2mags)) :
            mag = w2mags[i]
            if mag < 7 :
                if sat_warning == None :
                    print("Applying saturation correction for W2")
                    sat_warning = 'yes'
                find_mag = [abs(mag-i) for i in w2_lincorr[0]]
                index = np.where(find_mag == np.min(find_mag))
                mag_corr = mag +  w2_lincorr[1][index]
                corr_err = np.max(np.array([w2_lincorr[2][index],w2_lincorr[3][index]]))
                nonlin_unc_w2[np.where(w2mags == mag)] = corr_err
                w2mags[np.where(w2mags == mag)] = mag_corr[0]
        
        neowise_df["w1_nonlin_unc"] = nonlin_unc_w1; neowise_df["w2_nonlin_unc"] = nonlin_unc_w2
        
        neowise_df['w1flux'] = mag_to_fluxdens(neowise_df['w1mag'], self.f0_wise_3_4)
        #neowise_df['w1fluxsig'] =   neowise_df['w1flux'] / self.datatable['w1snr'][neowise_mask]
        neowise_df['w1fluxsig'] =   neowise_df['w1flux'] * mag_unc_to_flux_unc(neowise_df['w1sig'])
        
        # I did a test where I propogated the mag errs through instead of this
        # result was very similar, errors were a tiny bit less. So I will stick with this.
        neowise_df['w2flux'] = mag_to_fluxdens(neowise_df['w2mag'], self.f0_wise_4_6)
        #neowise_df['w2fluxsig'] =   neowise_df['w2flux'] / self.datatable['w2snr'][neowise_mask]
        neowise_df['w2fluxsig'] =   neowise_df['w2flux'] * mag_unc_to_flux_unc(neowise_df['w2sig'])

        self.data = neowise_df
        self.filtered = 'yes'
        print("Length:", len(self.data['w1mag']))
        if len(self.data['w1mag']) == 0:
            self.baddata ='yes'
        else :
            self.baddata ='no'
        
    def bin_data(self,plot='yes',mag_measure = 'mean',err_measure='SEM'):
        """
        Bins data from sets of observations. Will plot as default
        
        This function is a bit of a mess, now we do all 4 filters.
        Should rewrite as a loop to cut down on characters
        
        """
        
        # This is just for if you forget to filter, could remove
        if self.filtered == 'no':
            print('Perhaps you should filter first, making a workaround...')
            neowise_mask = [all(constraint) for constraint in zip(
                self.datatable['sep'] < self.allowed_sep,\
                self.datatable['qual_frame'] > 0,\
                self.datatable['qi_fact'] > 0,\
                [('X' not in i and 'U' not in i) for i in self.datatable['ph_qual']],\
                [abs(i) > 5 for i in self.datatable['saa_sep']],\
                self.datatable['moon_masked'] == 0,\
                ~np.isnan(self.datatable['w1mpro']),\
                ~np.isnan(self.datatable['w2mpro']))]
            neowise_df = pd.DataFrame({})
            neowise_df['mjd'] = self.datatable['mjd'][neowise_mask]
            neowise_df['w1mag'] = self.datatable['w1mpro'][neowise_mask]
            neowise_df['w1sig'] = self.datatable['w1sigmpro'][neowise_mask]
            neowise_df['w2mag'] = self.datatable['w2mpro'][neowise_mask]
            neowise_df['w2sig'] = self.datatable['w2sigmpro'][neowise_mask]
            self.data = neowise_df
            if len(self.data['w1mag']) == 0:
                self.baddata ='yes'
        if self.baddata == 'yes':
            print(self.source+': No good data!')
            return
        
                
        start_epoch = np.min(self.data['mjd'])
        end_epoch = np.max(self.data['mjd'])
        yr = u.year.to(u.d)    
        cycles = (end_epoch - start_epoch)/yr * 2
        #print(cycles) # should be close to .5 or .0, as WISE is on a 6 month cycle. Inspect below
        cycles = round(cycles,0)
        bins = [start_epoch - yr/4 + a*(yr/2) for a in np.arange(cycles + 2)]
        if plot == 'yes':
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.errorbar(self.data['mjd'], self.data['w1mag'], yerr=self.data['w1sig'],\
                        label=r"W1 measurements", color='orange', linestyle = '', marker = 'o', markersize=5)
            ax1.set_ylim(ax1.get_ylim()[::-1]) #invert y-axis
            
            ax2.errorbar(self.data['mjd'], self.data['w2mag'], yerr=self.data['w2sig'],\
                        label=r"W2 measurements", color='red', linestyle = '', marker = 'o', markersize=5)
            ax2.set_ylim(ax2.get_ylim()[::-1]) #invert y-axis
            
            for epoch in bins:
                ax1.axvline(epoch)
                ax2.axvline(epoch)
            fig.set_size_inches(12,6)
            
            plt.show()
            
            
        neowise_bin_df = pd.DataFrame({})

        # Bin mean mjd according to bin edges
        neowise_bin_df['mjd'] = scipy.stats.binned_statistic(self.data['mjd'].values,
                                                             self.data['mjd'].values,\
                                                       statistic=np.mean, bins=bins, range=None)[0]

        # we will assume the m_i are independent and drawn fro a gaussian distribution with the 
        # uncertainties correctly measured. Then we can use the weighted mean and weighted sigma, 
        # as described in (e.g.) Sokolovsky, 2017.
        
        def weighted_bins(data,filt):
            """filt: w1mag,w2mag,w1flux,w2flux"""
            filts = {'w1mag':'w1sig',"w2mag":"w2sig","w1flux":"w1fluxsig","w2flux":"w2fluxsig"}
            dates,bin_edges,binnumber = scipy.stats.binned_statistic(data['mjd'].values,data['mjd'].values,\
                        statistic=np.mean, bins=bins, range=None)
            weighted_mags = []; weighted_errs = []
            for n in range(1,np.max(binnumber)+1):  # Iterate over each bin
                pairs = zip(data[filt].values,data[filts[filt]].values,binnumber)
                in_bin = [(x,y) for x,y, nbin in pairs if nbin == n]
                if in_bin == []:
                    #print("no_data")
                    weighted_mags.append(np.nan)
                    weighted_errs.append(np.nan)
                    continue
                weighted_mags.append(weighted_mean(in_bin)[0])
                weighted_errs.append(weighted_mean(in_bin)[1])
                #print(weighted_mags)
            return np.array(weighted_mags), np.array(weighted_errs) 
        
        # This gives values for mag and error from the weighted mean and weighted sigma
#         w1mag_values, w1mag_err = weighted_bins(self.data,'w1mag')
#         #print(w1mag_values,w1mag_err)
#         w2mag_values, w2mag_err = weighted_bins(self.data,'w2mag')
#         w1flux_values, w1flux_err = weighted_bins(self.data, 'w1flux')
#         #print(w1flux_values)
#         w2flux_values, w2flux_err = weighted_bins(self.data,'w2flux')

  

        # Updating the ZP uncertainty to match that described here:
        # https://wise2.ipac.caltech.edu/docs/release/neowise/expsup/sec4_2d.html#monitor_zero
        # If the non-lin correction was applied, then additional unc from that is added   

        
        # Jarrett ZP uncs:
#         neowise_bin_df['w1mag'] = unp.uarray(w1mag_values,
#                             np.sqrt(w1mag_SEM**2 + w1mag_mean_nonlin_unc**2+ (2.5*np.log10(1.024))**2))        
#         neowise_bin_df['w2mag'] = unp.uarray(w2mag_values,
#                                 np.sqrt(w2mag_SEM**2+ (2.5*np.log10(1.028))**2 + w2mag_mean_nonlin_unc**2))        
#         neowise_bin_df['w1flux'] = unp.uarray(w1flux_values,
#                                     np.sqrt(w1flux_SEM**2+(w1flux_values*0.024)**2+ w1flux_mean_non_lin_unc**2))
#         neowise_bin_df['w2flux'] = unp.uarray(w2flux_values,
#                                     np.sqrt(w2flux_SEM**2+(w2flux_values*0.027)**2+ w2flux_mean_non_lin_unc**2 ) 
        
        
        if mag_measure == 'sigmean' :
            stat = np.mean(astropy.stats.sigma_clip)
            self.mag_measure = "sigma clipped mean"
        elif mag_measure == 'mean':
            stat = np.nanmean
            self.mag_measure = "mean"
        else :    
            stat = np.median
            self.mag_measure = "median"
            
        if err_measure == 'SEM':
            err_stat = SEM      
        elif err_measure == 'sigma':
            err_stat = np.std
        # If you want to use the weighted mean of the unc's for the points you are binning as the
        # overall error, use w1mag_err. The difference is small IF you are using the flux uncertertainty
        # term in the ZP, as that dominates.
        # The weighted average of the errors is less sensitive to outliers.
        
        # Bin mean magnitude, with error standard error of mean 
        
        w1mag_values = scipy.stats.binned_statistic(self.data['mjd'].values,self.data['w1mag'].values,\
                                                       statistic=stat, bins=bins, range=None)[0]
        w2mag_values = scipy.stats.binned_statistic(self.data['mjd'].values,self.data['w2mag'].values,\
                                                       statistic=stat, bins=bins, range=None)[0] 
        w1flux_values = scipy.stats.binned_statistic(self.data['mjd'].values,self.data['w1flux'].values,\
                                                       statistic=stat, bins=bins, range=None)[0]  
        w2flux_values = scipy.stats.binned_statistic(self.data['mjd'].values,self.data['w2flux'].values,\
                                                       statistic=stat, bins=bins, range=None)[0]
        
        # Want to take the mean weighted by the uncertainties
        
        w1mag_err = scipy.stats.binned_statistic(self.data['mjd'].values,self.data['w1mag'].values,\
                                                   statistic=err_stat, bins=bins, range=None)[0]
        w2mag_err = scipy.stats.binned_statistic(self.data['mjd'].values,self.data['w2mag'].values,\
                                                   statistic=err_stat, bins=bins, range=None)[0]
        w1flux_err = scipy.stats.binned_statistic(self.data['mjd'].values,self.data['w1flux'].values,\
                                                   statistic=err_stat, bins=bins, range=None)[0]
        w2flux_err = scipy.stats.binned_statistic(self.data['mjd'].values,self.data['w2flux'].values,\
                                                       statistic=err_stat, bins=bins, range=None)[0]
        
        w1mag_mean_nonlin_unc = scipy.stats.binned_statistic(self.data['mjd'].values,
                                                                       self.data['w1_nonlin_unc'].values,\
                                                       statistic=np.mean, bins=bins, range=None)[0] 
        w2mag_mean_nonlin_unc = scipy.stats.binned_statistic(self.data['mjd'].values,
                                                                       self.data['w2_nonlin_unc'].values,\
                                                       statistic=np.mean, bins=bins, range=None)[0] 
        w1flux_mean_non_lin_unc = (10**(0.4*w1mag_mean_nonlin_unc) - 1) * w1flux_values
        w2flux_mean_non_lin_unc = (10**(0.4*w2mag_mean_nonlin_unc) - 1) * w2flux_values      
        
        #print(w1flux_err)
        # new version from the NEOWISE website
        neowise_bin_df['w1mag'] = unp.uarray(w1mag_values,
                            np.sqrt(w1mag_err**2 + w1mag_mean_nonlin_unc**2+ (0.0026)**2))
        # New version? Take the 0.025 mags that is the max seasonal variation. Could do better
        # Take the 0.0061 RMS, it's not great but 0.025 is a bit silly
        neowise_bin_df['w2mag'] = unp.uarray(w2mag_values,
                                np.sqrt(w2mag_err**2+ (0.0061)**2 + w2mag_mean_nonlin_unc**2))
        
        neowise_bin_df['w1flux'] = unp.uarray(w1flux_values,
            np.sqrt(w1flux_err**2+(w1flux_values*mag_unc_to_flux_unc(0.0026))**2 + w1flux_mean_non_lin_unc**2))
        # new version Take the 0.025 mags that is the max seasonal variation. Could do better
        neowise_bin_df['w2flux'] = unp.uarray(w2flux_values,
                 np.sqrt(w2flux_err**2+(w2flux_values*mag_unc_to_flux_unc(0.0061))**2+ w2flux_mean_non_lin_unc**2))
        if self.ALLWISE_datatable.empty != True :
            try :
                if self.ALLWISE_datatable.empty == False :
                
                    w3mag_values,w3mag_err = bins_W3_W4(self.data,'w3mag',bins)
                    w3flux_values,w3flux_err = bins_W3_W4(self.data,'w3flux',bins)
                    w4mag_values,w4mag_err = bins_W3_W4(self.data,'w4mag',bins)
                    w4flux_values,w4flux_err = bins_W3_W4(self.data,'w4flux',bins)
        
        
                    neowise_bin_df['w3mag'] = unp.uarray(w3mag_values,
                                        np.sqrt(w3mag_err**2 + (flux_unc_to_mag_unc(0.045))**2))
                    neowise_bin_df['w4mag'] = unp.uarray(w4mag_values,
                                            np.sqrt(w4mag_err**2+ (flux_unc_to_mag_unc(0.057))**2))
                    neowise_bin_df['w3flux'] = unp.uarray(w3flux_values,
                        np.sqrt(w3flux_err**2+(w3flux_values*0.045)**2))
                    neowise_bin_df['w4flux'] = unp.uarray(w4flux_values,
                             np.sqrt(w4flux_err**2+(w4flux_values*0.057)**2))
            except KeyError :
                pass
            
        
        self.binned_data = neowise_bin_df   

    
    def plot_data(self,saveonly='no',save="yes",
                  path='/home/treynolds/data/LIRGS/WISE/WISE_analysis/Data/WISE_gal_plots/',
                 window = 'no',eplosion_epoch = [mjday('20190111')],flux='no',absmag='no',label='no'):
        """
        Makes plots of the W1 and W2 LCs. Will save to the path folder. saveonly = 'yes' not functional
        """
        
        label_1 = r"W1 weighted mean"
        label_2 = r"W2 weighted mean"

        
        if absmag != 'no' :
            self.data['w1mag'] = self.data['w1mag'] - absmag
            self.binned_data['w1mag'] = self.binned_data['w1mag'] - absmag
            self.data['w2mag'] = self.data['w2mag'] - absmag
            self.binned_data['w2mag'] = self.binned_data['w2mag'] - absmag
            
        if self.baddata == 'yes':
            print(self.source+': No good data!')
            return
        # Plot measurements and binned values
        fig, (ax1, ax2) = plt.subplots(1,2)
        
        if flux == 'yes':
            ax1.errorbar(self.data['mjd'], self.data['w1flux'], yerr=self.data['w1fluxsig'],
                        label=r"W1 measurements", color='black', linestyle = '', 
                         marker = 'o', markersize=5, alpha = .1, zorder = 0)         
            
            ax1.errorbar(np.array(self.binned_data['mjd']), unp.nominal_values(self.binned_data['w1flux']),
                         yerr = unp.std_devs(self.binned_data['w1flux']),
                        label=label_1, color='blue', linestyle = '',
                         marker = 'o', markersize=5, capsize = 3, elinewidth = 1, zorder=1)
            
            # W2
            ax2.errorbar(self.data['mjd'], self.data['w2flux'], yerr=self.data['w2fluxsig'],
                        label=r"W2 measurements", color='black', linestyle = '', 
                         marker = 'o', markersize=5, alpha = .1, zorder = 0)
            
            ax2.errorbar(np.array(self.binned_data['mjd']), unp.nominal_values(self.binned_data['w2flux']),
                         yerr = unp.std_devs(self.binned_data['w2flux']),
                        label=label_2, color='red', linestyle = '',
                         marker = 'o', markersize=5, capsize = 3, elinewidth = 1, zorder=1)
            
            ax1.set_ylim(ymin = min(self.data['w1flux'])*0.95,ymax = max(self.data['w1flux'])*1.05)
            ax2.set_ylim(ymin = min(self.data['w2flux'])*0.95, ymax = max(self.data['w2flux'])*1.05)
            ax1.set_ylabel(r'flux (Jy)')

            
        else :
        # W1
            ax1.errorbar(self.data['mjd'], self.data['w1mag'], yerr=self.data['w1sig'],
                        label=r"W1 measurements", color='black', linestyle = '', 
                         marker = 'o', markersize=5, alpha = .1, zorder = 0)         
            
            ax1.errorbar(np.array(self.binned_data['mjd']), unp.nominal_values(self.binned_data['w1mag']),
                         yerr = unp.std_devs(self.binned_data['w1mag']),
                        label=label_1, color='blue', linestyle = '',
                         marker = 'o', markersize=5, capsize = 3, elinewidth = 1, zorder=1)
            
            # W2
            ax2.errorbar(self.data['mjd'], self.data['w2mag'], yerr=self.data['w2sig'],
                        label=r"W2 measurements", color='black', linestyle = '', 
                         marker = 'o', markersize=5, alpha = .1, zorder = 0)
            
            ax2.errorbar(np.array(self.binned_data['mjd']), unp.nominal_values(self.binned_data['w2mag']),
                         yerr = unp.std_devs(self.binned_data['w2mag']),
                        label=label_2, color='red', linestyle = '',
                         marker = 'o', markersize=5, capsize = 3, elinewidth = 1, zorder=1)
            ax1.set_ylim(ymin = max(unp.nominal_values(self.binned_data['w1mag'])) + 0.2,
                         ymax = min(unp.nominal_values(self.binned_data['w1mag'])) - 0.2)
            ax2.set_ylim(ymin = max(unp.nominal_values(self.binned_data['w2mag'])) + 0.2,
                         ymax = min(unp.nominal_values(self.binned_data['w2mag'])) - 0.2)
            ax1.set_ylabel(r'mag')
        
        # saturation limits
        if flux != 'yes':
            ax1.axhline(y=8.0, color='red', linestyle = '--', linewidth = 1.0)
            ax2.axhline(y=7, color='red', linestyle = '--', linewidth = 1.0)
        
        # Optional: explosion/discovery epoch        
        if eplosion_epoch != None :
            for date in eplosion_epoch :
                ax1.axvline(x=date[0], color='black', linestyle = ':', linewidth = 1.0)#,label=date[1])
                ax2.axvline(x=date[0], color='black', linestyle = ':', linewidth = 1.0)#,label=date[1])
                if label == 'yes':
                    trans = ax1.get_xaxis_transform()
                    ax1.text(date[0]+50, 0.5, date[1], transform=trans,rotation=90)
                    trans = ax2.get_xaxis_transform()
                    ax2.text(date[0]+50, 0.5, date[1], transform=trans,rotation=90)        

        
        if window != 'no':
            ax1.set_ylim(ymin = window[0][0], ymax = window[0][1])
            ax2.set_ylim(ymin = window[1][0], ymax = window[1][1])            
        
        ax1.set_title(self.source+ "\n" + self.WISE_name + ' W1')
        ax2.set_title(self.source+ "\n" + self.WISE_name + ' W2')
        
        ax1.set_xlabel(r'MJD')
        ax2.set_xlabel(r'MJD')
        
        ax1.legend()
        ax2.legend()
        
        
        fig.set_size_inches(14,7)
        if save == "yes":
            save_loc = path + self.source + "_" + self.WISE_name + "_" + '_NeoWISE_lightcurve_2020.png'
            print("saving: ", save_loc)
            plt.savefig(save_loc,
                    bbox_inches='tight', dpi=300,transparent=False,facecolor="white")
        self.plots = [ax1,ax2]
        #print(self.plots)
        if saveonly != 'yes':
            plt.show()
                
    def write(self,path='/home/treynolds/data/LIRGS/WISE/WISE_analysis/Data/WISE_gal_processed_data/'):
        """
         Writes data tables to text files at the requested path.
        """
        if self.baddata == 'yes':
            print(self.source+': No good data!')
            return
        # Write masked and binned magnitudes to file
        self.data.to_csv(path + f'{self.source}_NeoWISE_masked.tbl', header = True, index = None, sep = '\t')
        self.binned_data.to_csv(path + f'{self.source}_NeoWISE_binned.tbl', header = True, index = None, sep = '\t')

    
#     def add_NIR_data(self,NIR_data):
#         """
#         If you have NIR_data, add it to this object. It can then be used for plotting or included in 
#         BB fits.
#         Input format is Date,Filter,Mag,Error
#         """
#         NIR_file = open(NIR_data)
#         NIR_read_df = pd.read_csv(NIR_data, skiprows = 1, names = ["Date","Filter","Mag","Err"])
#         mjdays = [mjday(str(NIR_read_df['Date'][i]))[0] for i in range(0,len(NIR_read_df))]
#         NIR_read_df['MJD'] = mjdays

    

    def normalised_plot(self,save='no',Dist=None,mjd_zero="auto",w1_zero="auto",w2_zero="auto",title=None,
                       scale=None,eplosion_epoch = [mjday('20190111')],w1_zero_err=None, w2_zero_err =None):
        """
        Distance: Dist in Mpc
        mjd_zero: Explosion epoch in mjd
        w1_zero: quiescent flux in W1 in Jy
        w2_zero: quiescent flux in W2 in Jy
        w*_zero_err: quiescent flux uncertainty in W* in Jy
            This will be ignored if not given, and in BB fitting the error will be assumed the same as for 
            the outburst flux (i.e *sqrt(2))
            THIS IS NOT FULLY IMPLEMENTED YET (20210325)
        
        """
        if Dist == None :
            Dist = self.dist
        if title == None :
            title = self.source
        # Make some guesses as to how to normalise the data
        if w1_zero == "auto":       
            w1_zero = np.mean(astropy.stats.sigma_clip(unp.nominal_values(self.data['w1flux']),sigma=1))
        if w2_zero == "auto":               
            w2_zero = np.mean(astropy.stats.sigma_clip(unp.nominal_values(self.data['w2flux']),sigma=1))
        if mjd_zero == "auto":
            mjd_zero = np.nanmin(self.data['mjd'])
#             x = np.array(list(zip(self.data['mjd'],self.data['w1flux'])))
#             min_pos = np.where(x==np.nanmin(x[:,1]))
#             mjd_zero = x[min_pos[0],0][0]
        if w1_zero_err != None and w2_zero_err != None :
            self.zeros = [mjd_zero,w1_zero,w2_zero,w1_zero_err,w2_zero_err]
        else :
            self.zeros = [mjd_zero,w1_zero,w2_zero]
    
        w1flux_sub = (self.data['w1flux'] - w1_zero) # host subbed flux in Jy
        w2flux_sub = (self.data['w2flux'] - w2_zero) # host subbed flux in Jy
        
        
        mag_mask = [all(constraint) for constraint in zip(
                self.binned_data['w1flux'] > w1_zero,\
                self.binned_data["w2flux"] > w2_zero)]
        
        w1flux_sub_transient = (self.binned_data['w1flux'] - w1_zero) # host subbed flux in Jy
        w2flux_sub_transient = (self.binned_data['w2flux'] - w2_zero) # host subbed flux in Jy   
        
        transient_mag = pd.DataFrame({})
        transient_mag['mjd'] = self.binned_data['mjd'][mag_mask]
        transient_mag['w1flux'] = w1flux_sub_transient[mag_mask]
        transient_mag['w2flux'] = w2flux_sub_transient[mag_mask]
        
        w1mag_sub = fluxdens_to_mag(transient_mag['w1flux'], self.f0_wise_3_4)
        w1mag_sub_err = flux_unc_to_mag_unc(unp.std_devs(transient_mag['w1flux']))
        
        w2mag_sub = fluxdens_to_mag(transient_mag['w2flux'], self.f0_wise_4_6)
        w2mag_sub_err = flux_unc_to_mag_unc(unp.std_devs(transient_mag['w2flux']))
 
        transient_mag["w1mag_transient"] = w1mag_sub
        transient_mag["w2mag_transient"] = w2mag_sub
        
#         print("W1 mag", w1mag_sub)
        
        w1Lum_sub = lamflam(w1flux_sub,3.4,Dist)
        w2Lum_sub = lamflam(w2flux_sub,4.6,Dist)
        print("Distance:", Dist)
        # assume errors in measurement and zero flux are similar in size, so * np.sqrt(2)
        w1Lum_sub_err = lamflam(self.data['w1fluxsig'],3.4,Dist) * np.sqrt(2) 
        w2Lum_sub_err = lamflam(self.data['w2fluxsig'],4.6,Dist) * np.sqrt(2)

        w1_binned_flux = (self.binned_data['w1flux']- w1_zero) # sub 
        w2_binned_flux = (self.binned_data['w2flux']- w2_zero) # sub 
        w1Lum_binned_sub = lamflam(unp.nominal_values(self.binned_data['w1flux'])- w1_zero,3.4,Dist)
        w2Lum_binned_sub = lamflam(unp.nominal_values(self.binned_data['w2flux'])- w2_zero,4.6,Dist)

        w1Lum_binned_sub_err = lamflam(unp.std_devs(self.binned_data['w1flux']),3.4,Dist)* np.sqrt(2)
        w2Lum_binned_sub_err = lamflam(unp.std_devs(self.binned_data['w2flux']),4.6,Dist)* np.sqrt(2)
        
        mjd_sub = self.data['mjd'] - mjd_zero; mjd_binned_sub = self.binned_data['mjd'] - mjd_zero
        
        # Putting all this processed data into the dataframe. 
        # This facilitates comparison plots
        self.data["w1Lum_sub"] = w1Lum_sub
        self.data["w2Lum_sub"] = w2Lum_sub
        self.data["w1Lum_sub_err"] = w1Lum_sub_err
        self.data["w2Lum_sub_err"] = w2Lum_sub_err
        
        self.binned_data["w1Lum_binned_sub"] = w1Lum_binned_sub
        self.binned_data["w2Lum_binned_sub"] = w2Lum_binned_sub
        self.binned_data["w1Lum_binned_sub_err"] = w1Lum_binned_sub_err
        self.binned_data["w2Lum_binned_sub_err"] = w2Lum_binned_sub_err
    
        # Try to add the subbed transient mags to the binned_data
        self.binned_data["w1mag_transient"] = transient_mag["w1mag_transient"]
        self.binned_data["w2mag_transient"] = transient_mag["w2mag_transient"]
    
        self.data['mjd_sub'] = mjd_sub ; self.binned_data["mjd_binned_sub"] = mjd_binned_sub

        fig, ax = plt.subplots(1,2)
        
        ax[0].errorbar(mjd_sub, w1Lum_sub, yerr=w1Lum_sub_err,
                    label=r"W1 measurements", color='black', linestyle = '', 
                     marker = 'o', markersize=5, alpha = .1, zorder = 0) 
        
        ax[0].errorbar(np.array(mjd_binned_sub), w1Lum_binned_sub,
                     yerr = w1Lum_binned_sub_err,
                    label=r"W1 measurements binned",
                     color='blue', linestyle = '',
                     marker = 'o', markersize=5, capsize = 3, elinewidth = 1, zorder=1)
        
        ax[1].errorbar(mjd_sub, w2Lum_sub, yerr=w2Lum_sub_err,
                    label=r"W2 measurements", color='black', linestyle = '', 
                     marker = 'o', markersize=5, alpha = .1, zorder = 0) 
        
        ax[1].errorbar(np.array(mjd_binned_sub), w2Lum_binned_sub,
                     yerr = w2Lum_binned_sub_err,
                    label=r"W2 measurements binned",
                     color='blue', linestyle = '',
                     marker = 'o', markersize=5, capsize = 3, elinewidth = 1, zorder=1)
        
        if scale == None :  
            ax[0].set_ylim(ymax=np.max([1E43,1.3*np.nanmax(w1Lum_binned_sub)]),
                       ymin=np.min([-1E43,1.3*np.nanmin(w1Lum_binned_sub)]))
            ax[1].set_ylim(ymax=np.max([1E43,1.3*np.nanmax(w2Lum_binned_sub)]),
                       ymin=np.min([-1E43,1.3*np.nanmin(w2Lum_binned_sub)]))
        else :
            ax[0].set_ylim(ymax=scale[0],ymin=-1*scale[0])
            ax[1].set_ylim(ymax=scale[1],ymin=-1*scale[1])
          
        
        for axis in ax.ravel() :
            axis.axhline(0,ls='--') 
            hline_mark = np.min([np.nanmin(w1Lum_binned_sub),np.nanmin(w2Lum_binned_sub)])
            axis.axhline(hline_mark,ls='--',color='black')                    
            axis.axhline(hline_mark + 10**42.5,ls='--',color='orange')     
            axis.axhline(hline_mark + 10**43,ls='--',color='green')        
            axis.set_xlabel(r'MJD')
            axis.legend()
            
#         if eplosion_epoch != None :
#             for date in eplosion_epoch : 
#                 ax[0].axvline(x=date[0]-mjd_zero, color='black', linestyle = ':', linewidth = 1.0)
#                 ax[1].axvline(x=date[0]-mjd_zero, color='black', linestyle = ':', linewidth = 1.0)        
            
#         for axis in ax[0] :
#             axis.axhline(0,ls='--')        
#             axis.set_ylabel(r'Subbed flux (mJy)')
#             axis.legend()    
        for axis in ax :
            axis.axhline(0,ls='--',lw='0.5',color='k') 
            axis.axvline(0,ls='--',lw='0.5',color='k')        
            axis.set_ylabel(r'Host subtracted L (erg/s)')
            axis.legend()    
        if title != None :
            ax[0].set_title(title); ax[1].set_title(title)

            
        fig.set_size_inches(14,7)
        if save == 'yes':
            plt.savefig("/home/treynolds/data/LIRGS/WISE/WISE_analysis/energy_plots/"+ self.source +"_2020.pdf",\
                        bbox_inches='tight')
        elif save == 'transient':
            print("/home/treynolds/data/LIRGS/WISE/WISE_analysis/energy_plots/transients/"+ self.source +"_2020.pdf")
            plt.savefig("/home/treynolds/temp/temp.pdf",bbox_inches='tight')            

        w1max = np.nanmax(w1Lum_binned_sub)
        w2max = np.nanmax(w2Lum_binned_sub)
        w1Lum_data = list(w1Lum_binned_sub)
        w2Lum_data = list(w2Lum_binned_sub)
        
        # Here is where we do our filtering?
        
        self.transient = 'no'       
        while w2max > 1E42 :
            index = w2Lum_data.index(w2max)
            if w1Lum_data[index] > 1E42 :
                self.transient = 'yes'
                print("Transient?!")
                break
            else :
                w2Lum_data.pop(index)
                w2max = np.nanmax(w2Lum_data)
        w2Lum_data = list(w2Lum_binned_sub)        
        while w1max > 1E42 :
            index = w1Lum_data.index(w1max)
            if w2Lum_data[index] > 1E42 :
                self.transient = 'yes'
                print("Transient?!")
                break
            else :
                w1Lum_data.pop(index)
                w1max = np.nanmax(w1Lum_data)
        self.ax = ax
        plt.show()        
  
    def fit_BB(self,plot='yes',thresh=[10,9999]):
        """
        Simple BB fit to the 2 wise points
        Use as starting point for the MCMC walkers?
        """
        self.BB_filepath = "/home/treynolds/data/LIRGS/WISE/WISE_analysis/BB/"+ self.source + '/'
        try :
            os.mkdir(self.BB_filepath)
        except FileExistsError :
            pass
        wl = np.array([3.4,4.6])/1E+6
        mjds = self.binned_data["mjd_binned_sub"]
        w1flux = (unp.nominal_values(self.binned_data['w1flux']) - self.zeros[1] )* 1E3
        w2flux = (unp.nominal_values(self.binned_data['w2flux']) - self.zeros[2] )* 1E3
        w1errs = unp.std_devs(self.binned_data['w1flux'])* 1E3 * np.sqrt(2)
        w2errs = unp.std_devs(self.binned_data['w2flux'])* 1E3 * np.sqrt(2)
        data_all = np.array(list(zip(mjds,w1flux,w2flux,w1errs,w2errs)))
        def BB_complete(wav,temp,radius,distance):
            h=const.h ; c  = const.c ; e = math.exp(1) ; k  = const.k
            fr=c/wav
            flux = 2*h*fr**3 / (c**2*(e**(h*fr/(k*temp))-1))
            sca = 1E29*np.pi*radius**2/(distance)**2
            model_flux_value = sca*flux
            return model_flux_value
        results=[]
        good_data = []
        for data in data_all:
            fl = data[1:3];fl_errs = data[3:5]
            if data[0] < thresh[0] or data[0] > thresh[1] or np.isnan(data[1]) == True:
                continue
            else :
                good_data.append(data)
            fmodel = Model(BB_complete)  
            params = fmodel.make_params(temp=1000, radius=0.1,distance=self.dist*1E6)
            params['distance'].vary = False
            result = fmodel.fit(fl, params, wav=wl)
            results.append([data[0],result.params['temp'].value,result.params['radius'].value])
        self.simple_BB = results
        
        if plot == 'yes':
            fig = plt.figure(figsize=(15,15))
            for i in range(1,len(results)+1):
                if len(results) <5 :
                    ax1 = fig.add_subplot(2,2,i)  # create an axes object in the figure
                if len(results) <7 :
                    ax1 = fig.add_subplot(3,2,i)  # create an axes object in the figure 
                elif len(results) <10 :
                    ax1 = fig.add_subplot(3,3,i)  # create an axes object in the figure 
                elif  len(results) <13 :
                    ax1 = fig.add_subplot(4,3,i)  # create an axes object in the figure 
                elif  len(results) <17 :
                    ax1 = fig.add_subplot(4,4,i)  # create an axes object in the figure 
                else :
                    ax1 = fig.add_subplot(4,5,i)  # create an axes object in the figure 

                fl = good_data[i-1][1:3]; fl_err = good_data[i-1][3:5]
                ax1.errorbar(wl,fl,yerr=fl_err, linestyle=':',marker = 'o',color = 'xkcd:red', label='Data')
                x = np.linspace(wl[0],wl[-1],1000)
                
                y = BB_complete(x,results[i-1][1],results[i-1][2],self.dist*1E6)
                ax1.plot(x,y)
                ax1.axes.set_xlim(wl[0]-1E-7,wl[-1]+1E-7)
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, \
                   ncol=2, mode="expand", borderaxespad=0.)
            plt.savefig(self.BB_filepath + 'simple_BBs_'+  self.source + "_2020.pdf",bbox_inches='tight')
            plt.show()
 
        
    def MCMC_BB_fit(self,scale='auto',temp='auto',nwalkers = 300, ndim=2, nsample=100,thresh=[10,9999],
                   zero_err = None):
        """
        Fit a BB to the subtracted W1 and W2 flux
        """
        self.BB_filepath = "/home/treynolds/data/LIRGS/WISE/WISE_analysis/BB/"+ self.source + '/'
        try :
            os.mkdir(self.BB_filepath)
        except FileExistsError :
            pass
        
        self.BB_plot_numbers = []
        if self.simple_BB != None and scale == 'auto' :
            shift = 'yes'
            scales = list(zip(*self.simple_BB))[2]
            print("SCALES: ", scales)
        if self.simple_BB != None and temp == 'auto' :
            temps = list(zip(*self.simple_BB))[1]
            print("TEMPS: ", temps)
        else :
            shift = 'no'
        
        def MCMC_run(wl,fl, fl_err, dist, epoch,scale=scale,temp=temp,
                     nwalkers = nwalkers, ndim=ndim, nsample=nsample) :
            """
            MCMC fittins
            """  
            
            dist= dist *1E6
            scale_lower,scale_upper = scale
            T_lower,T_upper = temp
            burnin = int(nsample * 0.5)
            print(wl, fl, fl_err,scale,temp,dist)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_likelihood, 
                                            args=[wl, fl, fl_err,scale,temp, self.dist])
        
            # Starting positions predetermined with small perturbations
        #    p0 = np.array([0.1, 1000])                                       # Starting positon for walkers
        #    pos0 = [p0 + 0.0001 * np.random.randn(ndim) for j in range(nwalkers)]     # Perturb starting positions
            
            # Starting positions across flat prior range
            pos0 = np.transpose([np.random.uniform(low=scale_lower, high=scale_upper, size=(nwalkers,)),\
                                 np.random.uniform(low=T_lower, high=T_upper, size=(nwalkers,))])
             
            # Starting positions drawn from normal distribution
        #    pos0 = np.transpose([np.random.normal(loc=scale_init, scale=0.2*scale_init, size=nwalkers),\
        #                         np.random.normal(loc=temp_init, scale=0.2*temp_init, size=nwalkers)])
            
            self.temp = (sampler,pos0,nsample)
            sampler.run_mcmc(pos0, nsample)  ## Run the sampler nsample times
        
            ### Make a trace plot
            label_names=[r"$Radius (pc)$", r"$T (K)$"]
            data = sampler.chain # *not* .flatchain
            fig, axs = plt.subplots(ndim, 1)
            for i in range(ndim): # free MCMC_run(wl,fl, fl_err, dist, epoch)parameters
                for j in range(int(np.floor(nwalkers/5))): # walkers
                    axs[i].plot( np.arange(nsample), data[j,:,i],lw=0.5)
                 # x-axis is just the current iteration number
                fig.savefig(self.BB_filepath + str(np.round(epoch,2)) +  "_trace.pdf",bbox_inches='tight')
            
            
            # Make a corner plot
            figure = corner.corner(sampler.chain[:, burnin:, :].reshape(-1, 2), 
                                 labels=label_names,
                                 quantiles=[0.16, 0.5, 0.84])
                                 # Discard first 300 samples (burn in)
        
            figure.savefig(self.BB_filepath +"corner_" + str(np.round(epoch,2))  + ".pdf",
                           bbox_inches='tight')
        
            samples = sampler.chain[:, burnin:, :].reshape(-1, 2) 
            self.samples = samples
            scale_best = np.median(sampler.chain[:, burnin:, 0])
            T_best     = np.median(sampler.chain[:, burnin:, 1])   # Averaging over sampler chains and walkers
        
            # BB temperature in K
            print("Best Temp: ", T_best)
            # BB radius in pc
            print("Best Radius (pc): ", scale_best)
            # BB radius in cm
            print("Best Radius (cm): ", scale_best*3.0856776E+18)
        
            Tlow_16 =  np.percentile(samples, 16, 0)
            Thigh_84 =  np.percentile(samples, 84, 0)
            
            scale_stddev = np.std(sampler.chain[:, burnin:, 0])
            T_stddev     = np.std(sampler.chain[:, burnin:, 1])
            
            # Produce 3 sigma sigma clipped stats
            scale_sigclipped_sd = np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 0],sigma=1))
            T_sigclipped_sd   =  np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 1],sigma=1))
        
            print (Tlow_16, Thigh_84)
        
            print("T_sigma with/without clipping:",T_stddev,T_sigclipped_sd)
            print("scale_sigma with/without clipping:",scale_stddev,scale_sigclipped_sd)
            print("Mean acceptance fraction: {0:.3f}"
                        .format(np.mean(sampler.acceptance_fraction)))
            #plot data and best values
            bestplot(T_best,scale_best,wl,fl,fl_err,epoch,self.dist)
            self.BB_plot_numbers.append([T_best,scale_best,wl,fl,fl_err,epoch,Tlow_16[0], Tlow_16[1], Thigh_84[0], Thigh_84[1]])
            return [T_best,scale_best,scale_best*3.0856776E+18, Tlow_16[0], Tlow_16[1], Thigh_84[0], Thigh_84[1], 
                    samples,(T_stddev,scale_stddev),(T_sigclipped_sd,scale_sigclipped_sd),epoch] 
    
        # Actually run the fit here
        #need stefan boltzmann
        stef = 5.670E-8 #SI
        # wavelengths for W1 and W2
        wl = np.array([3.4,4.6])/1E+6
        # Need fluxes in mJy
        mjds = self.binned_data["mjd_binned_sub"]
        w1flux = (unp.nominal_values(self.binned_data['w1flux']) - self.zeros[1] )* 1E3
        w2flux = (unp.nominal_values(self.binned_data['w2flux']) - self.zeros[2] )* 1E3
        
        if len(self.zeros) == 3 :
            w1errs = unp.std_devs(self.binned_data['w1flux'])* 1E3*np.sqrt(2)
            w2errs = unp.std_devs(self.binned_data['w2flux'])* 1E3*np.sqrt(2)
        else :
            w1errs = (unp.std_devs(self.binned_data['w1flux']) + self.zeros[3])* 1E3
            w2errs = (unp.std_devs(self.binned_data['w2flux']) + self.zeros[4])* 1E3           

        data_all = np.array(list(zip(mjds,w1flux,w2flux,w1errs,w2errs)))
        results = []
        i=0
        for data in data_all:
            print("Pre-run:", data[0],wl,data[1:3],data[3:5])
            if data[0] < thresh[0] or data[0] > thresh[1] or np.isnan(data[1]) == True:
                continue     
            if shift == 'yes':
                print('Deriving limits from simple BB fit')
                #print("DEBUG: \n", data,"\n", scales,"\n",temps)
                result = MCMC_run(wl,data[1:3],data[3:5],self.dist,data[0],
                    scale = [0.1*scales[i],6*scales[i]],temp=[0.1*temps[i],3*temps[i]])
            else :
                result = MCMC_run(wl,data[1:3],data[3:5],self.dist,data[0])
                
            Lum = 4*np.pi * (ufloat(result[1],result[-2][1])* 3.0856776E+16)**2 *\
                                stef *ufloat(result[0],result[-2][0])**4 *1E7
            result.append(Lum)
            results.append(result)
            i+=1
            #break
               
        for i in range(0,len(results)) :
            if i == 0 :
                results[i].append(0)
            else :
                cuma_energy = (results[i-1][-2]+results[i][-1])*0.5 *((results[i][-2] - results[i-1][-3])*86400) + \
                                results[i-1][-1]
                results[i].append(cuma_energy)
                
        self.BB_results = results
    
    
    def BB_fit_plots(self):
        """
        Run this to produce plots that show the BB fits from the MCMC code
        """
        c  = const.c
#         def bb_function(wav,temp):
#             h=const.h ; c  = const.c ; e = math.exp(1) ; k  = const.k 
#             fr=c/wav
#             flux = 2*h*fr**3 / (c**2*(e**(h*fr/(k*temp))-1))
#             return flux
        
#         def sc_func(radius):
#             sca = 1E29*np.pi*radius**2/(self.dist*1E6)**2
#             return sca    
        
        fig = plt.figure(figsize=(15,15)) # create a figure object
        for i in range(1,len(self.BB_plot_numbers)+1):
            T_best,scale_best,wl,flux,flux_err,epoch, Scale_low, T_low, Scale_high, T_high = \
                        self.BB_plot_numbers[i-1]
            print(T_best,scale_best,wl,flux,flux_err,epoch,T_low, Scale_low,T_high,Scale_high)
            if len(self.BB_plot_numbers) <5 :
                ax1 = fig.add_subplot(2,2,i)  # create an axes object in the figure
            if len(self.BB_plot_numbers) <7 :
                ax1 = fig.add_subplot(3,2,i)  # create an axes object in the figure 
            elif len(self.BB_plot_numbers) <10 :
                ax1 = fig.add_subplot(3,3,i)  # create an axes object in the figure 
            elif  len(self.BB_plot_numbers) <13 :
                ax1 = fig.add_subplot(4,3,i)  # create an axes object in the figure 
            else :
                ax1 = fig.add_subplot(4,4,i)  # create an axes object in the figure 

            fr = []
            for item in wl :
                fr.append(c/item)
            ax1.errorbar(wl,flux, flux_err, linestyle=':',marker = 'o',color = 'xkcd:red', label='Data')
            x = np.linspace(3.4E-7,1E-5,10000)
            #x = np.linspace(wl[0],wl[-1],1000)
            #print(wl)
            y = []
            for item in x :
                y.append(c/item)
            ax1.plot(x,sc_func(scale_best,self.dist)*bb_function(x,T_best), label='Fit')    
            ax1.fill_between(x,sc_func(scale_best,self.dist)*bb_function(x,T_best),sc_func(Scale_low,self.dist)*bb_function(x,T_high),
                            facecolor='xkcd:lime green', alpha=0.5,)
            ax1.fill_between(x,sc_func(scale_best,self.dist)*bb_function(x,T_best),sc_func(Scale_high,self.dist)*bb_function(x,T_low),
                             facecolor='xkcd:lime green', alpha=0.5,)
            plt.legend( loc="upper right", ncol=2)      
            ax1.axes.set_xlim(wl[0]-1E-7,wl[-1]+1E-7)    
            ax1.axes.set_xlim(x[0],x[-1])  
            ax1.set_title("Phase: " + str(np.round(self.BB_plot_numbers[i-1][5],2)))
            
        plt.savefig(self.BB_filepath + 'MCMC_BBs_'+  self.source + ".pdf",bbox_inches='tight')
        plt.show()       
    
    def BB_results_plot(self,evap=None,plot=None,simple='yes'):
        BBs = self.BB_results
        fig, (ax1,ax3) = plt.subplots(2,1,figsize=(10,20))         
        for BB in BBs:
            print(BB)
            break
        phase = [i[-3] for i in BBs]
        Temps = [i[0] for i in BBs]; Scales = [i[1] for i in BBs]
        ax1.errorbar(phase,Temps,
                    yerr=[[abs(i[4]-i[0]) for i in BBs],[abs(i[6]-i[0]) for i in BBs]],
                     color='xkcd:blue',marker='o',
                    label='Temp',elinewidth=1,capsize=5)
        ax2 = ax1.twinx()
        ax2.errorbar(phase,Scales,
                    yerr=[[abs(i[3]-i[1]) for i in BBs],[abs(i[5]-i[1]) for i in BBs]],
                     color='xkcd:red',marker='o',
                    label='Radius',elinewidth=1,capsize=5)
        
        
        ax1.tick_params(axis='y', labelcolor='xkcd:blue')
        ax2.tick_params(axis='y', labelcolor='xkcd:red')
        
        ax1.axes.set_xlabel('phase',fontsize=18) 
        ax1.axes.set_ylabel('Temp, K',fontsize=18,color='xkcd:blue')
        ax2.axes.set_ylabel('radius (pc)',fontsize=18,color='xkcd:red')
        #ax1.axes.set_xlim(0,1000)
        
        Lums = unp.nominal_values([i[-2] for i in BBs])
        Lums_errs = unp.std_devs([i[-2] for i in BBs])

        ax3.axes.errorbar(phase,Lums,Lums_errs,color='xkcd:blue')
        ax4 = ax3.twinx()
        Cuma_energy = unp.nominal_values([i[-1] for i in BBs])
        Cuma_energy_errs = unp.std_devs([i[-1] for i in BBs])
        ax4.errorbar(phase,Cuma_energy,Cuma_energy_errs,color='xkcd:red')    
 
        if self.simple_BB != None and simple == 'yes':
            mjd_BB = list(zip(*self.simple_BB))[0]
            T_BB = list(zip(*self.simple_BB))[1]
            rad_BB = list(zip(*self.simple_BB))[2]
            ax1.plot(mjd_BB,T_BB,color='xkcd:blue',marker='^',markersize=12,ls='',label='simple fit')
            ax2.plot(mjd_BB,rad_BB,color='xkcd:red',marker='^',ls='',markersize=12)
            

        ax1.tick_params(axis='y', labelcolor='xkcd:blue',labelsize=14)
        ax2.tick_params(axis='y', labelcolor='xkcd:red',labelsize=14)
        ax3.tick_params(axis='y', labelcolor='xkcd:blue',labelsize=14)
        ax4.tick_params(axis='y', labelcolor='xkcd:red',labelsize=14)   
        
        ax3.axes.set_xlabel('phase',fontsize=18)   
        ax3.axes.set_ylabel('Lum, erg/s',fontsize=18,color='xkcd:blue')
        ax4.axes.set_ylabel('Cumulative E (ergs)',fontsize=18,color='xkcd:red') 
        
        if evap != None:
            ax2.axhline(evap,ls='--',color='black')
        
        if plot != None :
            ax1.errorbar(plot[0],plot[1],yerr = plot[2])
            ax2.errorbar(plot[0],plot[3],yerr = plot[4])
        
        plt.subplots_adjust(wspace=0.3)
        
        fig.suptitle('BB_fitting for ' + self.source, fontsize=16)
        

        plt.savefig(self.BB_filepath +  self.source + ".pdf",
                   bbox_inches='tight')
        
        plt.show()

    def latex_BB_table(self,output='latex'):
        """
        Creates a latex table with the BB fitting results
        """
        BBs = self.BB_results
        phase = [i[-3] for i in BBs]
        Temps = [i[0] for i in BBs]; Scales = [i[1] for i in BBs]
        T_high = [abs(i[6]-i[0]) for i in BBs]; T_low = [abs(i[4]-i[0]) for i in BBs]
        Sca_high = [abs(i[5]-i[1]) for i in BBs]; Sca_low = [abs(i[3]-i[1]) for i in BBs]
        
        Lums = unp.nominal_values([i[-2] for i in BBs]); Lums_errs = unp.std_devs([i[-2] for i in BBs])
        Cuma_energy = unp.nominal_values([i[-1] for i in BBs])
        Cuma_energy_errs = unp.std_devs([i[-1] for i in BBs]) 
        
        
        
        table_output = open(self.BB_filepath +  self.source + "_" + output + "_results.txt",'w')
        if output == 'simple':        
            headers = ["Phase (d)",'Temp (K)','Temp_lower_err (K)','T_higher_err (K)',
                       'Radius (pc)','Rad_lower_err (pc)','Rad_higher_err (pc)','L (erg/s)','L_err (erg/s)',
                       'Cumulative E (erg)','Cumulative E errors']
            table_output.write(', '.join(headers) + '\n') 
            for i in range(0,len(phase)):                
                newlist = [str(np.round(phase[i],2)),str(np.round(Temps[i],2)),str(np.round(T_low[i],2))
                           ,str(np.round(T_high[i],2)),str(np.round(Scales[i],2)),str(np.round(Sca_low[i],2))
                           ,str(np.round(Sca_high[i],2)),str(np.round(Lums[i],-3)), str(np.round(Lums_errs[i],-3))
                           ,str(np.round(Cuma_energy[i],-3)),str(np.round(Cuma_energy_errs[i],-3))]
                newline = ', '.join(newlist) + '\n'
                table_output.write(newline)
                print(newline)        
                        
            
        else :
            headers = ['Temps','Radius','L','Cumulative E']
            table_output.write(' & '.join(headers) + + '\\ \hline \n')
            for i in range(0,len(phase)):
                Temp_total = str(np.round(Temps[i],2))+'$_{-' + str(np.round(T_low[i],2)) + \
                    '}^{+'+ str(np.round(T_high[i],2)) +'}$'
                Rad_total = str(np.round(Scales[i],2))+'$_{-' + str(np.round(Sca_low[i],2)) + '}^{+'+ \
                    str(np.round(Sca_high[i],2)) +'}$'
                Lum_total = str(np.round(Lums[i],-3)) + "$ \pm$ " + str(np.round(Lums_errs[i],-3))
                Cuma_energy_total = str(np.round(Cuma_energy[i],-3)) + "$ \pm$ " + str(np.round(Cuma_energy_errs[i],-3))
                
                newlist = [str(np.round(phase[i],2)),Temp_total,Rad_total,Lum_total,Cuma_energy_total]
                newline = ' & '.join(newlist) + '\\ \hline  \n'
                table_output.write(newline)
                print(newline)
        
        table_output.close()

        
        
    def WISE_colour(self,date_range=None):
        """
        Calculate the W1-W2 colour.
        Enter a range of dates to only use those values. 
        
        """
        if self.baddata == 'yes':
            print(self.source+': No good data!')
            return
        colour = self.binned_data["w1mag"]-self.binned_data["w2mag"]
        self.binned_data["colour"] = colour
        
        try :
            if date_range != None :
                filtered_colours = [row.colour for row in self.binned_data.itertuples(index=False) \
                                if date_range[0] <= row.mjd <= date_range[-1]]   
                self.colour = np.nanmean(unp.nominal_values(filtered_colours))
            else :
                mean_colour = np.nanmean(unp.nominal_values(colour))
                self.colour = mean_colour
        except ValueError :
                filtered_colours = [row.colour for row in self.binned_data.itertuples(index=False) \
                                if date_range[0] <= row.mjd <= date_range[date_range.index[-1]]]   
                self.colour = np.nanmean(unp.nominal_values(filtered_colours))                



# Some other, more involved functions, they are used by the MCMC fitting
        
def fit_bb(wl,fl,fl_errs,Dist,plot='no',MCMC='no'):
    """
    Simple BB fit to general data
    Use as starting point for the MCMC walkers
    wavelength in micron, flux in mJy, Dist in pc
    
    If MCMC = 'yes', then use the simple BB as a starting point and use the MCMC hammer.
    
    """

#     def BB_complete(wav,temp,radius):
#         h=const.h ; c  = const.c ; e = math.exp(1) ; k  = const.k
#         fr=c/wav
#         flux = 2*h*fr**3 / (c**2*(e**(h*fr/(k*temp))-1))
#         sca = 1E29*np.pi*radius**2/(Dist)**2
#         model_flux_value = sca*flux
#         return model_flux_value
    # non-linear least squares fit
    
    fmodel = Model(BB_complete)  
    params = fmodel.make_params(temp=1000, radius=0.1,distance=Dist)
    params['distance'].vary = False
    result = fmodel.fit(fl, params, wav=wl,weights=1/fl_errs)

    popt, pcov = curve_fit(BB_complete,wl,fl,sigma=fl_errs,p0=[1000,0.1])
    perr = np.sqrt(np.diag(pcov))
    plt.figure(figsize=(8,5))
    ax = plt.gca()
    print(wl,fl,fl_errs)
    ax.errorbar(wl, fl,fl_errs, marker='o',color='k')
    wl_full = np.linspace(wl[0],wl[-1],1000)
    ax.plot(wl_full, BB_complete(wl_full, *popt),'r-',

         label='fit: temp=%5.3f, scale=%5.3f' % tuple(popt))

    plt.xlabel('Position')
    plt.ylabel('Flux')
    plt.legend(loc=2)
    plt.show()
    if MCMC != 'yes':
        print("Not doing MCMC unless you ask :) ")
        return popt, perr

# Here are the standalone functions for BB fitting

def bb_function(wav,temp):
    """
    wav units are m
    """
    h=const.h ; c  = const.c ; e = math.exp(1) ; k  = const.k 
    fr=c/wav
    flux = 2*h*fr**3 / (c**2*(e**(h*fr/(k*temp))-1))
    return flux

def sc_func(radius,distance):
    sca = 1E29*np.pi*radius**2/(distance*1E6)**2
    return sca

# Liklihood function
def ln_likelihood(theta, *args):
    wl = args[0]
    fl = args[1]
    fl_err = args[2]
    scale_lower,scale_upper = args[3]
    T_lower,T_upper = args[4]
    distance = args[5]

    #print fl
    scale, T = theta
    if not (scale_upper >= scale >= scale_lower) \
    or not (T_upper >= T >= T_lower):
        return -np.inf
        print(scale, T)

    model_flux = np.zeros(len(wl))

    for i, wavelength_value in enumerate(wl):
 
        model_flux_value = sc_func(scale,distance)*bb_function(wavelength_value, T)
        model_flux[i] = model_flux_value

    chi_sq = ((model_flux - fl)**2)/(np.array(fl_err)**2)
    return -(chi_sq.sum())

def bestplot(T,R,wl,flux,flux_err,epoch,distance):      
    plt.clf()
    c  = const.c
    central_wav_dict = {'U':3650,'B':4450,'V':5510,'R':6580,'I':8060,'u':3540,'g':4570,
                        'r':6220,'i':7630,'z':9050,
                    'J':12350,'H':16620,'K':21590}
    fig = plt.figure(figsize=(6,8))  # create a figure object
    ax1 = fig.add_subplot(1, 1, 1)  # create an axes object in the figure 
    fr = []
    for item in wl :
        fr.append(c/item)
    ax1.errorbar(wl,flux, flux_err, linestyle=':',marker = 'o',color = 'xkcd:red', label='Data')

    x = np.linspace(wl[0],wl[-1],1000)
    y = []
    for item in x :
        y.append(c/item)
    
#    x = np.linspace(0.3,0.9,2000)
    ax1.plot(x,sc_func(R,distance)*bb_function(x,T), label='Fit')       
    #ax = plt.gca()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, \
           ncol=2, mode="expand", borderaxespad=0.)      
    #for key_filter in central_wav_dict.keys():
    #    ax1.axvline(central_wav_dict[key_filter]/1E10/1.067,lw=0.4,color='k',ls='--')
    ax1.axes.set_xlim(wl[0]-1E-7,wl[-1]+1E-7)    
    
    #Comment one out
    #fig.savefig()
    plt.show()

# This is just a combination of sc_func and bb_function
def BB_complete(wav,temp,radius,distance):
        h=const.h ; c  = const.c ; e = math.exp(1) ; k  = const.k
        fr=c/wav
        flux = 2*h*fr**3 / (c**2*(e**(h*fr/(k*temp))-1))
        sca = 1E29*np.pi*radius**2/(distance)**2
        model_flux_value = sca*flux
        return model_flux_value    
    
    
# Some functions for testing variability

# Lets code functions to run the variability tests of Sokolovsky 2017

def chi_square(data):
    """
    Input a WISE object, object.binned_data
    """
    
    fluxes = unp.nominal_values(data['w1flux'])
    errs = unp.std_devs(data['w1flux'])
    #print("Errs: ",errs)
    fluxes = fluxes[~np.isnan(fluxes)]
    errs = errs[~np.isnan(errs)]
    fluxes_with_errs = np.column_stack((fluxes,errs))
    N = len(fluxes)
    
    Weighted_mean_flux = np.nansum(np.array([i[0]/i[1]**2 for i in fluxes_with_errs])) \
        / np.nansum(np.array([1/i[1]**2 for i in fluxes_with_errs]))
    
    Chi_squared = np.nansum(np.array([(i[0]-Weighted_mean_flux)**2/i[1]**2 for i in fluxes_with_errs]))
    Reduced_Chi_squared = Chi_squared / (N-1)
    return Chi_squared, Reduced_Chi_squared 

def Von_Neumann(data):
    """
    Input a WISE object, object.binned_data
    """
    
    mags = unp.nominal_values(data['w1mag'])
    errs = unp.std_devs(data['w1mag'])
    #print(mags,errs)
    mags_W1 = mags[~np.isnan(mags)]
    errs_W1 = errs[~np.isnan(errs)]
    
    mags = unp.nominal_values(data['w2mag'])
    errs = unp.std_devs(data['w2mag'])
    #print(mags,errs)
    mags_W2 = mags[~np.isnan(mags)]
    errs_W2 = errs[~np.isnan(errs)]    
    
    mags_with_errs_W1 = np.column_stack((mags_W1,errs_W1))
    mags_with_errs_W2 = np.column_stack((mags_W2,errs_W2))
    N_W1 = len(mags_W1)
    N_W2 = len(mags_W2)  
    
    mean_W1 = mags_W1.mean(); mean_W2 = mags_W2.mean()
    
    Weighted_mean_mag = np.nansum(np.array([i[0]/i[1]**2 for i in mags_with_errs_W1])) \
        / np.nansum(np.array([1/i[1]**2 for i in mags_with_errs_W1]))
    delta_square = np.nansum(np.array([(mags_W1[i+1]-mags_W1[i])**2/(N_W1-1) for i in range(0,N_W1-1)])) 
    sigma_square = np.nansum(np.array([(mags_W1[i]-mean_W1)**2/(N_W1-1) for i in range(0,N_W1)])) 
    
    Eta_W1 = delta_square / sigma_square
    
    Weighted_mean_mag = np.nansum(np.array([i[0]/i[1]**2 for i in mags_with_errs_W2])) \
        / np.nansum(np.array([1/i[1]**2 for i in mags_with_errs_W2]))
    delta_square = np.nansum(np.array([(mags_W2[i+1]-mags_W2[i])**2/(N_W2-1) for i in range(0,N_W2-1)])) 
    sigma_square = np.nansum(np.array([(mags_W2[i]- mean_W2)**2/(N_W2-1) for i in range(0,N_W2)])) 
    
    Eta_W2 = delta_square / sigma_square    
    
    return np.array([Eta_W1,Eta_W2])

def mag_filter(data):
    """
    Input a WISE object.
    Takes the min and max value, calculates the implied luminosity change for the transient.
    Returns the Lum change
    """

    w1_mag_min = np.nanmin(unp.nominal_values(data.binned_data['w1mag']))
    w2_mag_min = np.nanmin(unp.nominal_values(data.binned_data['w2mag']))
    w1_mag_max = np.nanmax(unp.nominal_values(data.binned_data['w1mag']))
    w2_mag_max = np.nanmax(unp.nominal_values(data.binned_data['w2mag']))
    
    print(w1_mag_max,w1_mag_min)
    
    w1_mag_diff = w1_mag_max-w1_mag_min ; w2_mag_diff = w2_mag_max-w2_mag_min 
        
    return (w1_mag_diff,w2_mag_diff)

def Lum_filter(data,NEOWISE_only="yes",save="no"):
    """
    Input a WISE object.
    Takes the min and max value, calculates the implied luminosity change for the transient.
    Returns the Lum change
    """
    # This checks if the dataframe has data from before the NEOWISE mission in it.
    
    if data.baddata == 'yes':
            print(data.source+': No good data!')
            return
    
    if NEOWISE_only == "yes" : #and data.binned_data["mjd"][1] < 56700 :
        
        mask = [all(constraint) for constraint in zip(
                56700 < data.binned_data['mjd'])]

        NEOWISE_only_w1= data.binned_data['w1flux'][mask]
        NEOWISE_only_w2 = data.binned_data['w2flux'][mask]
#         print(NEOWISE_only_w1)
        
        if NEOWISE_only_w1.empty == True :
            return (0,0)
        
        w1_flux_min = np.nanmin(unp.nominal_values(NEOWISE_only_w1))
        w2_flux_min = np.nanmin(unp.nominal_values(NEOWISE_only_w2))
        w1_flux_max = np.nanmax(unp.nominal_values(NEOWISE_only_w1))
        w2_flux_max = np.nanmax(unp.nominal_values(NEOWISE_only_w2))
        
    else :
        w1_flux_min = np.nanmin(unp.nominal_values(data.binned_data['w1flux']))
        w2_flux_min = np.nanmin(unp.nominal_values(data.binned_data['w2flux']))
        w1_flux_max = np.nanmax(unp.nominal_values(data.binned_data['w1flux']))
        w2_flux_max = np.nanmax(unp.nominal_values(data.binned_data['w2flux']))   
        
    print(w1_flux_max,w1_flux_min)
    
#     w1_Lum_min = lamflam(w1_flux_min,3.4,data.dist); w2_Lum_min = lamflam(w2_flux_min,4.6,data.dist)
#     w1_Lum_max = lamflam(w1_flux_max,3.4,data.dist); w2_Lum_max = lamflam(w2_flux_max,4.6,data.dist)    
#     w1_Lum_diff = w1_Lum_max-w1_Lum_min ; w2_Lum_diff = w2_Lum_max-w2_Lum_min 
    
    w1_Lum_diff = lamflam(w1_flux_max-w1_flux_min,3.4,data.dist)
    w2_Lum_diff = lamflam(w2_flux_max-w2_flux_min,4.6,data.dist)

    if save == "yes":
        data.Lum_test = (w1_Lum_diff,w2_Lum_diff)
    
    return (w1_Lum_diff,w2_Lum_diff)

def AGN_filter(data, quiescent, sigma=3):
    """
    Input: WISE object
    Tests if the peak and troughs are a certain number of sigma from the mean
    sigma only calculated pre-explosion, with quiescient being the index of the pre-explosion
    """
    w1_flux_min = np.nanmin(unp.nominal_values(data.binned_data['w1flux']))
    w1_flux_max = np.nanmax(unp.nominal_values(data.binned_data['w1flux']))
    w1_flux_diff = w1_flux_max - w1_flux_min
    w1_flux_sigma = np.nanstd(unp.nominal_values(data.binned_data['w1flux']))
    print(sigma,"sigma-test: \n",sigma,"sigma = ", w1_flux_sigma * sigma)
    print('\n diff= ',w1_flux_diff)
    
    return w1_flux_diff - w1_flux_sigma*sigma
    
def mag_minmax(data):
    """
    Input a WISE object.
    Takes the min and max value, calculates the implied luminosity change for the transient.
    Returns the Lum change
    """

    w1_mag_min = np.nanmin(unp.nominal_values(data.binned_data['w1mag']))
    w2_mag_min = np.nanmin(unp.nominal_values(data.binned_data['w2mag']))
    w1_mag_max = np.nanmax(unp.nominal_values(data.binned_data['w1mag']))
    w2_mag_max = np.nanmax(unp.nominal_values(data.binned_data['w2mag']))
    
    return((w1_mag_max,w1_mag_min),(w2_mag_max,w2_mag_min))
       
    
    
    



