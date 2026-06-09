# NEOWISE_transient_LCs
## Description  

## Installation  

## Usage


sys.path.append('/PATH/TO/FOLDER/NEOWISE_transient_LCs/')
import WISE_module as WISE 

name = "TRANSIENT NAME"
coords = (RA,DEC)
print("downloading data for: ",name)
# This downloads the data
WISE.IRAS_query(PATH_TO_FOLDER_WHERE_YOU_WILL_DOWNLOAD_DATA,
    catalog="all",pos=coords,radius=30,name=name)

file_in = PATH_TO_FOLDER_WHERE_YOU_WILL_DOWNLOAD_DATA

# This initialises the class
WISE_data = WISE.WISE_Data(all_files_path=file_in,
                               pos=coords,
                     source=name,allowed_sep=2,
                       dist=DISTANCE_IN_Mpc)


# This queries the WISE server, downloads the image and shows the detections overplotted
# The flags are not very self-explanatory and I can't remember exactly how they work
WISE_data.position_diag(download='no',diff=100,set_source='W1',rad=3,
               cut=[[800,1000],[600,1000]],save="no"
               )

# This shows all the detections and whether they are flagged
WISE_data.phot_diag()

# This processes the detections removing the flagged and offset data
# The soft flag allows some flagged data in, is needed sometimes (Arp299)
WISE_data.filter_data(soft='no')

# This bins the data. The median is used, and the error is the standard error of mean
WISE_data.bin_data(plot="no",mag_measure='median',err_measure='SEM')

# This plots the data, using the eplosion epoch as a zero, in mags (or fluxes if you want)
WISE_data.plot_data(eplosion_epoch = [WISE.mjday("20210409")],
		 flux='no',save="no"
		 )


# This subtracts off a zero value that you specify in flux.
# It then plots the luminosity
# It also adds a "transient mag" column to the output dataframe based on this subtraction
WISE_data.normalised_plot(mjd_zero=56000,
      w1_zero=unp.nominal_values(WISE_data.binned_data['w1flux'][13]),
      w2_zero=unp.nominal_values(WISE_data.binned_data['w2flux'][13]))
      

