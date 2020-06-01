import numpy as np
import pdb as pdb
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def divspec(datadir, srcfile, stdfile, dtau=0, dpix=0, mode=None,plot=True):

    #Read in data for source and standard
    hdulist_src=fits.open(datadir+srcfile)
    data_src=hdulist_src[1].data
    hdr_src=hdulist_src[0].header

    hdulist_std=fits.open(datadir+stdfile)
    data_std=hdulist_std[1].data

    #Normalize spectra
    pflux_src=data_src['flux_pos']/np.median(data_src['flux_pos'])
    nflux_src=data_src['flux_neg']/np.median(data_src['flux_neg'])
    pflux_std=data_std['flux_pos']/np.median(data_std['flux_pos'])
    nflux_std=data_std['flux_neg']/np.median(data_std['flux_neg'])

    pwave_src=data_src['wave_pos']
    nwave_src=data_src['wave_neg']
    pwave_std=data_std['wave_pos']
    nwave_std=data_std['wave_neg']

    #If desired, do airmass correction of standard
    np.seterr(invalid='ignore')
    pflux_std=np.exp((1+dtau)*np.log(pflux_std))
    nflux_std=np.exp((1+dtau)*np.log(nflux_std))

    #If desired, do wavelength shift of standard
    dwave=(pwave_src[1]-pwave_src[0])*dpix    #Convert pixel shift to wavelength shift
    pwave_std=pwave_std+dwave
    nwave_std=nwave_std+dwave

    #Interpolate everything onto positive src wavelengths
    nf_src=interp1d(nwave_src,nflux_src,bounds_error=False)
    pf_std=interp1d(pwave_std,pflux_std,bounds_error=False)
    nf_std=interp1d(nwave_std,nflux_std,bounds_error=False)

    nflux_src=nf_src(pwave_src)
    pflux_std=pf_std(pwave_src)
    nflux_std=nf_std(pwave_src)

    #Divide source by standard
    pdiv=pflux_src/pflux_std
    ndiv=nflux_src/nflux_std

    #Combine positive and negative beams
    divflux=(pdiv+ndiv)/2.
    srcflux=(pflux_src+nflux_src)/2.
    stdflux=(pflux_std+nflux_std)/2.

    #Compute SNR's in desired regions
    if(mode is not None):
        if(mode=='M0'):
            xsnr1_left=4.66
            xsnr1_right=4.67
            xsnr2_left=4.685
            xsnr2_right=4.69
            xsnr3_left=4.70
            xsnr3_right=4.715
        if(mode=='M1'):
            xsnr1_left=4.73
            xsnr1_right=4.75
            xsnr2_left=4.75
            xsnr2_right=4.77
            xsnr3_left=4.77
            xsnr3_right=4.79
        if(mode=='M2'):
            xsnr1_left=4.965
            xsnr1_right=4.975
            xsnr2_left=4.985
            xsnr2_right=5.0
            xsnr3_left=5.01
            xsnr3_right=5.015
        if(mode=='M3'):
            xsnr1_left=5.04
            xsnr1_right=5.05
            xsnr2_left=5.065
            xsnr2_right=5.075
            xsnr3_left=5.09
            xsnr3_right=5.095
        w1=((pwave_src > xsnr1_left) & (pwave_src<xsnr1_right))
        w2=((pwave_src > xsnr2_left) & (pwave_src<xsnr2_right))
        w3=((pwave_src > xsnr3_left) & (pwave_src<xsnr3_right))
    else:
        w1=np.isfinite(srcflux)
        w2=w1
        w3=w1
                
    snr1=np.nanmean(divflux[w1])/np.nanstd(divflux[w1])
    snr2=np.nanmean(divflux[w2])/np.nanstd(divflux[w2])
    snr3=np.nanmean(divflux[w3])/np.nanstd(divflux[w3])
    print('SNR:', snr1,snr2,snr3)

    #Create columns
    c1  = fits.Column(name='wave', format='D', array=pwave_src)
    c2  = fits.Column(name='div', format='D', array=divflux)
    c3  = fits.Column(name='src', format='D', array=srcflux)
    c4  = fits.Column(name='std', format='D', array=stdflux)
    coldefs = fits.ColDefs([c1,c2,c3,c4])
    tbhdu = fits.BinTableHDU.from_columns(coldefs)
    primary_hdu = fits.PrimaryHDU(header=hdr_src)    
    hdul = fits.HDUList([primary_hdu, tbhdu])

   #Create output file name
    srcname=srcfile.split('_')[0]
    date=srcfile.split('_')[1]
    stdname=stdfile.split('_')[0]
    srcnum=srcfile.split('_')[2]
    if(mode is not None): 
        outfile=srcname+'_'+stdname+'_'+date+'_'+srcnum+'_'+mode+'_div.fits'
    else:
        outfile=srcname+'_'+stdname+'_'+date+'_'+srcnum+'_div.fits'

    #Write to fits file    
    hdul.writeto(outfile,overwrite=True) 
    print('Writing to: ', outfile)

    #Create astropy table
    spectrum_table = Table([pwave_src, divflux, srcflux, stdflux], names=('wave', 'div', 'src','std'),  dtype=('f8', 'f8','f8','f8'))
    spectrum_table['wave'].unit = 'micron'

    if(plot==True):
        fig=plt.figure(figsize=(14,6))
        ax1=fig.add_subplot(211)
        ax1.plot(spectrum_table['wave'],spectrum_table['src'],label='src')
        ax1.plot(spectrum_table['wave'],spectrum_table['std'],label='std')
        ax1.legend()

        ax2=fig.add_subplot(212)
        ax2.plot(spectrum_table['wave'],spectrum_table['div'],label='div')
        ax2.set_ylim(0.8,1.2)
        ax2.legend()

        if(mode is not None):
            ax1.axvline(xsnr1_left,linestyle='--',color='C2')
            ax1.axvline(xsnr1_right,linestyle='--',color='C2')
            ax1.axvline(xsnr2_left,linestyle='--',color='C2')
            ax1.axvline(xsnr2_right,linestyle='--',color='C2')
            ax1.axvline(xsnr3_left,linestyle='--',color='C2')
            ax1.axvline(xsnr3_right,linestyle='--',color='C2')
            ax2.axvline(xsnr1_left,linestyle='--',color='C2')
            ax2.axvline(xsnr1_right,linestyle='--',color='C2')
            ax2.axvline(xsnr2_left,linestyle='--',color='C2')
            ax2.axvline(xsnr2_right,linestyle='--',color='C2')
            ax2.axvline(xsnr3_left,linestyle='--',color='C2')
            ax2.axvline(xsnr3_right,linestyle='--',color='C2')
        plt.show()


    return spectrum_table

def gluespec(datadir, filelist):
    mode=[]
    date=[]
    src=[]
    for myfile in filelist:
        src.append(myfile.split('_')[0])
        date.append(myfile.split('_')[2])
        mode.append(myfile.split('_')[4])
    outfile=src[0]+'_'+date[0]+'_'+'glue.fits'
    m0bool=(np.array(mode)=='M0')
    m1bool=(np.array(mode)=='M1')
    m2bool=(np.array(mode)=='M2')
    m3bool=(np.array(mode)=='M3')
    m0files=[]
    m1files=[]
    m2files=[]
    m3files=[]
    if(m0bool.any()):
        m0files=np.array(filelist)[m0bool]
        for i,myfile in enumerate(m0files):
            m0file=datadir+myfile
            hdulist=fits.open(m0file)
            data=hdulist[1].data
            wave=data['wave']
            div=data['div']
            std=data['std']
            if(i==0): 
                wave0_m0=wave
                std_m0=std
                div_all_m0=div
            else:
                f=interp1d(wave,div)
                div_interp=f(wave0_m0)
                div_all_m0=div_all_m0+div_interp
        div_all_m0=div_all_m0/np.size(m0files)
    else:
        div_all_m0=np.array([])
        wave0_m0=np.array([])
        std_m0=np.array([])
        
    if(m1bool.any()):
        m1files=np.array(filelist)[m1bool]
        for i,myfile in enumerate(m1files):
            m1file=datadir+myfile
            hdulist=fits.open(m1file)
            data=hdulist[1].data
            wave=data['wave']
            div=data['div']
            std=data['std']
            if(i==0): 
                wave0_m1=wave
                div_all_m1=div
                std_m1=std
            else:
                f=interp1d(wave,div)
                div_interp=f(wave0_m1)
                div_all_m1=div_all_m1+div_interp
        div_all_m1=div_all_m1/np.size(m1files)
    else:
        div_all_m1=np.array([])
        wave0_m1=np.array([])
        std_m1=np.array([])

    if(m2bool.any()):
        m2files=np.array(filelist)[m2bool]
        for i,myfile in enumerate(m2files):
            m2file=datadir+myfile
            hdulist=fits.open(m2file)
            data=hdulist[1].data
            wave=data['wave']
            div=data['div']
            std=data['std']
            if(i==0): 
                wave0_m2=wave
                div_all_m2=div
                std_m2=std
            else:
                f=interp1d(wave,div)
                div_interp=f(wave0_m2)
                div_all_m2=div_all_m2+div_interp
        div_all_m2=div_all_m2/np.size(m2files)
    else:
        div_all_m2=np.array([])
        wave0_m2=np.array([])
        std_m2=np.array([])

    if(m3bool.any()):
        m3files=np.array(filelist)[m3bool]
        for i,myfile in enumerate(m3files):
            m3file=datadir+myfile
            hdulist=fits.open(m3file)
            data=hdulist[1].data
            wave=data['wave']
            div=data['div']
            std=data['std']
            if(i==0): 
                wave0_m3=wave
                std_m3=std
                div_all_m3=div
            else:
                f=interp1d(wave,div)
                div_interp=f(wave0_m3)
                div_all_m3=div_all_m3+div_interp
        div_all_m3=div_all_m3/np.size(m3files)
    else:
        div_all_m3=np.array([])
        wave0_m3=np.array([])
        std_m3=np.array([])

    wave=np.concatenate(np.array([wave0_m0,wave0_m1,wave0_m2,wave0_m3]))
    div=np.concatenate(np.array([div_all_m0,div_all_m1,div_all_m2,div_all_m3]))
    std=np.concatenate(np.array([std_m0,std_m1,std_m2,std_m3]))

#Create columns                                                                                                             
    c1  = fits.Column(name='wave', format='D', array=wave)
    c2  = fits.Column(name='div', format='D', array=div)
    c3  = fits.Column(name='std', format='D', array=std)
    coldefs = fits.ColDefs([c1,c2,c3])
    tbhdu = fits.BinTableHDU.from_columns(coldefs)
    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu, tbhdu])
#Write to fits file                                                                                                         
    hdul.writeto(outfile,overwrite=True)
    print('Writing to: ', outfile)
    
    return (wave,div)
