# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 10:50:32 2014

@author: yung
"""

import numpy as np
import matplotlib.pyplot as plt
import libtiff
#import mahotas as mh
#from mpl_toolkits.mplot3d import Axes3D
#import PIL.Image as Image
#import matplotlib.animation as animation 
from sub import polygon, point

plt.close("all")

filePath='/Users/yung/virtualbox share folder/070914QCSE G H/'
fileName='G_400_3'

mov = libtiff.TiffFile(filePath+fileName+'.tif')
movie = mov.get_tiff_array()
movie=np.array(movie[:,:,:],dtype='d')

backGND_corr=1           # 1 = apply correction, else = no correction
Photobleaching_corr=1    # 1 = apply correction, else = no correction
frame=len(movie[:,0,0])
row=len(movie[0,:,0])
col=len(movie[0,0,:])
dt=0.125
frame_start=3
t=frame*dt
T = np.arange(0,t,dt)
movie[0:frame_start,:,:]=movie[frame_start,:,:]
scan=2 # extract 5x5 pixels around QD

abs_I_diff=np.zeros((row, col))
for i in range(frame-1):       
    c=movie[i,:,:]
    d=movie[i+1,:,:]
    abs_I_diff=abs_I_diff+np.absolute(d-c)  

fig, ax = plt.subplots()
ax.imshow(abs_I_diff, vmin=abs_I_diff.min(), vmax=abs_I_diff.max(),cmap='gray')
plt.title('Differential image')

"""
Background and photobleaching correction
"""

print 'What is background?'
bg, bg_3d = polygon.mean_polygon(movie, abs_I_diff, ax, fig) 
bg_fit=np.polyfit(T, bg, 7) #fitting to background
p=np.polyval(bg_fit,T)

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

window_size=5

#p=movingaverage(bg,window_size)
bg_3d=np.tile(p[:,np.newaxis,np.newaxis],(1,row,col))
movie_bgcr=movie[:,:,:]-bg_3d
movie_bgcr1=np.sum(np.sum(movie_bgcr,axis=1),axis=1)/(row*col)
movie_pb=movingaverage(movie_bgcr1,window_size)
pb_constant=np.polyfit(T[frame_start:len(T)-window_size:1],movie_pb[frame_start:len(T)-window_size:1],1)
pbleach=np.polyval(pb_constant,T)
pbc=pb_constant[1]/pbleach

fig,(ax,ax2,ax3)=plt.subplots(3,1,sharex=True)

line_sbg=ax.plot(T,(np.sum(np.sum(bg_3d,axis=1),axis=1)/(row*col)),'m',label='smoothen bg')
line_bg=ax.plot(T,bg,'c',label='bg')
ax.set_title('Background')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, bbox_to_anchor=(0.93, 1), loc=2, borderaxespad=0, fontsize=12)

line_movie=ax2.plot(T,(np.sum(np.sum(movie,axis=1),axis=1)/(row*col)),'g',label='before bgcr')
line_movie_bgcr=ax2.plot(T,movie_bgcr1,'y',label='after bgcr')
ax2.set_title('Background correction')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, bbox_to_anchor=(0.93, 1), loc=2, borderaxespad=0, fontsize=12)


line_smoothen, = ax3.plot(T[frame_start:len(T)-window_size:1],movie_pb[frame_start:len(T)-window_size:1], label="Smoothen I")
line_pbleaching, = ax3.plot(T[frame_start:len(T)-window_size:1],pbleach[frame_start:len(T)-window_size:1], label="Photobleaching")
line_pb_correct_I, = ax3.plot(T[frame_start:len(T)-window_size:1],np.multiply(movie_pb,pbc)[frame_start:len(T)-window_size:1], label="P.B. corrected I")
ax3.set_title('Photobleaching correction')
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels,bbox_to_anchor=(0.93, 1), loc=2, borderaxespad=0, fontsize=12)

#plt.legend([line_smoothen, line_pbleaching, line_pb_correct_I,line_movie,line_movie_bgcr,line_sbg, line_bg], ["Smoothen I", "Photobleaching", "P.B. corrected I","before bgcr","after bgcr","Smoothen bg", "Background"], loc=5)
plt.xlabel('time [s]')
plt.show()
    
if backGND_corr == 1:
    if Photobleaching_corr == 1:
        mov_f=np.zeros((frame,row,col))
        for i in range(frame):
            mov_f[i,:,:]=movie_bgcr[i,:,:]*pbc[i] 
    else: 
        mov_f=movie_bgcr
else:
    mov_f=movie

#fig=plt.figure()
#ims = []
#for i in range(frame):
#    im=plt.imshow(mov_f[i,:,:],vmin=mov_f.min(),vmax=mov_f.max(),cmap='seismic')
#    ims.append([im]) 
#ani = animation.ArtistAnimation(fig, ims, interval=dt, blit=True,
#    repeat_delay=1000)
#plt.title('movie')  

abs_I_diff=np.zeros((row,col))
for i in range(frame-1):       
    c=mov_f[i,:,:]
    d=mov_f[i+1,:,:]
    abs_I_diff=abs_I_diff+np.absolute(d-c)

       
"""
Define points (QDs) of interest, and their nearest peak position
"""


fig, ax = plt.subplots()
im = ax.imshow(abs_I_diff, vmin=abs_I_diff.min(), vmax=abs_I_diff.max(),cmap='seismic')

pts = point.pIO(mov_f, ax, fig)
pts = np.array(pts)
pts_new = point.localmax(abs_I_diff, pts, ax, fig)
npoint = np.size(pts_new[:,0])


"""
Extracting mean intensity of 5 X 5 mask in your points of interest and
Calculate delta F between voltage on and off
"""
spot_intensity=np.zeros((frame,npoint))
for n in range(npoint):
    for i in range(frame):
        temp = point.mask(mov_f[i,:,:], pts_new[n,:], scan)
        spot_intensity[i,n] = temp.mean()


fig, axarr = plt.subplots(npoint,2)
for n in range (npoint):    
    axarr[n,0].plot(spot_intensity[frame_start:,n])
    std5 = np.std(spot_intensity[frame_start:,n],axis=0,ddof=1,dtype='d')/5    
    thre_constant = spot_intensity[frame_start:,n].mean()-std5      
    threshold = np.tile(thre_constant,frame)  
    axarr[n,0].plot(threshold,'r')    
    Von = np.array(spot_intensity[::2])
    Voff = np.array(spot_intensity[1::2])           
    Von_thre = np.array([Von[i,n] for i in range(frame/2) if Von[i,n] > threshold[i]])
    Voff_thre = np.array([Voff[i,n] for i in range(frame/2) if Voff[i,n] > threshold[i]])
    F = (np.sum(Von_thre)+np.sum(Voff_thre))/(len(Von_thre)+len(Voff_thre))    
    dFF = (Voff_thre.mean()-Von_thre.mean())/F*100
#    for i in range (frame-1):
#        dFF=0
#        if spot_intensity[i,n] > thre_constant:
#            if spot_intensity[i+1,n] > thre_constant:
#                F=(spot_intensity[i,n]+spot_intensity[i+1,n])/2
#                dF=(spot_intensity[i+1,n]-spot_intensity[i,n])
#            else:
#                F, dF = 1, 0
#        else:
#            F, df = 1, 0
#        dFF=dFF+dF/F
#    print dFF 
    axarr[n,0].annotate(r'$\Delta F/F=${}%'.format(round(dFF,3)),xy=(0,0), xytext=(0,0.85), xycoords='axes fraction')  
    axarr[n,0].set_ylim([spot_intensity[frame_start:,n].min(), spot_intensity[frame_start:,n].max()])
    axarr[n,0].set_xlabel('frame')
      
    blink_on=[]
    for i in range (frame_start, frame):
        if spot_intensity[i,n] > threshold[i]:
            blink_on=np.append(blink_on,spot_intensity[i,n])
    axarr[n,1].hist(Von_thre, bins=(len(blink_on)/10), color='y', histtype='stepfilled',alpha=0.5, label='Von')    
    axarr[n,1].hist(Voff_thre, bins=(len(blink_on)/10), color='g', histtype='stepfilled',alpha=0.5, label='Voff')   
    handles, labels = axarr[n,1].get_legend_handles_labels()    
    axarr[n,1].legend(handles, labels,bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0, fontsize=12)
    
    
plt.show()

#from matplotlib.backends.backend_pdf import PdfPages
#pp = PdfPages(filePath+fileName+'analysis.pdf')
plt.savefig(filePath+fileName+'analysis.pdf', format='pdf')
print 'Analysis result is saved to'+filePath

