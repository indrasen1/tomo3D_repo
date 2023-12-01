'''
2023-11-15: Indrasen Bhattacharya
Module to implement the tomographic printing work

Class for treating an optimization target

data fields:
- physical parameters for photopolymerization
- target geometry
- pixel function
- projection function in the current iteration
- dose distribution in current iteration
- fabricated volume in the current iteration
- violated volumes of both types
- number of rotations

member functions:
- read in data from a png file
- read in data from an STL file (to be implemented later)
- generate the initial guess using inverse radon transform
     - this will use histogram thresholding to limit the max intensity
     - the number of rotations will be calculated self consistently here
- loss function
- constraint function on the projections
- gradient of linear penalty loss function
- forward model to generate the dose distribution from the current projections
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

from skimage.transform import rescale, radon, iradon
from scipy.signal import resample
from scipy.ndimage import binary_dilation, binary_erosion
#from stlread_utils import stltovoxel

import stltovoxel
from scipy import optimize
from scipy.io import savemat
import shutil

import time
import timeit
from IPython.display import display, clear_output
from matplotlib import animation
import matplotlib as mpl
import sys

mpl.rcParams['animation.ffmpeg_path'] = r'/Users/indrasen/Desktop/ffmpeg'
#mpl.rcParams.update(mpl.rcParamsDefault)

#rc('text', usetex=True)


class CAL():
    
    '''
    Constructor to import STL 
    INPUTS:
        - self object
        - stlName: the extension is stripped internally if it is included
        - lateralRes: the lateral resolution, everything is scaled according to this
        - nDil: number of pixels to dilate for the buffer zone
        
    '''
    def __init__(self, 
                 stlName, 
                 paramDict,
                 extractSTL=True,
                 lateralRes=100, 
                 nDil=3,
                 nAngles=361):
        
        stlName = os.path.splitext(stlName)[0]
        
        stlDir = './stlread_utils/STL_database/'
        stlPath = stlDir + stlName + '.stl'

        pngDir = './stlread_utils/PNG_database/'
        pngDirPath = pngDir + stlName
        
        pngPath = pngDirPath + '/' + stlName + '.png'

        self.stlPath = stlPath
        self.fName = stlName
        self.pngDirPath = pngDirPath
        self.lateralRes = lateralRes
        self.nDil = nDil
        
        
        #if the STL file needs to be converted to PNG series
        if extractSTL:  
            
            print('CREATING PNG DIRECTORY')

            #remove the directory if it exists
            if os.path.isdir(pngDirPath):
                try:
                    shutil.rmtree(pngDirPath)
                except:
                    print('Error while deleting directory')
            
            try:
                os.mkdir(pngDirPath)
            except OSError:
                print ("Creation of the directory %s failed" % pngDirPath)
            else:
                print ("Successfully created the directory %s " % pngDirPath)
            

            print('CONVERTING STL TO PNG')
            stltovoxel.convert_file(stlPath, pngPath, lateralRes)
        
        
        print('IMPORTING PNG IMAGES AS ARRAY')
        self.__readPNG()
        
        #initialize angles used to represent the radon transform
        #potentially change defaults later
        minAngle = 0
        maxAngle = 180
        self.nAngles = paramDict['nAngles']
        #assert (self.nAngles <= 361), "Assertion failed: expected angular resolution to be poorer than 0.5 degrees due to stage speed constraint"
        self.angles = np.linspace( minAngle, maxAngle, paramDict['nAngles'] )
        
        #loss function parameters
        assert (paramDict['order']>=1), "Assertion failed: expected loss function order to be at least 1"
        self.constraintMethod = paramDict['constraintMethod']
        self.p = paramDict['order']
        self.sigmoid_sigma = paramDict['sigmoid_sigma']
        
        #initialize the dose thresholds
        self.dl = paramDict['dl']
        self.dh = paramDict['dh']
        assert (self.dl < self.dh), "Assertion failed: expected lower threshold to be lesser than the upper threshold"
        
        
        #initialize the projection space constraints
        self.gl = paramDict['gl']
        self.gh = paramDict['gh']
        assert (self.gl < self.gh), "Assertion failed: expected the lower limit in projection space to be lesser than the upper limit"
        
        
        #initialize the loss weights
        self.rho1 = paramDict['rho1']
        self.rho2 = paramDict['rho2']
        
        #name to save the final results
        self.saveName = paramDict['saveName']
        self.imgDir = './' + paramDict['saveName'] + '/'
        
        #remove the image directory if it exists
        if os.path.isdir(self.imgDir):
            try:
                shutil.rmtree(self.imgDir)
            except:
                print('Error while deleting directory')
        
        try:
            os.mkdir(self.imgDir)
        except OSError:
            print ("Creation of the directory %s failed" % self.imgDir)
        else:
            print ("Successfully created the directory %s " % self.imgDir)
        

        
    
    #member function to read the png files and create an array
    def __readPNG(self):

        nZ = len(os.listdir(self.pngDirPath))
        #sZ = str(len(str(nZ))+1)
        sZ = str(len(str(nZ)))
        
        nDil = self.nDil

        #read in one image in order to determine the shape of the array
        iH = 0
        path = (self.pngDirPath + '/' + self.fName + '_' + '%0' + sZ + 'd.png')%iH

        #assumes uint8 data
        #scale = 2**8-1
        scale = 1

        imgLoc = np.double( imageio.imread_v2(path) ) / scale
        imgLoc = (imgLoc == 1)
        imgShape = np.array(np.shape(imgLoc))
        imgParity = (1-imgShape%2)

        #expand to nearest odd
        imgLoc = np.pad(imgLoc, ((0, imgParity[0]), (0, imgParity[1])), 'edge')
        imgShape = np.shape(imgLoc)

        #expand till both are of same size
        padPx = (imgShape[1] - imgShape[0])//2
        padPx_0 = np.maximum(0, padPx)
        padPx_1 = np.maximum(0, -padPx)

        imgLoc = np.pad(imgLoc, ((padPx_0, padPx_0), (padPx_1, padPx_1)), 'edge')
        circlePad = np.int32(np.shape(imgLoc)[0]*(np.sqrt(2)-1)/2)
        imgLoc = np.pad(imgLoc, ( (circlePad, circlePad), (circlePad, circlePad) ), 'edge')


        imgShape = np.array(np.shape(imgLoc))

        #dimensions: (nZ, nX, nY, 3)
        imgArr = np.zeros((nZ, imgShape[0], imgShape[1], 3))

        #morphological structuring element
        #nDil = 3
        struct_element = np.ones((nDil, nDil))
        struct_element[0,0] = 0
        struct_element[0, nDil-1] = 0
        struct_element[nDil-1, 0] = 0
        struct_element[nDil-1, nDil-1] = 0

        #read in the array
        for iH in range(nZ):
            path = (self.pngDirPath+ '/' + self.fName + '_' + '%0' + sZ + 'd.png')%iH

            #perform padding operations
            imgLoc = np.double( imageio.imread_v2(path) ) / scale
            imgLoc = (imgLoc == 1)
            imgLoc = np.pad(imgLoc, ((0, imgParity[0]), (0, imgParity[1])), 'edge')
            imgLoc = np.pad(imgLoc, ((padPx_0, padPx_0), (padPx_1, padPx_1)), 'edge')
            imgLoc = np.pad(imgLoc, ( (circlePad, circlePad), (circlePad, circlePad) ), 'edge')

            #use the local image to obtain the core geometry and core background
            imgCore = binary_erosion(imgLoc, structure=struct_element).astype('double')
            imgBG = np.logical_not(binary_dilation(imgLoc, structure=struct_element).astype('double'))
            imgBuffer = np.logical_and( np.logical_not(imgCore) , np.logical_not(imgBG) ).astype('double')

            imgArr[iH, :, :, 0] = imgCore
            imgArr[iH, :, :, 1] = imgBuffer
            imgArr[iH, :, :, 2] = imgBG


        print('SHAPE OF IMAGE ARRAY:')
        print(np.shape(imgArr))

        self.imgArr = imgArr
        
    
    #private method to calculate projections
    def __radonCalc(self, real3D):
        
        nTheta_loc = self.nAngles
        angles_loc = self.angles

        region_shape = np.shape(real3D)
        projections = np.zeros( (region_shape[0], region_shape[1], nTheta_loc) )

        for iZ in range(region_shape[0]):
            projections[iZ,...] = radon(real3D[iZ,...], theta=angles_loc, circle=True)

        return projections
    
    
    #private method to produce reconstruction from projections
    def __reconstruction(self, projections, reconFilter=None):
        
        angles_loc = self.angles
        proj_shape = np.shape(projections)

        region_3D_recon = np.zeros((proj_shape[0], proj_shape[1], proj_shape[1]))

        factor = 1

        if reconFilter is None:
            factor = np.pi/180

        for iZ in range(proj_shape[0]):
            region_3D_recon[iZ,...] = iradon(projections[iZ,...], theta=angles_loc, 
                                             filter_name=reconFilter, circle=True)

        region_3D_recon = region_3D_recon * factor
        
        return region_3D_recon
    
    #private method to plot projections and associated reconstruction
    def __plotProj(self, proj, iZ=0, saveName=None):
        
        orig = self.imgArr[...,0]
        
        recon = self.__reconstruction(proj)

        region_shape = np.shape(orig)

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,4))

        colormap = 'cubehelix'

        img = ax[0].imshow(orig[iZ,...], cmap=colormap, vmin=-0.1, vmax=1.2)
        fig.colorbar(img, ax=ax[0])
        ax[0].set_title('HIGH DOSE TARGET, iZ=' + str(iZ))
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')

        img = ax[1].imshow(np.swapaxes(proj[iZ,...], 0, 1), cmap='inferno')
        fig.colorbar(img, ax=ax[1])
        ax[1].set_title('PROJECTIONS, iZ=' + str(iZ))
        ax[1].set_aspect(region_shape[1]/len(self.angles))
        ax[1].set_xlabel('RADIUS')
        ax[1].set_ylabel('ANGLE')

        recon_shape = np.shape(recon)

        img = ax[2].imshow(recon[iZ,...], cmap=colormap, vmin=-0.1, vmax=1.2)
        fig.colorbar(img, ax=ax[2])
        ax[2].set_title('RECONSTRUCTION, iZ=' + str(iZ))
        ax[2].set_xlabel('X')
        ax[2].set_ylabel('Y')

        if saveName is not None:
            plt.savefig(saveName)
            
        plt.show()
        
        
    #private method to generate geometry from dose
    #take the normalized dose and generate geometries: update the object members
    def __geometries_fromDose(self, dose):

        R1 = np.zeros_like(dose)
        R2 = np.zeros_like(dose)
        V1 = np.zeros_like(dose)
        V2 = np.zeros_like(dose)
        
        regions = self.imgArr
        
        dl = self.dl
        dh = self.dh

        region_shape = np.shape(dose)
        regions_logical = (regions==1)

        for iZ in range(region_shape[0]):
            R1[iZ,...] = (dose[iZ,...]>=dh)
            R2[iZ,...] = (dose[iZ,...]<=dl)

        V1 = np.logical_and( regions[...,0], np.logical_not(R1))
        V2 = np.logical_and( regions[...,2], np.logical_not(R2))
        
        self.R1 = R1
        self.R2 = R2
        self.V1 = V1
        self.V2 = V2
        
        
    #display dose distribution and violations arising from a particular 3D dose distribution
    #the Z-slice used to display is also an input
    def __plotRegions(self, dose, iZ=0, saveName=None):
        
        self.__geometries_fromDose(dose)
        
        R1 = self.R1
        R2 = self.R2
        V1 = self.V1
        V2 = self.V2
        

        region_shape = np.shape(R1)

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,4))

        colormap = 'cubehelix'


        img = ax[0].imshow(dose[iZ,...], cmap=colormap, vmin=-0.1, vmax=1.2)
        ax[0].imshow(V1[iZ,...], cmap=colormap, vmin=-0.1, vmax=1.2, alpha=0.2)
        fig.colorbar(img, ax=ax[0])
        ax[0].set_title('DOSE DISTRIBUTION, iZ=' + str(iZ))
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')

        img = ax[1].imshow(V1[iZ,...], cmap=colormap, vmin=-0.1, vmax=1.2)
        fig.colorbar(img, ax=ax[1])
        ax[1].set_title('TYPE1 VIOLATION, iZ=' + str(iZ))
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')

        img = ax[2].imshow(V2[iZ,...], cmap=colormap, vmin=-0.1, vmax=1.2)
        fig.colorbar(img, ax=ax[2])
        ax[2].set_title('TYPE2 VIOLATION, iZ=' + str(iZ))
        ax[2].set_xlabel('X')
        ax[2].set_ylabel('Y')
        
        if saveName is not None:
            plt.savefig(saveName)

        plt.show()
        
    
    #private methods to calculate the Ram-Lak kernels
    def __Ram_Lak(self, nP=101):

        rl_filter = np.zeros( (nP,) )
        center = nP//2

        nArr = np.linspace(-center, center, nP).astype('int')
        parity_mask = (nArr%2 == 1)

        rl_filter[parity_mask] = -1/(np.pi * nArr[parity_mask])**2
        rl_filter[center] = 0.25

        #default radon implementation is restricted over 180 degrees
        factor = 2

        return np.expand_dims(rl_filter, axis=-1)*factor
    
    
    #function to generate the Shepp-Logan kernel
    def __Shepp_Logan(self, nP=101):

        sl_filter = np.zeros( (nP,) )
        center = nP//2

        nArr = np.linspace(-center, center, nP).astype('int')

        sl_filter = -2/(np.pi**2) * 1/(4*nArr**2 - 1)

        factor = 2

        return np.expand_dims(sl_filter, axis=-1)*factor
    
    
    #generate filtered projections
    def __returnFiltered(self, projections, RamLak):
        
        angles_loc = self.angles

        proj_shape = np.shape(projections)
        filtered_proj = np.zeros_like(projections)

        if RamLak:
            fbp_filter = self.__Ram_Lak(nP=proj_shape[1])
        else:
            fbp_filter = self.__Shepp_Logan(nP=proj_shape[1])

        #this will be un-done by the reconstruction
        factor = 180/np.pi

        filtered_proj = np.apply_along_axis(lambda m: np.convolve(m, np.squeeze(fbp_filter), mode='same'), axis=1, arr=projections)

        return filtered_proj*factor
    
    
    #return the sigmoid
    def __sigmoid(self, x, sigma):
        return 1/(1 + np.exp(x/sigma))

    #return the derivative of the sigmoid
    def __derivSigmoid(self, x, sigma):
        y = self.__sigmoid(x, sigma)
        return y*(y-1)/sigma
    
    #next step: public function to generate unfiltered, RL filtered and SL filtered initializations
    #have a public function to also plot these
    def initializeProjections(self, displZ = None):
        
        projLoc = self.__radonCalc(real3D=self.imgArr[...,0])
        angles = self.angles
        
        filtered_loc_rl = self.__returnFiltered(projLoc, RamLak=True)
        filtered_loc_sl = self.__returnFiltered(projLoc, RamLak=False)

        #determine the filtered reconstructions
        recon_None = self.__reconstruction(projections=projLoc)
        recon_RL = self.__reconstruction(projections=filtered_loc_rl)
        recon_SL = self.__reconstruction(projections=filtered_loc_sl)

        #constrained reconstructions
        gh = self.gh
        gl = self.gl
        
        none_masked = np.logical_and( (projLoc>=gl), (projLoc<=gh) ) * projLoc
        rl_masked = np.logical_and( (filtered_loc_rl>=gl), (filtered_loc_rl<=gh) ) * filtered_loc_rl
        sl_masked = np.logical_and( (filtered_loc_sl>=gl), (filtered_loc_sl<=gh) ) * filtered_loc_sl
        
        recon_none_pos = self.__reconstruction(projections=none_masked, reconFilter=None)
        recon_RL_pos = self.__reconstruction(projections=rl_masked, reconFilter=None)
        recon_SL_pos = self.__reconstruction(projections=sl_masked, reconFilter=None)
        
        self.none_masked = none_masked
        self.rl_masked = rl_masked
        self.sl_masked = sl_masked
        
        if displZ is None:
            #displZ = np.array([0, 10, 20, 40, 80, 100, 110, 120])*(np.shape(recon_none_pos)[0]-1)/120 
            displZ = np.array([25, 50, 75])*(np.shape(recon_none_pos)[0]-1)/100
            displZ = displZ.astype('int')

            
        
        #display the dose distribution for a specific value of displZ
        print('DISPLAYING NO-FILTER RECONSTRUCTIONS:')
        for iZ_loc in displZ:
            self.__plotRegions(dose=recon_none_pos, iZ=iZ_loc)
            
        print('DISPLAYING SHEPP-LOGAN FILTERED RECONSTRUCTIONS:')
        for iZ_loc in displZ:
            self.__plotRegions(dose=recon_SL_pos, iZ=iZ_loc)
            
            
            
    #include public member functions to return the loss and loss gradient
    #also include a public member function for the callback
    
    #loss function: eliminated the regions argument since it is an object member
    def loss(self, proj_iter):

        #determine shape and angular sampling to be used
        #this part would be constructor-ized in a class-based implementation
        #proj_shape = np.shape(proj_iter)
        
        regions = self.imgArr

        iZ = np.shape(regions)[0]
        iR = np.amax(np.shape(regions)[1:2])
        iTheta = self.nAngles
        proj_shape = (iZ, iR, iTheta)
        proj_iter = np.reshape(proj_iter, proj_shape)

        rho1 = self.rho1
        rho2 = self.rho2
        dl_param = self.dl
        dh_param = self.dh
        angles = self.angles
        
        assert(len(angles) == proj_shape[-1]),"Angles and projections are inconsistent!"


        #calculate the dose distribution
        dose_3D_iter = self.__reconstruction(projections=proj_iter)

        #obtain the violated regions from the dose
        #note the fixed dose: might be good to constructor-ize later
        self.__geometries_fromDose(dose=dose_3D_iter)
        
        V1_loc_iter = self.V1
        V2_loc_iter = self.V2
        
        #note that R1 and R2 are not changing with iterates
        virtualDose_target = self.imgArr[...,0].astype('double')
        virtualDose_bg = self.imgArr[...,2].astype('double')

        #define the integrand for the loss
        if self.constraintMethod:
            loss_integrand = rho1*np.power(dose_3D_iter-dl_param, self.p)*V2_loc_iter + rho2*np.power(dh_param-dose_3D_iter, self.p)*V1_loc_iter
        else:
            sigmoid_sigma = self.sigmoid_sigma
            virtualThresh_target = self.dh + sigmoid_sigma
            virtualThresh_bg = 0
            smooth_target = self.__sigmoid(dose_3D_iter - virtualThresh_target, sigmoid_sigma) * virtualDose_target
            smooth_bg = (1 - self.__sigmoid(dose_3D_iter - virtualThresh_bg, sigmoid_sigma)) * virtualDose_bg
            
            loss_integrand = rho1*np.power( np.abs(  smooth_bg - virtualDose_bg  ), self.p ) + rho2*np.power( np.abs(  smooth_target - virtualDose_target  ), self.p )

        loss_iter = np.sum(loss_integrand)
        

        return loss_iter        
    
    
    #loss gradient
    def loss_gradient(self, proj_iter):
        
        regions = self.imgArr
        
        iZ = np.shape(regions)[0]
        iR = np.amax(np.shape(regions)[1:2])
        iTheta = self.nAngles
        proj_shape = (iZ, iR, iTheta)
        proj_iter = np.reshape(proj_iter, proj_shape)

        rho1 = self.rho1
        rho2 = self.rho2
        dl_param = self.dl
        dh_param = self.dh
        angles = self.angles

        #calculate the dose distribution
        dose_3D_iter = self.__reconstruction(projections=proj_iter)

        #obtain the violated regions from the dose
        #note the fixed dose: might be good to constructor-ize later
        self.__geometries_fromDose(dose=dose_3D_iter)

        V1_loc_iter = self.V1
        V2_loc_iter = self.V2
        
        #note that R1 and R2 are not changing with iterates
        virtualDose_target = self.imgArr[...,0].astype('double')
        virtualDose_bg = self.imgArr[...,2].astype('double')


        #determine the gradient dose
        #this quantity will be radon-transformed to determine the per pixel loss gradient
        
        
        #the following takes way too long

        if self.constraintMethod:
            gradient_dose_iter = rho1*np.double(V2_loc_iter)*(self.p)*np.power( dose_3D_iter-dl_param, self.p-1 ) - rho2*np.double(V1_loc_iter)*(self.p)*np.power( dh_param-dose_3D_iter, self.p-1 )
        else:
            sigmoid_sigma = self.sigmoid_sigma
            virtualThresh_target = self.dh + sigmoid_sigma
            virtualThresh_bg = 0
            smooth_target = self.__sigmoid(dose_3D_iter - virtualThresh_target, sigmoid_sigma) * virtualDose_target
            smooth_bg = (1 - self.__sigmoid(dose_3D_iter - virtualThresh_bg, sigmoid_sigma)) * virtualDose_bg
            
            grad_term_target = virtualDose_target * self.__derivSigmoid(dose_3D_iter - virtualThresh_target, sigmoid_sigma) * np.sign( smooth_target - virtualDose_target ) * self.p * np.power( np.abs( smooth_target - virtualDose_target ), self.p-1)
            grad_term_bg = -virtualDose_bg * self.__derivSigmoid(dose_3D_iter - virtualThresh_bg, sigmoid_sigma) * np.sign( smooth_bg - virtualDose_bg ) * self.p * np.power( np.abs( smooth_bg - virtualDose_bg ), self.p-1)
            
            gradient_dose_iter = rho1 * grad_term_bg + rho2 * grad_term_target

        #determine the projection space gradient using the projection operation: __radonCalc
        gradient_iter = self.__radonCalc(real3D=gradient_dose_iter)

        #flatten the vector
        gradient_iter = np.reshape(gradient_iter, np.prod(proj_shape))

        factor = np.pi/180

        return gradient_iter*factor
    
    
    #loss callback to be displayed
    def displLoss(self, proj_iter, 
                  zLoc=60, start=False):

        if not start:
            print('Finished an iteration')

        localLoss = self.loss(proj_iter)
        
        #make sure to initialize lossArr in call to optimization
        self.lossArr.append(localLoss)

        print('Loss = ' + str(localLoss))

        proj_shape = np.shape(self.none_masked)
        angles = self.angles
        proj_iter = np.reshape(proj_iter, proj_shape)
        
        result_recon = self.__reconstruction(projections=proj_iter)
        
        #generate the 3D geometries and violated regions
        self.__geometries_fromDose(dose=result_recon)
        
        #call a function to update the number of violating pixels
        
        violations_vs_Z = np.sum( self.V1 + self.V2, axis = (1, 2) )
        zLoc = np.argmax( violations_vs_Z )
        
        
        #plot the regions
        self.__plotRegions(dose=result_recon, iZ=zLoc)
        
        
    #function to run the optimization
    #finish and save the result in the object
    def runOpt(self, options_loc=None, displZ=None):
        
        imgName = self.saveName
        
        if options_loc is None:
            options_loc = {}
            options_loc['maxiter'] = 50
            options_loc['ftol'] = 1e-12
            options_loc['disp'] = True
            options_loc['maxcor'] = 10
            options_loc['maxls'] = 30
        
        self.lossArr = []
        
        self.initializeProjections(displZ)
        
        x0_loc = self.none_masked
        proj_shape = np.shape(x0_loc)
        x0_loc = np.reshape(x0_loc, (np.prod(proj_shape)))

        bounds_loc = [[[(self.gl,self.gh) for k in range(proj_shape[2]) ] for j in range(proj_shape[1]) ]  for i in range(proj_shape[0]) ]
        bounds_loc = np.reshape(bounds_loc, (np.prod(proj_shape), 2))

        self.displLoss(x0_loc, start=True)

        result = optimize.minimize(fun=self.loss, 
                                   x0=x0_loc, 
                                   callback=self.displLoss,
                                   method='L-BFGS-B', 
                                   jac=self.loss_gradient, 
                                   bounds=bounds_loc, 
                                   options=options_loc)
        
        #display result
        print(result)
        
        
        self.finalProj = np.reshape(result.x, proj_shape)
        self.finalDose = self.__reconstruction(projections=self.finalProj)

        self.__geometries_fromDose(dose=self.finalDose)
        
        totalPixels = np.sum( self.imgArr[...,0] )
        totalViolations = np.sum(self.V1) + np.sum(self.V2)
        
        print('RESULTS:')
        print('TOTAL PIXELS = ' + str(totalPixels))
        print('TOTAL VIOLATIONS = ' + str(totalViolations))
        print('VIOLATING PIXEL FRACTION = ' + str(totalViolations/totalPixels)[:6])
        
        if displZ is None:
            #displZ = np.array([0, 10, 20, 40, 80, 100, 110, 120])*(np.shape(self.finalProj)[0]-1)/120 
            displZ = np.array([25, 50, 75])*(np.shape(self.finalProj)[0]-1)/100 
            displZ = displZ.astype('int')
        
        for iZ_loc in displZ:
            self.__plotRegions(dose=self.finalDose, iZ=iZ_loc,
                               saveName=self.imgDir + 'finalViolations_Z=' + str(iZ_loc) + '_' + imgName)
        
        
        saveDict = {}
        saveDict['dose_3D_optimized'] = self.finalDose
        saveDict['dose_3D_init_none'] = self.__reconstruction(projections=self.none_masked)
        saveDict['dose_3D_init_SL'] = self.__reconstruction(projections=self.sl_masked)
        saveDict['regions'] = self.imgArr
        saveDict['proj_optimized'] = self.finalProj
        saveDict['proj_init'] = self.none_masked
        
        
        savemat(self.imgDir + self.saveName + '.mat', saveDict)
        
        self.__displLearning()
        
        self.__niceDispl()
        
        
    #function to display projections after convergence
    def __niceDispl(self):
        angles_loc = np.array([0, 30, 60, 90, 120, 150])*self.nAngles/361
        angles_loc = angles_loc.astype(int)
        nAngles = len(angles_loc)
        fig, ax = plt.subplots(nrows=3, ncols=nAngles, figsize=(4.5*nAngles, 11), facecolor='w')

        colormap='gist_stern'
        vmax_loc=0 #np.amax(finalProj)
        vmin_loc=255 #np.amin(finalProj)

        for iTheta in range(nAngles):

            titleStr = '{angle}'.format(angle=angles_loc[iTheta])

            iRow=0
            localImg = ax[iRow,iTheta].imshow(np.flipud(self.none_masked[...,angles_loc[iTheta]] ), cmap=colormap, vmin=vmin_loc, vmax=vmax_loc)
            #ax[iRow,iTheta].set_xticks([])
            #ax[iRow,iTheta].set_yticks([])
            #ax[iRow,iTheta].set_title(r'INITIALIZED, $\mathbf{\theta}$ = ' + titleStr + '$^\circ$')
            ax[iRow,iTheta].set_title(r'INITIALIZED')
            fig.colorbar(localImg, ax=ax[iRow,iTheta])

            iRow=2
            localImg = ax[iRow,iTheta].imshow(np.flipud(self.sl_masked[...,angles_loc[iTheta]] ), cmap=colormap, vmin=vmin_loc, vmax=vmax_loc)
            #ax[iRow,iTheta].set_xticks([])s
            #ax[iRow,iTheta].set_yticks([])
            #ax[iRow,iTheta].set_title(r'SHEPP LOGAN, $\mathbf{\theta}$ = ' + titleStr + '$^\circ$')
            ax[iRow,iTheta].set_title(r'SHEPP LOGAN')
            fig.colorbar(localImg, ax=ax[iRow,iTheta])

            iRow=1
            localImg = ax[iRow,iTheta].imshow(np.flipud(self.finalProj[...,angles_loc[iTheta]] ), cmap=colormap, vmin=vmin_loc, vmax=vmax_loc)
            #ax[iRow,iTheta].set_xticks([])
            #ax[iRow,iTheta].set_yticks([])
            #ax[iRow,iTheta].set_title(r'OPTIMIZED, $\mathbf{\theta}$ = ' + titleStr + '$^\circ$')
            ax[iRow,iTheta].set_title(r'OPTIMIZED')
            fig.colorbar(localImg, ax=ax[iRow,iTheta])


        plt.savefig(self.imgDir + self.saveName + '_proj.png')    
        plt.show()
        
        
    #function to display pre- and post-convergence projections along with the learning curve
    def __displLearning(self):
        
        iZ = 20
        x0_loc = self.none_masked
        proj_shape = np.shape( self.none_masked )
        
        
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

        x0_loc = np.reshape(x0_loc, proj_shape)

        img = ax[0].imshow(np.swapaxes(x0_loc[iZ,...], 0, 1), cmap='inferno')
        fig.colorbar(img, ax=ax[0])
        ax[0].set_title('INITIALIZATION, iZ=' + str(iZ))
        ax[0].set_aspect(proj_shape[1]/proj_shape[2])
        ax[0].set_xlabel('RADIUS')
        ax[0].set_ylabel('ANGLE')


        ax[1].plot(self.lossArr)
        ax[1].grid(True)
        ax[1].set_yscale('log')
        ax[1].set_ylabel('LOSS')
        ax[1].set_xlabel('EPOCHS')
        ax[1].set_title('OPTIMIZATION HISTORY')


        img = ax[2].imshow(np.swapaxes(self.finalProj[iZ,...], 0, 1), cmap='inferno')
        fig.colorbar(img, ax=ax[2])
        ax[2].set_title('CONVERGED PROJECTIONS, iZ=' + str(iZ))
        ax[2].set_aspect(proj_shape[1]/proj_shape[2])
        ax[2].set_xlabel('RADIUS')
        ax[2].set_ylabel('ANGLE')

        plt.savefig(self.imgDir + 'finalResult_' + self.saveName)

        plt.show()
        
        
    #function to generate an animation of the desired 3D array 
    #the first axis is taken to be the axis of variation (or time axis)
    
    def generateAnimation( self, arr_loc, cmapName='RdPu_r', fName='test.mp4' ):
        #function to generate an animation from the fully computed result
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), layout='compressed')

        fig.set_constrained_layout_pads()

        #sLoc = species

        ims = []

        for t in range( np.shape( arr_loc )[0] ):
            im = ax.imshow( arr_loc[t, ...], animated=True, cmap=cmapName )
            ax.set_axis_off()
            if t==0:
                ax.imshow( arr_loc[t, ...], cmap=cmapName )
                ax.set_axis_off()
            ims.append([im])

        #fig.set_tight_layout(tight=True)

        ani = animation.ArtistAnimation( fig, ims, interval=50, blit=True, repeat_delay=1000 )

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='whodat'), bitrate=1800)


        ani.save(filename=fName, writer=writer)
        plt.show()
        
        
        